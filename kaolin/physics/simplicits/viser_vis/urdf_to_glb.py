#!/usr/bin/env python3
"""
urdf_to_glb.py — build a single GLB mesh from a URDF (or .urdf.xacro) file.

Parses every <visual> element in the URDF, resolves the mesh filename relative
to a user-supplied mesh directory, applies the joint + visual origin transforms,
and exports the whole assembly as one GLB file ready for Viser's add_glb().

Supports:
  • plain URDF  (.urdf)
  • xacro files (.urdf.xacro) — loaded as plain XML (no xacro macro expansion);
    properties / xacro:property tags are parsed for simple numeric substitution;
    macro parameters (e.g. ${prefix}) are injected via --xacro_args
  • package:// URI scheme (resolved against --mesh_dir, or the URDF's own directory)
  • per-link colour override via --colors  (link_name:#rrggbb, ...)

Usage examples
--------------
# Full URDF (resolves package:// against mesh dir automatically):
python urdf_to_glb.py \
    --urdf   /path/to/robot.urdf \
    --mesh_dir /path/to/meshes \
    --output /path/to/out.glb

# Cadière Forceps from dvrk_model (xacro, partial — no macro expansion):
python urdf_to_glb.py \
    --urdf    /path/to/psm_tool_caudier.urdf.xacro \
    --mesh_dir /home/nan/Desktop/datasets/dvrk_meshes \
    --output   /home/nan/Desktop/datasets/dvrk_meshes/caudier_from_urdf.glb \
    --colors   caudier_jaw1:#c8c8c8,caudier_jaw2_mimic_1:#dcb464,caudier_jaw2_mimic_2:#dcb464

# Include only specific links:
python urdf_to_glb.py --urdf robot.urdf --mesh_dir meshes --output out.glb \
    --include_links link_jaw1 link_jaw2 link_shaft

# Canonical correction (put a named link's visual origin at mesh origin):
python urdf_to_glb.py ... --canonical_link caudier_jaw2_mimic_1

# Post-transform: rotate the whole assembly (e.g. canonical frame fixup):
python urdf_to_glb.py ... --post_rpy "0,1.5708,0" --post_xyz "0,-0.0199,0"
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


# ── Transform helpers ─────────────────────────────────────────────────────────

def rpy_xyz_to_T(rpy, xyz) -> np.ndarray:
    """Build a 4×4 homogeneous transform from (roll,pitch,yaw) and translation."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def _snap_to_pi_multiples(v: float, tol: float = 1e-4) -> float:
    """Snap a value close to a multiple of π/2 to the exact multiple."""
    for k in range(-8, 9):
        exact = k * np.pi / 2
        if abs(v - exact) < tol:
            return exact
    return v


def parse_rpy_xyz(origin_elem):
    """
    Parse an XML <origin rpy="r p y" xyz="x y z"/> element.
    Returns (rpy_array, xyz_array).  Defaults to zero if attribute missing.
    Values close to multiples of π/2 (within 1e-4 rad) are snapped to exact
    values so that floating-point representations match Python np.pi literals.
    """
    rpy = np.zeros(3)
    xyz = np.zeros(3)
    if origin_elem is None:
        return rpy, xyz
    rpy_str = origin_elem.get("rpy", "0 0 0")
    xyz_str = origin_elem.get("xyz", "0 0 0")
    rpy = np.array([_snap_to_pi_multiples(float(v)) for v in rpy_str.split()])
    xyz = np.array([float(v) for v in xyz_str.split()])
    return rpy, xyz


# ── xacro property substitution (best-effort, no macro expansion) ─────────────

_PROP_RE = re.compile(r"\$\{([^}]+)\}")

def _eval_expr(expr: str, props: dict) -> str:
    """
    Evaluate a simple arithmetic expression that may reference xacro properties.
    E.g.  "${M_PI / 2}"  →  "1.5707963..."
    """
    # Substitute known properties
    for k, v in props.items():
        expr = expr.replace(k, str(v))
    # Substitute M_PI
    expr = expr.replace("M_PI", str(np.pi))
    try:
        result = eval(expr, {"__builtins__": {}}, {"pi": np.pi})  # noqa: S307
        return str(result)
    except Exception:
        return expr  # return as-is if evaluation fails


def substitute_xacro(text: str, props: dict) -> str:
    """Replace all ${...} expressions in text using the property dict."""
    def repl(m):
        return _eval_expr(m.group(1), props)
    return _PROP_RE.sub(repl, text)


def collect_xacro_properties(root, extra: dict | None = None) -> dict:
    """
    Collect all <xacro:property name="..." value="..."/> tags into a dict.
    extra: pre-seeded values (e.g. macro params from --xacro_args).
    Values that are themselves expressions referencing other properties are
    resolved iteratively (up to 10 passes).
    """
    props = {"M_PI": str(np.pi), "pi": str(np.pi)}
    if extra:
        props.update(extra)
    for elem in root.iter():
        if elem.tag.endswith("property"):
            name  = elem.get("name",  "")
            value = elem.get("value", "")
            if name:
                props[name] = value
    # Iterative resolution
    for _ in range(10):
        changed = False
        for k, v in props.items():
            new_v = substitute_xacro(v, props)
            if new_v != v:
                props[k] = new_v
                changed = True
        if not changed:
            break
    return props


# ── Mesh URI resolution ───────────────────────────────────────────────────────

def resolve_mesh_path(uri: str, mesh_dir: str, urdf_dir: str,
                      mesh_map: dict | None = None) -> str | None:
    """
    Resolve a mesh URI to an absolute path.

    Search order for package:// URIs:
      1. mesh_dir / basename(path)          — user-supplied mesh dir, flat lookup
      2. mesh_dir / sub/path/to/mesh.stl    — user-supplied mesh dir, full sub-path
      3. urdf_dir / ../../ ... / path       — walk up from the URDF and try the
                                              package-relative path (works when the
                                              URDF lives inside the package repo)
    """
    # mesh_map: remap URDF basenames to local filenames before any search
    raw_basename = os.path.basename(uri.split("/")[-1])
    if mesh_map and raw_basename in mesh_map:
        mapped = mesh_map[raw_basename]
        candidate = mapped if os.path.isabs(mapped) else os.path.join(mesh_dir, mapped)
        if os.path.exists(candidate):
            return candidate

    if uri.startswith("package://"):
        rest = uri[len("package://"):]        # "pkg_name/sub/path/mesh.stl"
        pkg_rel = rest.split("/", 1)[-1]      # "sub/path/mesh.stl"  (drop package name)
        basename = os.path.basename(pkg_rel)  # "mesh.stl"

        # 1. flat lookup in mesh_dir
        c1 = os.path.join(mesh_dir, basename)
        if os.path.exists(c1):
            return c1
        # 2. sub-path inside mesh_dir
        c2 = os.path.join(mesh_dir, pkg_rel)
        if os.path.exists(c2):
            return c2
        # 3. walk up from urdf_dir looking for the pkg_rel path
        search = urdf_dir
        for _ in range(6):
            c3 = os.path.join(search, pkg_rel)
            if os.path.exists(c3):
                return c3
            parent = os.path.dirname(search)
            if parent == search:
                break
            search = parent
        return None

    if uri.startswith("file://"):
        return uri[len("file://"):]
    if os.path.isabs(uri):
        return uri
    # Relative: try mesh_dir first, then urdf_dir
    c = os.path.join(mesh_dir, uri)
    if os.path.exists(c):
        return c
    return os.path.join(urdf_dir, uri)


# ── URDF parsing ──────────────────────────────────────────────────────────────

def parse_urdf_visuals(
    urdf_path: str,
    mesh_dir: str,
    include_links: list[str] | None,
    xacro_args: dict | None = None,
    no_chain: bool = False,
    joint_links: list[str] | None = None,
    mesh_map: dict | None = None,
):
    """
    Parse the URDF/xacro and return a list of visual entries:
        [{"link": str, "T": np.ndarray(4,4), "mesh": str, "scale": np.ndarray(3)}, ...]

    no_chain: if True, skip kinematic chain — each visual uses only its own
              <visual><origin> transform.  Combine with joint_links to apply
              the direct parent joint for specific links only.

    joint_links: when no_chain=True, these links additionally get their
                 immediate parent joint transform applied (depth=1).
                 Example: "tool_wrist_caudier_link_2_right" needs its
                 jaw_mimic_2 joint (R_x(π)) applied so R_z(-π) @ R_x(π)
                 becomes R_y(π) — matching the hand-built GLB.

    xacro_args: extra key→value pairs injected before xacro substitution.
    include_links: exact or suffix match against resolved link names.
    """
    with open(urdf_path, "r") as f:
        raw = f.read()

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    # Pre-process xacro
    if urdf_path.endswith(".xacro"):
        root_tmp = ET.fromstring(raw)
        props = collect_xacro_properties(root_tmp, extra=xacro_args)
        raw = substitute_xacro(raw, props)

    root = ET.fromstring(raw)

    # ── Collect joint origin transforms (parent→child, keyed by child link name) ──
    joint_T = {}   # child_link_name → T_parent_to_child (4×4)
    joint_parent = {}  # child_link_name → parent_link_name
    for joint in root.iter("joint"):
        child_elem  = joint.find("child")
        parent_elem = joint.find("parent")
        if child_elem is None or parent_elem is None:
            continue
        child_link  = child_elem.get("link",  "")
        parent_link = parent_elem.get("link", "")
        origin = joint.find("origin")
        rpy, xyz = parse_rpy_xyz(origin)
        joint_T[child_link]      = rpy_xyz_to_T(rpy, xyz)
        joint_parent[child_link] = parent_link

    def link_to_root_T(link_name: str) -> np.ndarray:
        """Accumulate transforms from link_name up to the root."""
        if no_chain:
            # Optionally apply only the immediate parent joint for specific links
            # (exact or suffix match, same convention as --include_links)
            if joint_links and link_name in joint_T:
                if link_name in joint_links or any(link_name.endswith(j) for j in joint_links):
                    return joint_T[link_name]
            return np.eye(4)
        T = np.eye(4)
        current = link_name
        visited = set()
        while current in joint_parent:
            if current in visited:
                break  # cycle guard
            visited.add(current)
            T = joint_T[current] @ T
            current = joint_parent[current]
        return T

    # ── Collect visuals ───────────────────────────────────────────────────────
    visuals = []
    for link in root.iter("link"):
        link_name = link.get("name", "")
        # Match exact name OR suffix (handles ${prefix}foo → foo after substitution)
        if include_links and link_name not in include_links \
                and not any(link_name.endswith(inc) for inc in include_links):
            continue

        T_link = link_to_root_T(link_name)

        for visual in link.iter("visual"):
            origin  = visual.find("origin")
            geo     = visual.find("geometry")
            if geo is None:
                continue
            mesh_elem = geo.find("mesh")
            if mesh_elem is None:
                continue

            rpy, xyz  = parse_rpy_xyz(origin)
            T_visual  = rpy_xyz_to_T(rpy, xyz)
            T_total   = T_link @ T_visual

            filename  = mesh_elem.get("filename", "")
            scale_str = mesh_elem.get("scale", "1 1 1")
            scale     = np.array([float(v) for v in scale_str.split()])

            mesh_path = resolve_mesh_path(filename, mesh_dir, urdf_dir, mesh_map=mesh_map)
            if mesh_path is None:
                print(f"  [warn] mesh not found: {filename!r}  (link={link_name})")
                continue

            visuals.append({
                "link":  link_name,
                "T":     T_total,
                "mesh":  mesh_path,
                "scale": scale,
            })

    return visuals


# ── Assembly ──────────────────────────────────────────────────────────────────

# Default colour palette (cycles if more links than colours)
_DEFAULT_COLORS = [
    [200, 200, 200, 255],
    [220, 180, 100, 255],
    [160, 200, 220, 255],
    [220, 140, 140, 255],
    [180, 220, 160, 255],
    [200, 160, 220, 255],
]


def build_glb(
    visuals: list[dict],
    color_map: dict[str, tuple],
    post_T: np.ndarray | None,
    canonical_link: str | None,
) -> bytes:
    """
    Load each mesh, apply its transform, assemble into a trimesh Scene, export GLB.

    canonical_link: if set, translate the whole assembly so that the centroid
                    of that link's visual geometry lands at the world origin.
    post_T:         optional 4×4 post-transform applied to the entire assembly.
    """
    scene = trimesh.Scene()
    link_colors: dict[str, list] = {}
    color_idx = 0

    # When there is no canonical_link, pre-combine post_T into each mesh's
    # transform so the entire pipeline is a single matrix multiply per vertex —
    # matching the hand-built GLB which did  m.apply_transform(post_T @ T_base).
    # When canonical_link is set we must build the scene first to measure the
    # centroid, so we fall back to a two-step apply (small float difference ok).
    pre_combine = (canonical_link is None) and (post_T is not None)

    for vis in visuals:
        link = vis["link"]
        m = trimesh.load(vis["mesh"], force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        # Strip metadata fields that trimesh adds from the filepath; they show up
        # in GLB 'extras' and differ from a hand-built GLB.
        # We pass geom_name explicitly so the geometry key still matches the filename.
        geom_name = os.path.basename(vis["mesh"])
        for key in ("name", "node"):
            m.metadata.pop(key, None)

        # Apply per-axis mesh scale (from URDF <mesh scale="..."/>)
        T_vis = vis["T"].copy()
        if not np.allclose(vis["scale"], 1.0):
            S = np.diag([*vis["scale"], 1.0])
            T_vis = T_vis @ S

        # Single combined transform: post_T @ T_vis (matches hand-built GLB)
        combined = post_T @ T_vis if pre_combine else T_vis
        m.apply_transform(combined)

        # Assign colour
        if link not in link_colors:
            if link in color_map:
                link_colors[link] = color_map[link]
            else:
                link_colors[link] = _DEFAULT_COLORS[color_idx % len(_DEFAULT_COLORS)]
                color_idx += 1
        m.visual.face_colors = link_colors[link]

        scene.add_geometry(m, node_name=f"part_{len(scene.geometry)}", geom_name=geom_name)

    if not scene.geometry:
        raise RuntimeError("No geometry was loaded — check mesh paths and link names.")

    if not pre_combine:
        # Two-step path (canonical_link case or no post_T)
        extra_T = np.eye(4)

        if canonical_link is not None:
            canon_verts = []
            for name, geom in scene.geometry.items():
                if name.startswith(canonical_link + "_"):
                    canon_verts.append(geom.vertices)
            if canon_verts:
                T_centre = np.eye(4)
                T_centre[:3, 3] = -np.concatenate(canon_verts).mean(axis=0)
                extra_T = T_centre @ extra_T
            else:
                print(f"  [warn] canonical_link={canonical_link!r} not found in geometry names")

        if post_T is not None:
            extra_T = post_T @ extra_T

        if not np.allclose(extra_T, np.eye(4)):
            for geom in scene.geometry.values():
                geom.apply_transform(extra_T)

    return scene.export(file_type="glb")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_colors(s: str) -> dict[str, list]:
    """Parse "link1:#rrggbb,link2:#rrggbb" into {link: [r,g,b,255]}."""
    result = {}
    if not s:
        return result
    for item in s.split(","):
        item = item.strip()
        if ":" not in item:
            continue
        link, hex_col = item.split(":", 1)
        hex_col = hex_col.lstrip("#")
        r = int(hex_col[0:2], 16)
        g = int(hex_col[2:4], 16)
        b = int(hex_col[4:6], 16)
        result[link.strip()] = [r, g, b, 255]
    return result


def main():
    p = argparse.ArgumentParser(
        description="Build a single GLB from a URDF/xacro file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--urdf",      required=True,  help="Path to .urdf or .urdf.xacro")
    p.add_argument("--mesh_dir",  required=True,  help="Directory containing STL/DAE/OBJ meshes")
    p.add_argument("--output",    default=None,   help="Output .glb path (required unless --list_links)")
    p.add_argument("--include_links", nargs="*",  help="Only include these link names (default: all)")
    p.add_argument("--colors",    default="",
                   help="Per-link colours: 'link1:#rrggbb,link2:#rrggbb'")
    p.add_argument("--canonical_link", default=None,
                   help="Translate assembly so this link's centroid is at the origin")
    p.add_argument("--post_rpy",  default=None,
                   help="Post-transform rotation as 'r,p,y' in radians (applied after canonical)")
    p.add_argument("--post_xyz",  default=None,
                   help="Post-transform translation as 'x,y,z'")
    p.add_argument("--no_chain", action="store_true",
                   help="Skip kinematic-chain accumulation — use visual origin transforms "
                        "only (no joint offsets). Combine with --joint_links for links that "
                        "need their direct parent joint applied.")
    p.add_argument("--joint_links", nargs="*",
                   help="When --no_chain is set, apply the immediate parent joint transform "
                        "for these links only (suffix match). E.g. tool_wrist_caudier_link_2_right")
    p.add_argument("--xacro_args", default="",
                   help="Inject xacro macro parameters: 'prefix=PSM1_,other=val'. "
                        "Use 'prefix=' (empty) for single-robot xacro files.")
    p.add_argument("--list_links", action="store_true",
                   help="Parse URDF and print all link names, then exit")
    p.add_argument("--mesh_map", default="",
                   help="Remap URDF mesh basenames to local files: "
                        "'urdf_name.stl:local_name.stl,...'. "
                        "Used so the loaded mesh matches a hand-built GLB that "
                        "used renamed copies of the same geometry.")
    args = p.parse_args()

    # ── Parse xacro_args ───────────────────────────────────────────────────────
    xacro_args: dict = {}
    if args.xacro_args:
        for pair in args.xacro_args.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                xacro_args[k.strip()] = v.strip()

    # ── Parse mesh_map ─────────────────────────────────────────────────────────
    mesh_map: dict = {}
    if args.mesh_map:
        for pair in args.mesh_map.split(","):
            pair = pair.strip()
            if ":" in pair:
                src, dst = pair.split(":", 1)
                mesh_map[src.strip()] = dst.strip()

    # ── Parse ──────────────────────────────────────────────────────────────────
    print(f"Parsing: {args.urdf}")
    if xacro_args:
        print(f"  xacro_args: {xacro_args}")
    visuals = parse_urdf_visuals(
        args.urdf,
        args.mesh_dir,
        include_links=args.include_links,
        xacro_args=xacro_args,
        no_chain=args.no_chain,
        joint_links=args.joint_links,
        mesh_map=mesh_map or None,
    )

    if args.list_links:
        links = sorted({v["link"] for v in visuals})
        print(f"\nLinks with visual geometry ({len(links)}):")
        for lk in links:
            print(f"  {lk}")
        return

    print(f"  {len(visuals)} visual(s) across {len({v['link'] for v in visuals})} link(s)")

    # ── Post-transform ─────────────────────────────────────────────────────────
    post_T = None
    if args.post_rpy or args.post_xyz:
        rpy = np.array([_snap_to_pi_multiples(float(v)) for v in args.post_rpy.split(",")]) \
              if args.post_rpy else np.zeros(3)
        xyz = np.array([float(v) for v in args.post_xyz.split(",")]) if args.post_xyz else np.zeros(3)
        post_T = rpy_xyz_to_T(rpy, xyz)
        print(f"  Post-transform: rpy={np.degrees(rpy).round(1)}°  xyz={xyz}")

    if not args.output:
        p.error("--output is required when not using --list_links")

    # ── Build ──────────────────────────────────────────────────────────────────
    color_map = parse_colors(args.colors)
    print("Building GLB …")
    glb_bytes = build_glb(
        visuals,
        color_map=color_map,
        post_T=post_T,
        canonical_link=args.canonical_link,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(glb_bytes)
    print(f"Saved → {out}  ({len(glb_bytes) // 1024} KB)")


if __name__ == "__main__":
    main()


# ── Example commands ───────────────────────────────────────────────────────────
#
# 1. List all links in the xacro (inject empty prefix to resolve ${prefix}):
#    python urdf_to_glb.py \
#        --urdf      /home/nan/Desktop/dvrk_model/urdf/Classic/psm_tool_caudier.urdf.xacro \
#        --mesh_dir  /home/nan/Desktop/dvrk_model/meshes/Classic/PSM \
#        --xacro_args "prefix=" \
#        --list_links
#
# 2. Build the Cadière Forceps GLB (closed jaws, canonical frame) matching caudier_grasper.glb:
#    python urdf_to_glb.py \
#        --urdf      /home/nan/Desktop/dvrk_model/urdf/Classic/psm_tool_caudier.urdf.xacro \
#        --mesh_dir  /home/nan/Desktop/datasets/dvrk_meshes \
#        --output    /home/nan/Desktop/datasets/dvrk_meshes/caudier_from_urdf.glb \
#        --xacro_args "prefix=" \
#        --include_links tool_wrist_caudier_link tool_wrist_caudier_link_shaft \
#                        tool_wrist_caudier_link_2_left tool_wrist_caudier_link_2_right \
#        --colors "tool_wrist_caudier_link:#c8c8c8,tool_wrist_caudier_link_shaft:#b4b4b4,tool_wrist_caudier_link_2_left:#dcb464,tool_wrist_caudier_link_2_right:#dcb464" \
#        --no_chain \
#        --joint_links tool_wrist_caudier_link_2_right \
#        --mesh_map "tool_wrist_caudier_link_1.stl:caudier_jaw1.stl,tool_wrist_caudier_link_1_shaft.stl:caudier_shaft.stl,tool_wrist_caudier_link_2.stl:caudier_jaw2.stl" \
#        --post_rpy "0,1.5708,0" --post_xyz "0,-0.0199,0"
#
# 3. Build any URDF robot as GLB (all links):
#    python urdf_to_glb.py \
#        --urdf    /path/to/robot.urdf \
#        --mesh_dir /path/to/meshes \
#        --output   /tmp/robot.glb
