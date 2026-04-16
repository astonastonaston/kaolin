#!/usr/bin/env python3
"""
Viser visualization of Simplicits tissue-tool simulation.

Shows:
  - Tissue surface point cloud with GT RGB colors
  - Internal simulation points (optional)
  - Tool SE3 pose: coordinate frame + shaft cylinder + PSM tip STL mesh

Flip X/Y/Z applies consistently to both tissue AND tool trajectory.

Usage:
    python visualize.py
    python visualize.py --tissue /path/to/sim_tissue_trajectory_with_rgb.pkl \
                        --tool   /path/to/3d_tool_poses.pkl \
                        --stl_dir /path/to/dvrk/psm/high_res \
                        --port   8080
"""

import argparse
import os
import pickle
import threading
import time

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

# ── Default data paths ─────────────────────────────────────────────────────────
TISSUE_PKL = (
    "/home/nan/Desktop/datasets/xpbd/"
    "StereoMIS_tissue_tool_trajectories_XPBD/sim_particles/"
    "sim_tissue_trajectory_with_rgb.pkl"
)
TOOL_PKL = (
    "/home/nan/Desktop/datasets/xpbd/"
    "StereoMIS_tissue_tool_trajectories_XPBD/sim_particles/"
    "sim_tool_trajectory.pkl"
)
STL_DIR = "/home/nan/Desktop/datasets/dvrk_meshes"
# Cadière Forceps — a laparoscopic tissue grasper (jhu-dvrk/dvrk_model).
# Pre-assembled from URDF visual transforms: wrist body + shaft + two mirrored jaw pieces.
# Built by build_caudier_grasper.py; load as a single GLB instead of per-STL.
GRIPPER_GLB = "/home/nan/Desktop/datasets/dvrk_meshes/caudier_from_urdf.glb"
# GRIPPER_GLB = "/home/nan/Desktop/datasets/dvrk_meshes/caudier_grasper.glb"
# Fallback STL list (used only if GRIPPER_GLB is missing)
TIP_STL_FILES = [
    "caudier_jaw1.stl",
    "caudier_shaft.stl",
    "caudier_jaw2.stl",
]

# ── Visual defaults ────────────────────────────────────────────────────────────
DEFAULT_SURFACE_PT_SIZE  = 0.003
DEFAULT_INTERNAL_PT_SIZE = 0.008
DEFAULT_SHAFT_LENGTH     = 0.12
DEFAULT_SHAFT_RADIUS     = 0.005
DEFAULT_AXES_LENGTH      = 0.06   # tool frame axis length
DEFAULT_AXES_RADIUS      = 0.004  # tool frame axis radius
DEFAULT_STL_SCALE        = 15.0   # STL is in metres; tissue is in ~[-1,1] units
DEFAULT_STL_OFFSET_Z     = 0.0    # shift mesh along local tool-Z
DEFAULT_SURFACE_K        = 60     # nearest tissue pts used for surface-normal alignment


# ── Math helpers ──────────────────────────────────────────────────────────────

def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → viser (w,x,y,z) quaternion."""
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def flip_matrix(fx: bool, fy: bool, fz: bool) -> np.ndarray:
    """Diagonal ±1 matrix for the selected flip axes."""
    return np.diag([
        -1.0 if fx else 1.0,
        -1.0 if fy else 1.0,
        -1.0 if fz else 1.0,
    ])


def apply_pose_flip(R: np.ndarray, t: np.ndarray, F: np.ndarray):
    """
    Reflect a rigid-body pose through the flip plane F.
      t' = F @ t
      R' = F @ R @ F    (always det=+1 because det(F)^2 = 1)
    """
    t_f = (F @ t).astype(np.float32)
    R_f = F @ R @ F
    return R_f, t_f


def compute_surface_grasp_R(
    tool_pos: np.ndarray,
    tool_R_f: np.ndarray,
    surface_xyz_f: np.ndarray,
    k: int = 60,
) -> np.ndarray:
    """
    Estimate a gripper rotation that makes the grasper look like it is
    closing onto the tissue.

    Since tool_t is the *midpoint* of the 3D gripper, the jaw tips (at
    canonical +Y from the mesh origin) must point **toward the tissue surface**
    so they appear to reach into / grasp the tissue.

    The canonical GLB frame is:
        jaw tips  → origin (0,0,0), jaw bodies extend in -Y
        jaw dir   → +Y  (jaws face toward positive Y)
        shaft     → extends in -Z

    Strategy:
      +Y (jaw direction) → surface normal pointing from tool toward tissue
      -Z (shaft)        → seeded from tool_R_f's shaft; kept ⊥ to jaw dir

    Args:
        tool_pos:      (3,)   gripper midpoint (tool_t) in scene coords
        tool_R_f:      (3,3)  tool rotation in scene (flipped) coords
        surface_xyz_f: (N,3)  tissue surface in scene coords
        k:             number of nearest pts used for PCA normal estimate
    Returns:
        (3,3) SO(3) rotation R with R @ [0,1,0] ≈ toward-tissue normal
    """
    # ── Local surface normal at grasp point ───────────────────────────────────
    dists = np.linalg.norm(surface_xyz_f - tool_pos, axis=1)
    near_pts = surface_xyz_f[np.argsort(dists)[:k]]

    centered = near_pts - near_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # smallest singular value = surface normal

    # Orient toward tissue: the vector from tool_pos to the nearest patch
    # centre tells us which side of the surface the tool is on.
    to_tissue = near_pts.mean(axis=0) - tool_pos
    if np.dot(normal, to_tissue) < 0:
        normal = -normal
    # normal now points from tool_pos toward the tissue surface ✓

    # ── Build SO(3): y = jaw dir (toward tissue), z chosen from tool shaft ────
    y_col = normal / np.linalg.norm(normal)

    # Shaft hint: project tool_R_f's shaft direction onto the plane ⊥ to y_col
    shaft_dir = tool_R_f @ np.array([0., 0., -1.])
    z_hint = shaft_dir - np.dot(shaft_dir, y_col) * y_col
    n = np.linalg.norm(z_hint)
    if n < 1e-6:
        shaft_dir = tool_R_f @ np.array([1., 0., 0.])
        z_hint = shaft_dir - np.dot(shaft_dir, y_col) * y_col
        n = np.linalg.norm(z_hint)
    if n < 1e-6:
        fb = np.array([1., 0., 0.]) if abs(y_col[0]) < 0.9 else np.array([0., 0., 1.])
        z_hint = fb - np.dot(fb, y_col) * y_col
        n = np.linalg.norm(z_hint)
    # canonical +Z = -shaft_dir (shaft goes in -Z, so +Z = toward tissue from shaft end)
    z_col = -z_hint / n

    # x = y × z  (right-hand; det = 1 guaranteed when y ⊥ z)
    x_col = np.cross(y_col, z_col)
    x_col /= np.linalg.norm(x_col)

    return np.column_stack([x_col, y_col, z_col])


def shaft_transform(tool_R: np.ndarray, tool_t: np.ndarray, length: float):
    """Center + wxyz for a cylinder shaft extending along local -Z from tool_t."""
    shaft_dir = tool_R @ np.array([0.0, 0.0, -1.0])
    center = tool_t + shaft_dir * (length / 2.0)
    y = np.array([0.0, 1.0, 0.0])
    d = shaft_dir / (np.linalg.norm(shaft_dir) + 1e-9)
    cross = np.cross(y, d)
    cn = np.linalg.norm(cross)
    dot = float(np.dot(y, d))
    if cn < 1e-6:
        wxyz = np.array([1., 0., 0., 0.]) if dot > 0 else np.array([0., 1., 0., 0.])
    else:
        angle = np.arctan2(cn, dot)
        xyzw = Rotation.from_rotvec(cross / cn * angle).as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    return center.astype(np.float32), wxyz


# ── STL loading ───────────────────────────────────────────────────────────────

def load_tip_glb(gripper_glb: str, stl_dir: str, filenames: list[str]) -> bytes | None:
    """
    Load the pre-assembled Cadière grasper GLB if it exists, otherwise
    fall back to assembling from individual STL files (without URDF transforms).
    """
    if os.path.exists(gripper_glb):
        with open(gripper_glb, "rb") as f:
            data = f.read()
        print(f"  Loaded pre-assembled grasper GLB: {gripper_glb} ({len(data)//1024} KB)")
        return data
    # Fallback: load STL files as-is (no URDF transforms — alignment will be approximate)
    print(f"  [warn] {gripper_glb} not found, falling back to STL assembly")
    scene = trimesh.Scene()
    loaded = []
    for fname in filenames:
        path = os.path.join(stl_dir, fname)
        if not os.path.exists(path):
            print(f"  [warn] STL not found: {path}")
            continue
        mesh = trimesh.load(path, force="mesh")
        mesh.visual.vertex_colors = np.tile([200, 200, 210, 220], (len(mesh.vertices), 1))
        scene.add_geometry(mesh, geom_name=fname)
        loaded.append(fname)
    if not loaded:
        return None
    print(f"  Loaded tip STLs: {loaded}")
    return scene.export(file_type="glb")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Viser sim visualizer")
    parser.add_argument("--tissue",  default=TISSUE_PKL)
    parser.add_argument("--tool",    default=TOOL_PKL)
    parser.add_argument("--stl_dir", default=STL_DIR)
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--fps",     type=float, default=10.0)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading tissue trajectory …")
    with open(args.tissue, "rb") as f:
        tissue_data = pickle.load(f)
    print("Loading tool poses …")
    with open(args.tool, "rb") as f:
        tool_raw = pickle.load(f)

    frames = tissue_data["frames"]
    N = len(frames)
    print(f"  {N} frames | {frames[0]['surface_xyz'].shape[0]:,} surface pts/frame")

    # Normalise tool data to list of {R, t} aligned with tissue frames.
    # Supports two formats:
    #   new: {meta, frames: [{tool_R, tool_t, frame_id, ...}, ...]}  (sim_tool_trajectory.pkl)
    #   old: {frame_id: {R, t}, ...}                                  (3d_tool_poses*.pkl)
    def _build_aligned_tool(tool_raw, tissue_frames):
        if isinstance(tool_raw, dict) and "frames" in tool_raw:
            # New format — index-aligned list; build frame_id → entry lookup
            tool_by_fid = {fr["frame_id"]: fr for fr in tool_raw["frames"]}
            out = []
            for tfr in tissue_frames:
                fid = tfr["frame_id"]
                if fid in tool_by_fid:
                    tf = tool_by_fid[fid]
                    out.append({"R": np.array(tf["tool_R"], dtype=np.float64),
                                "t": np.array(tf["tool_t"], dtype=np.float32)})
                else:
                    # Fallback to tool pose embedded in tissue frame
                    out.append({"R": np.array(tfr["tool_R"], dtype=np.float64),
                                "t": np.array(tfr["tool_t"], dtype=np.float32)})
            return out
        else:
            # Old format — dict keyed by frame_id, keys are R/t
            out = []
            for tfr in tissue_frames:
                fid = tfr["frame_id"]
                if fid in tool_raw:
                    tp = tool_raw[fid]
                    out.append({"R": np.array(tp["R"], dtype=np.float64),
                                "t": np.array(tp["t"], dtype=np.float32)})
                else:
                    out.append({"R": np.array(tfr["tool_R"], dtype=np.float64),
                                "t": np.array(tfr["tool_t"], dtype=np.float32)})
            return out

    aligned_tool = _build_aligned_tool(tool_raw, frames)
    print(f"  Tool poses loaded: {len(aligned_tool)} frames")

    # ── Load tool tip GLB ──────────────────────────────────────────────────────
    print("Loading tool tip STL meshes …")
    tip_glb_bytes = load_tip_glb(GRIPPER_GLB, args.stl_dir, TIP_STL_FILES)
    has_stl = tip_glb_bytes is not None

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer(port=args.port, label="Simplicits Sim")
    server.scene.set_up_direction("+y")
    server.scene.configure_default_lights()
    server.scene.add_grid(
        "grid", width=3.0, height=3.0,
        cell_color=(30, 30, 30), section_color=(70, 70, 70),
    )

    # ── GUI ────────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=N - 1, step=1, initial_value=0
        )
        play_btn = server.gui.add_button("▶  Play")
        fps_num  = server.gui.add_number(
            "FPS", initial_value=args.fps, min=1.0, max=60.0, step=1.0
        )

    with server.gui.add_folder("Visibility"):
        show_surface  = server.gui.add_checkbox("Surface pts (RGB)",  initial_value=True)
        show_internal = server.gui.add_checkbox("Internal pts",       initial_value=False)
        show_frame    = server.gui.add_checkbox("Tool frame axes",    initial_value=True)
        show_shaft    = server.gui.add_checkbox("Tool shaft cylinder", initial_value=False)
        show_stl      = server.gui.add_checkbox(
            "Tool tip mesh (STL)", initial_value=has_stl,
        )

    with server.gui.add_folder("Display"):
        surf_pt_size = server.gui.add_number(
            "Surface pt size", initial_value=DEFAULT_SURFACE_PT_SIZE,
            min=0.001, max=0.05, step=0.001,
        )

    with server.gui.add_folder("Tool Frame Axes"):
        axes_length = server.gui.add_number(
            "Axes length", initial_value=DEFAULT_AXES_LENGTH,
            min=0.005, max=0.5, step=0.005,
            hint="Length of the X/Y/Z axis arrows at the tool tip",
        )
        axes_radius = server.gui.add_number(
            "Axes radius", initial_value=DEFAULT_AXES_RADIUS,
            min=0.001, max=0.05, step=0.001,
            hint="Thickness of the axis arrows",
        )

    with server.gui.add_folder("Tool Shaft"):
        shaft_len = server.gui.add_number(
            "Shaft length", initial_value=DEFAULT_SHAFT_LENGTH,
            min=0.01, max=2.0, step=0.01,
            hint="One end is always anchored at the tool SE3 origin (tool_t)",
        )
        shaft_radius = server.gui.add_number(
            "Shaft radius", initial_value=DEFAULT_SHAFT_RADIUS,
            min=0.001, max=0.05, step=0.001,
        )

    with server.gui.add_folder("Flip (tissue + tool)"):
        flip_x = server.gui.add_checkbox("Flip X", initial_value=False)
        flip_y = server.gui.add_checkbox("Flip Y", initial_value=False)
        flip_z = server.gui.add_checkbox("Flip Z", initial_value=False)

    with server.gui.add_folder("Tool Tip Mesh Alignment"):
        stl_scale = server.gui.add_number(
            "STL scale", initial_value=DEFAULT_STL_SCALE,
            min=0.1, max=200.0, step=0.5,
        )
        surface_align = server.gui.add_checkbox(
            "Surface-align rotation", initial_value=True,
            hint="Align gripper shaft with local tissue surface normal at the grasp point",
        )
        surface_k = server.gui.add_number(
            "Surface K pts", initial_value=DEFAULT_SURFACE_K,
            min=10, max=300, step=10,
            hint="Number of nearest tissue points used to estimate the surface normal",
        )
        stl_offset_y = server.gui.add_number(
            "Y offset (jaw dir)", initial_value=0.0,
            min=-0.5, max=0.5, step=0.005,
            hint="Shift mesh along jaw direction (+Y toward tissue). "
                 "Positive = push jaw tips further into tissue.",
        )
        stl_offset_z = server.gui.add_number(
            "Z offset (local)", initial_value=DEFAULT_STL_OFFSET_Z,
            min=-0.5, max=0.5, step=0.005,
            hint="Shift mesh along local Z (fine-tune depth)",
        )
        stl_rot_x = server.gui.add_number(
            "Local rot X (deg)", initial_value=0.0, min=-180., max=180., step=5.,
            hint="Fine-tune mesh rotation around local X",
        )
        stl_rot_y = server.gui.add_number(
            "Local rot Y (deg)", initial_value=0.0, min=-180., max=180., step=5.,
        )
        stl_rot_z = server.gui.add_number(
            "Local rot Z (deg)", initial_value=0.0, min=-180., max=180., step=5.,
        )

    server.gui.add_markdown(
        f"**Frames:** {N}  \n"
        f"**Surface pts:** {frames[0]['surface_xyz'].shape[0]:,}/frame  \n"
        f"**STL tip:** {'loaded' if has_stl else 'not found'}"
    )

    # ── Scene update ──────────────────────────────────────────────────────────
    def update_scene(ti: int):
        fr = frames[ti]
        tp = aligned_tool[ti]

        # Current flip matrix
        F = flip_matrix(flip_x.value, flip_y.value, flip_z.value)

        # ── Tissue surface ─────────────────────────────────────────────────────
        # Always compute xyz (needed for surface-align rotation even when hidden)
        xyz = (fr["surface_xyz"].astype(np.float32)) @ F.T
        if show_surface.value:
            rgb = fr["surface_rgb"][:len(xyz)].astype(np.uint8)
            server.scene.add_point_cloud(
                "tissue/surface",
                points=xyz, colors=rgb,
                point_size=surf_pt_size.value,
                point_shape="circle",
            )
        else:
            try: server.scene.remove_by_name("tissue/surface")
            except Exception: pass

        # ── Internal simulation nodes ──────────────────────────────────────────
        if show_internal.value:
            ixyz = (fr["internal_xyz"].astype(np.float32)) @ F.T
            n = len(ixyz)
            server.scene.add_point_cloud(
                "tissue/internal",
                points=ixyz,
                colors=np.tile(np.array([255, 230, 0], dtype=np.uint8), (n, 1)),
                point_size=DEFAULT_INTERNAL_PT_SIZE,
                point_shape="diamond",
            )
        else:
            try: server.scene.remove_by_name("tissue/internal")
            except Exception: pass

        # ── Tool pose ──────────────────────────────────────────────────────────
        if tp is None or tp.get("R") is None:
            for name in ("tool/frame", "tool/shaft", "tool/mesh"):
                try: server.scene.remove_by_name(name)
                except Exception: pass
            return

        tool_R = np.array(tp["R"], dtype=np.float64)
        tool_t = np.array(tp["t"], dtype=np.float32)

        # Apply flip to the pose
        tool_R_f, tool_t_f = apply_pose_flip(tool_R, tool_t, F)
        wxyz = mat_to_wxyz(tool_R_f)

        # ── Tool frame axes ────────────────────────────────────────────────────
        if show_frame.value:
            server.scene.add_frame(
                "tool/frame",
                wxyz=wxyz, position=tool_t_f,
                axes_length=axes_length.value,
                axes_radius=axes_radius.value,
            )
        else:
            try: server.scene.remove_by_name("tool/frame")
            except Exception: pass

        # ── Shaft cylinder ─────────────────────────────────────────────────────
        # One end is PINNED to tool_t_f (the SE3 translation origin).
        # The shaft extends along the tool's local -Z direction.
        #   center = tool_t_f + shaft_dir * (length/2)
        #   near end = center - shaft_dir*(length/2) = tool_t_f  ← pinned
        if show_shaft.value:
            scenter, swxyz = shaft_transform(tool_R_f, tool_t_f, shaft_len.value)
            server.scene.add_cylinder(
                "tool/shaft",
                radius=shaft_radius.value,
                height=shaft_len.value,
                color=(200, 200, 210),
                wxyz=swxyz, position=scenter,
                opacity=0.85,
            )
        else:
            try: server.scene.remove_by_name("tool/shaft")
            except Exception: pass

        # ── Tool tip STL mesh ──────────────────────────────────────────────────
        if show_stl.value and has_stl:
            # Base rotation: surface-aligned (shaft → local surface normal)
            # or raw tool_R if surface-align is off.
            if surface_align.value and len(xyz) > 3:
                base_R = compute_surface_grasp_R(
                    tool_t_f, tool_R_f, xyz, k=int(surface_k.value)
                )
            else:
                base_R = tool_R_f

            # Optional fine-tune rotation applied in mesh-local frame
            local_R = Rotation.from_euler(
                "xyz",
                [stl_rot_x.value, stl_rot_y.value, stl_rot_z.value],
                degrees=True,
            ).as_matrix()
            combined_R = base_R @ local_R
            mesh_wxyz = mat_to_wxyz(combined_R)

            # Jaw tips are at mesh origin → place directly at tool_t_f.
            # Y offset pushes jaw tips toward (+) or away from (-) tissue.
            # Z offset shifts along the gripper shaft axis for fine-tuning.
            local_y_dir = base_R @ np.array([0., 1., 0.])
            local_z_dir = base_R @ np.array([0., 0., 1.])
            mesh_pos = (tool_t_f
                        + local_y_dir * stl_offset_y.value
                        + local_z_dir * stl_offset_z.value)

            server.scene.add_glb(
                "tool/mesh",
                glb_data=tip_glb_bytes,
                wxyz=mesh_wxyz,
                position=mesh_pos.astype(np.float32),
                scale=stl_scale.value,
            )
        else:
            try: server.scene.remove_by_name("tool/mesh")
            except Exception: pass

    # ── Callbacks ─────────────────────────────────────────────────────────────
    _playing = threading.Event()

    def _refresh(_=None):
        update_scene(frame_slider.value)

    frame_slider.on_update(_refresh)
    show_surface.on_update(_refresh)
    show_internal.on_update(_refresh)
    show_frame.on_update(_refresh)
    show_shaft.on_update(_refresh)
    show_stl.on_update(_refresh)
    surf_pt_size.on_update(_refresh)
    axes_length.on_update(_refresh)
    axes_radius.on_update(_refresh)
    shaft_len.on_update(_refresh)
    shaft_radius.on_update(_refresh)
    flip_x.on_update(_refresh)
    flip_y.on_update(_refresh)
    flip_z.on_update(_refresh)
    stl_scale.on_update(_refresh)
    surface_align.on_update(_refresh)
    surface_k.on_update(_refresh)
    stl_offset_y.on_update(_refresh)
    stl_offset_z.on_update(_refresh)
    stl_rot_x.on_update(_refresh)
    stl_rot_y.on_update(_refresh)
    stl_rot_z.on_update(_refresh)

    @play_btn.on_click
    def _(_):
        if _playing.is_set():
            _playing.clear()
            play_btn.label = "▶  Play"
        else:
            _playing.set()
            play_btn.label = "⏸  Pause"

    # ── Playback thread ───────────────────────────────────────────────────────
    def _playback():
        while True:
            if _playing.is_set():
                # Setting .value fires on_update → update_scene automatically
                frame_slider.value = (frame_slider.value + 1) % N
                time.sleep(1.0 / max(fps_num.value, 1.0))
            else:
                time.sleep(0.05)

    threading.Thread(target=_playback, daemon=True).start()

    update_scene(0)

    print(f"\nViser running → open http://localhost:{args.port} in your browser")
    print("Tips:")
    print("  • Use Flip X/Y/Z to orient the scene — both tissue AND tool flip together")
    print("  • Use 'Tool Tip Mesh Alignment' sliders to align the STL with the pose axes")
    print("  • 'Local rot X/Y/Z' rotates the STL within the tool frame (tune to align jaws)")
    server.sleep_forever()


if __name__ == "__main__":
    main()

# laparoscopic atraumatic grasper (specifically looks like a Johan/Dolphin-nose grasper or Cadiere-style forceps) 
