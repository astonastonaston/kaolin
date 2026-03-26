#!/usr/bin/env python3
"""
Temporal smoothing of tool SE3 poses (R, t per frame).

Translation : Savitzky-Golay filter (no phase shift, preserves local shape).
Rotation    : Two-stage pipeline —
  1. Outlier clamp  — frames where angular step > --max_rot_deg are replaced
                      by SLERP between their nearest valid neighbours.
  2. SO(3) mean     — sliding-window Fréchet mean via Rotation.mean(weights),
                      with Gaussian weights.  Runs --rot_passes times.
  This is far more principled than flat quaternion averaging and handles
  large-angle jitter properly.

Supports both pkl formats:
  old: dict{ frame_id: {R, t} }
  new: dict{ 'frames': [{frame_id, tool_R, tool_t, ...}] }   (sim_tool_trajectory.pkl)

Usage:
    python smooth_tool_poses.py \\
        --input  path/to/tool_poses.pkl \\
        --output path/to/tool_poses_smooth.pkl \\
        --window 9 --poly 3 \\
        --rot_window 21 --rot_passes 2 --max_rot_deg 20

    # with plots:
    python smooth_tool_poses.py ... --plot
"""

import argparse
import pickle

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


# ── Helpers ───────────────────────────────────────────────────────────────────

def angular_speed_deg(R_seq: np.ndarray) -> np.ndarray:
    """Geodesic angular speed in deg/frame, shape (N-1,). No Euler, no gimbal."""
    rots = Rotation.from_matrix(R_seq)
    speeds = []
    for i in range(len(rots) - 1):
        dR = rots[i].inv() * rots[i + 1]
        speeds.append(np.degrees(dR.magnitude()))
    return np.array(speeds)


def _odd_window(window: int, poly: int) -> int:
    win = window if window % 2 == 1 else window + 1
    return max(win, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)


# ── Translation smoothing ─────────────────────────────────────────────────────

def smooth_translations(t_seq: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay per axis. Shape (N,3) → (N,3)."""
    win = _odd_window(window, poly)
    out = np.zeros_like(t_seq)
    for ax in range(3):
        out[:, ax] = savgol_filter(t_seq[:, ax], window_length=win, polyorder=poly)
    return out


# ── Rotation smoothing ────────────────────────────────────────────────────────

def clamp_rotation_outliers(
    R_seq: np.ndarray,
    max_deg: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Replace any frame whose angular step from the previous frame exceeds
    max_deg with a SLERP interpolation between its nearest valid neighbours.
    Iterates until no outliers remain (handles consecutive bad frames).
    """
    rots = Rotation.from_matrix(R_seq).as_quat()   # (N,4) xyzw
    N = len(rots)
    total_replaced = 0

    for _ in range(N):               # at most N iterations (converges fast)
        speeds = angular_speed_deg(Rotation.from_quat(rots).as_matrix())
        bad = np.where(speeds > max_deg)[0] + 1    # indices of the bad frames (after the step)
        if len(bad) == 0:
            break
        total_replaced += len(bad)
        for idx in bad:
            # Find nearest valid neighbour before and after
            prev = max(0, idx - 1)
            nxt  = min(N - 1, idx + 1)
            if prev == idx:    # at boundary
                rots[idx] = rots[nxt]
            elif nxt == idx:
                rots[idx] = rots[prev]
            else:
                slerp = Slerp([0, 1], Rotation.from_quat(np.stack([rots[prev], rots[nxt]])))
                rots[idx] = slerp([0.5]).as_quat()[0]
    else:
        if verbose:
            remaining = (angular_speed_deg(Rotation.from_quat(rots).as_matrix()) > max_deg).sum()
            print(f"    [warn] {remaining} frames still above threshold after max iterations")

    if verbose and total_replaced:
        print(f"    Clamped {total_replaced} outlier frame(s) (>{max_deg:.1f}°/frame) via SLERP")

    return Rotation.from_quat(rots).as_matrix()


def smooth_rotations_so3(
    R_seq: np.ndarray,
    window: int,
    sigma: float | None = None,
) -> np.ndarray:
    """
    One pass of SO(3) sliding-window Fréchet mean (Rotation.mean with
    Gaussian weights).  This is the correct way to average rotations on
    the manifold — unlike flat quaternion averaging it handles large angles
    and doesn't drift off SO(3).
    """
    N = len(R_seq)
    half = window // 2
    if sigma is None:
        sigma = half / 2.0 if half > 0 else 1.0

    ks = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (ks / sigma) ** 2)
    kernel /= kernel.sum()

    rots = Rotation.from_matrix(R_seq)
    smoothed = []
    for i in range(N):
        idx = np.clip(np.arange(i - half, i + half + 1), 0, N - 1)
        w   = kernel.copy()
        # Weight boundary-clamped frames by their accumulated weight
        smoothed.append(rots[idx].mean(weights=w))

    return Rotation.concatenate(smoothed).as_matrix()


def smooth_rotations(
    R_seq: np.ndarray,
    window: int,
    sigma: float | None = None,
    passes: int = 1,
    max_rot_deg: float | None = None,
) -> np.ndarray:
    """Full rotation smoothing pipeline."""
    R = R_seq.copy()

    # Stage 1: clamp outlier jumps
    if max_rot_deg is not None:
        print(f"  Stage 1: clamping rotation steps > {max_rot_deg}° …")
        R = clamp_rotation_outliers(R, max_deg=max_rot_deg)
        spd = angular_speed_deg(R)
        print(f"    After clamp: max step = {spd.max():.2f}°  mean = {spd.mean():.2f}°")

    # Stage 2: SO(3) Fréchet mean, multiple passes
    for p in range(passes):
        print(f"  Stage 2 pass {p+1}/{passes}: SO(3) sliding-window mean (window={window}) …")
        R = smooth_rotations_so3(R, window=window, sigma=sigma)

    return R


# ── Data loading ──────────────────────────────────────────────────────────────

def load_poses(path: str):
    """
    Returns (frame_ids, t_seq (N,3), R_seq (N,3,3), orig_dtype_t).
    Handles both pkl formats automatically.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict) and "frames" in raw:
        # New format: {meta, frames:[{frame_id, tool_t, tool_R, ...}]}
        frame_list = sorted(raw["frames"], key=lambda x: x["frame_id"])
        frame_ids = [fr["frame_id"] for fr in frame_list]
        t_seq = np.array([fr["tool_t"] for fr in frame_list], dtype=np.float64)
        R_seq = np.array([np.array(fr["tool_R"], dtype=np.float64) for fr in frame_list])
        orig_t_dtype = frame_list[0]["tool_t"].dtype
        return frame_ids, t_seq, R_seq, orig_t_dtype, "new"
    else:
        # Old format: {frame_id: {R, t}}
        frame_ids = sorted(raw.keys())
        t_seq = np.array([raw[fid]["t"] for fid in frame_ids], dtype=np.float64)
        R_seq = np.array([np.array(raw[fid]["R"], dtype=np.float64) for fid in frame_ids])
        orig_t_dtype = raw[frame_ids[0]]["t"].dtype
        return frame_ids, t_seq, R_seq, orig_t_dtype, "old"


def save_poses(path: str, frame_ids, R_smooth, t_smooth, orig_t_dtype, fmt: str):
    """Save in the same format as the input."""
    if fmt == "new":
        # Rebuild frames list with smoothed values
        frames_out = []
        for i, fid in enumerate(frame_ids):
            frames_out.append({
                "frame_id": fid,
                "tool_R": R_smooth[i],
                "tool_t": t_smooth[i].astype(orig_t_dtype),
            })
        out = {"frames": frames_out, "meta": {"smoothed": True}}
    else:
        out = {}
        for i, fid in enumerate(frame_ids):
            out[fid] = {
                "R": R_smooth[i],
                "t": t_smooth[i].astype(orig_t_dtype),
            }
    with open(path, "wb") as f:
        pickle.dump(out, f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smooth tool SE3 poses temporally")
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    # Translation
    parser.add_argument("--window",      type=int,   default=9,
                        help="SavGol window for translation (default: 9)")
    parser.add_argument("--poly",        type=int,   default=3,
                        help="SavGol polynomial order (default: 3)")
    # Rotation
    parser.add_argument("--rot_window",  type=int,   default=None,
                        help="SO(3) smoothing window for rotation (default: same as --window)")
    parser.add_argument("--rot_sigma",   type=float, default=None,
                        help="Gaussian sigma for rotation (default: rot_window/4)")
    parser.add_argument("--rot_passes",  type=int,   default=2,
                        help="Number of SO(3) smoothing passes (default: 2)")
    parser.add_argument("--max_rot_deg", type=float, default=20.0,
                        help="Clamp rotation steps above this deg/frame before smoothing "
                             "(default: 20). Set 0 to disable.")
    parser.add_argument("--plot",        action="store_true")
    args = parser.parse_args()

    rot_win = args.rot_window if args.rot_window is not None else args.window
    max_rot = args.max_rot_deg if args.max_rot_deg > 0 else None

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading: {args.input}")
    frame_ids, t_seq, R_seq, orig_t_dtype, fmt = load_poses(args.input)
    N = len(frame_ids)
    print(f"  {N} frames, format='{fmt}', IDs {frame_ids[0]}…{frame_ids[-1]}")

    spd0 = angular_speed_deg(R_seq)
    print(f"  Initial rotation speed: mean={spd0.mean():.2f}°  max={spd0.max():.2f}°  "
          f"frames>{max_rot or 999:.0f}°: {(spd0 > (max_rot or 999)).sum()}")

    # ── Smooth ────────────────────────────────────────────────────────────────
    print(f"\nSmoothing translations (SavGol window={args.window}, poly={args.poly}) …")
    t_smooth = smooth_translations(t_seq, window=args.window, poly=args.poly)

    print(f"\nSmoothing rotations (rot_window={rot_win}, passes={args.rot_passes}, "
          f"max_rot_deg={max_rot}) …")
    R_smooth = smooth_rotations(
        R_seq,
        window=rot_win,
        sigma=args.rot_sigma,
        passes=args.rot_passes,
        max_rot_deg=max_rot,
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    td_b = np.linalg.norm(np.diff(t_seq,    axis=0), axis=1)
    td_a = np.linalg.norm(np.diff(t_smooth, axis=0), axis=1)
    spd1 = angular_speed_deg(R_smooth)

    print(f"\n{'─'*55}")
    print(f"  Translation step  before: mean={td_b.mean():.5f}  max={td_b.max():.5f}")
    print(f"  Translation step  after:  mean={td_a.mean():.5f}  max={td_a.max():.5f}")
    print(f"  Rotation speed    before: mean={spd0.mean():.2f}°  max={spd0.max():.2f}°")
    print(f"  Rotation speed    after:  mean={spd1.mean():.2f}°  max={spd1.max():.2f}°")
    print(f"{'─'*55}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving → {args.output}")
    save_poses(args.output, frame_ids, R_smooth, t_smooth, orig_t_dtype, fmt)
    print("Done.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        fig.suptitle(
            f"Smoothing  t-window={args.window}  rot-window={rot_win}  "
            f"passes={args.rot_passes}  max_rot={max_rot}°",
            fontsize=11,
        )

        # Row 0: translation
        for ax_i, (ax, lbl) in enumerate(zip(axes[0], ["t_x", "t_y", "t_z"])):
            ax.plot(t_seq[:, ax_i],    alpha=0.55, color="steelblue", label="original")
            ax.plot(t_smooth[:, ax_i], alpha=0.95, color="tomato",    label="smoothed", lw=1.8)
            ax.set_title(lbl)
            ax.legend(fontsize=8)

        # Row 1: angular speed (geodesic, no Euler / no gimbal lock)
        ax = axes[1][0]
        ax.plot(spd0, alpha=0.55, color="steelblue", label="original")
        ax.plot(spd1, alpha=0.95, color="tomato",    label="smoothed", lw=1.8)
        ax.axhline(max_rot or 0, color="grey", ls="--", lw=1, label=f"clamp={max_rot}°")
        ax.set_title("Rotation speed (deg/frame, geodesic)")
        ax.legend(fontsize=8)

        # Per-axis angular velocity using rotvec (less gimbal-lock than Euler)
        rv_before = np.array([
            (Rotation.from_matrix(R_seq[i].T    @ R_seq[i+1])   .as_rotvec(degrees=True))
            for i in range(N - 1)
        ])
        rv_after = np.array([
            (Rotation.from_matrix(R_smooth[i].T @ R_smooth[i+1]).as_rotvec(degrees=True))
            for i in range(N - 1)
        ])
        for ax_i, (ax, lbl) in enumerate(zip(axes[1][1:], ["rot-vel Y (rotvec)", "rot-vel Z (rotvec)"])):
            ax.plot(rv_before[:, ax_i + 1], alpha=0.55, color="steelblue", label="original")
            ax.plot(rv_after[:,  ax_i + 1], alpha=0.95, color="tomato",    label="smoothed", lw=1.8)
            ax.set_title(f"{lbl} (deg/frame)")
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

# ── Example commands ──────────────────────────────────────────────────────────
# Old-format poses:
# python viser_vis/smooth_tool_poses.py \
#     --input  /home/nan/Desktop/datasets/StereoMIS_EndoNeRF/StereoMIS_compilance_gaussians/3d_tool_poses.pkl \
#     --output /home/nan/Desktop/datasets/StereoMIS_EndoNeRF/StereoMIS_compilance_gaussians/3d_tool_poses_smooth.pkl \
#     --window 9 --rot_window 21 --rot_passes 2 --max_rot_deg 20 --plot
#
# New-format (sim_tool_trajectory):
# python viser_vis/smooth_tool_poses.py \
#     --input  /home/nan/Desktop/datasets/xpbd/StereoMIS_tissue_tool_trajectories_XPBD/sim_particles/sim_tool_trajectory.pkl \
#     --output /home/nan/Desktop/datasets/xpbd/StereoMIS_tissue_tool_trajectories_XPBD/sim_particles/sim_tool_trajectory_smooth.pkl \
#     --window 9 --rot_window 21 --rot_passes 2 --max_rot_deg 20 --plot
