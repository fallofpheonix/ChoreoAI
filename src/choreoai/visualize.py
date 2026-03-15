"""
visualize.py — Skeleton animation and export utilities.

Functions:
  * :func:`animate_skeleton` — render a ``(T, K, 3)`` pose sequence as an
    in-memory matplotlib animation (optionally saved to an MP4 file).
  * :func:`export_threejs_json` — export skeleton data as a Three.js-compatible
    JSON structure for web-based visualisation.

Skeleton connectivity is specified as a list of ``(joint_a, joint_b)`` pairs.
Defaults to a COCO-17 connectivity table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

# Lazy matplotlib import to avoid requiring display in headless environments
_plt_imported = False


def _import_matplotlib() -> tuple[Any, Any, Any]:
    global _plt_imported
    import matplotlib
    matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    _plt_imported = True
    return matplotlib, plt, animation


# ---------------------------------------------------------------------------
# Default COCO-17 skeleton connectivity
# ---------------------------------------------------------------------------

COCO17_CONNECTIONS: list[tuple[int, int]] = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]


# ---------------------------------------------------------------------------
# animate_skeleton
# ---------------------------------------------------------------------------


def animate_skeleton(
    poses: Tensor | np.ndarray,
    *,
    connections: list[tuple[int, int]] | None = None,
    fps: int = 30,
    output_path: str | Path | None = None,
    title: str = "ChoreoAI — Generated Motion",
    joint_color: str = "steelblue",
    bone_color: str = "lightblue",
    background_color: str = "black",
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 120,
) -> Any:
    """Render a skeleton pose sequence as a matplotlib animation.

    Args:
        poses: ``(T, K, 3)`` pose tensor or numpy array.
        connections: List of ``(joint_a, joint_b)`` bone pairs.
            Defaults to :data:`COCO17_CONNECTIONS`.
        fps: Frames per second.
        output_path: If provided, save the animation to this MP4 file.
        title: Figure title.
        joint_color: Colour for joint scatter points.
        bone_color: Colour for bone lines.
        background_color: Axes background colour.
        figsize: Figure size in inches.
        dpi: Figure resolution.

    Returns:
        :class:`matplotlib.animation.FuncAnimation` instance.
    """
    _, plt, animation = _import_matplotlib()

    if connections is None:
        connections = COCO17_CONNECTIONS

    if isinstance(poses, Tensor):
        data = poses.detach().cpu().numpy()  # (T, K, 3)
    else:
        data = np.asarray(poses, dtype=np.float32)

    T, K, _ = data.shape

    # Determine plot bounds
    x_min, x_max = data[:, :, 0].min(), data[:, :, 0].max()
    y_min, y_max = data[:, :, 1].min(), data[:, :, 1].max()
    z_min, z_max = data[:, :, 2].min(), data[:, :, 2].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-3)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(background_color)
    ax.set_title(title, color="white", fontsize=10)
    ax.tick_params(colors="grey")
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    ax.set_xlabel("X", color="grey")
    ax.set_ylabel("Y", color="grey")
    ax.set_zlabel("Z", color="grey")

    # Initialise artists
    scatter = ax.scatter([], [], [], c=joint_color, s=30, depthshade=False)
    lines = [
        ax.plot([], [], [], color=bone_color, linewidth=1.5)[0]
        for _ in connections
    ]
    time_text = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes, color="white", fontsize=8
    )

    def _update(frame: int) -> list[Any]:
        joints = data[frame]  # (K, 3)
        scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])

        for line, (a, b) in zip(lines, connections):
            if a < K and b < K:
                xs = [joints[a, 0], joints[b, 0]]
                ys = [joints[a, 1], joints[b, 1]]
                zs = [joints[a, 2], joints[b, 2]]
                line.set_data(xs, ys)
                line.set_3d_properties(zs)

        time_text.set_text(f"Frame {frame + 1}/{T}")
        return [scatter, *lines, time_text]

    interval_ms = int(1000 / fps)
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=T,
        interval=interval_ms,
        blit=False,
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(fps=fps, metadata={"title": title})
        anim.save(str(out), writer=writer, dpi=dpi)
        plt.close(fig)

    return anim


# ---------------------------------------------------------------------------
# export_threejs_json
# ---------------------------------------------------------------------------


def export_threejs_json(
    poses: Tensor | np.ndarray,
    *,
    connections: list[tuple[int, int]] | None = None,
    fps: int = 30,
    name: str = "generated_motion",
) -> dict[str, Any]:
    """Export a pose sequence as a Three.js-compatible JSON object.

    The output dict can be serialised with ``json.dumps`` and loaded
    directly in a Three.js scene with a custom skeleton renderer.

    Schema::

        {
          "name": str,
          "fps": int,
          "num_frames": int,
          "num_joints": int,
          "connections": [[a, b], ...],
          "frames": [
            [[x, y, z], ...],   # frame 0 — one entry per joint
            ...
          ]
        }

    Args:
        poses: ``(T, K, 3)`` pose tensor or numpy array.
        connections: Skeleton bone connectivity.
        fps: Playback framerate metadata.
        name: Animation name.

    Returns:
        Three.js-compatible dictionary.
    """
    if connections is None:
        connections = COCO17_CONNECTIONS

    if isinstance(poses, Tensor):
        data = poses.detach().cpu().numpy().tolist()
    else:
        data = np.asarray(poses, dtype=float).tolist()

    T = len(data)
    K = len(data[0]) if T > 0 else 0

    return {
        "name": name,
        "fps": fps,
        "num_frames": T,
        "num_joints": K,
        "connections": [list(c) for c in connections],
        "frames": data,
    }
