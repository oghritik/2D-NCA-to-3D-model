#!/usr/bin/env python3
"""
build_viewer.py
================
Run this LOCALLY, next to your nca_all_transitions.npz. It never uploads
anything anywhere -- it reads the npz, classifies pixels into categories,
compresses everything to uint8 grids, and writes ONE finished HTML file
(nca_room_3d_v2.html) with the data baked in. Just open that file in a browser.

Usage:
    python build_viewer.py nca_all_transitions.npz
    python build_viewer.py nca_all_transitions.npz --out my_viewer.html
    python build_viewer.py nca_all_transitions.npz --stride 2   # keep every 2nd step (smaller file)

Requires: numpy, pillow   (pip install numpy pillow)
"""

import sys
import json
import base64
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Palette: calibrated from your 4 clean layout PNGs (exact colors, not guessed)
# category id -> (name, RGB reference color, extrusion height, hex for JS)
# Heights are an aesthetic choice -- edit freely, doesn't touch the model data.
# ---------------------------------------------------------------------------
PALETTE = {
    1: dict(name="wall",   ref=(40, 40, 40),    height=3.1,  hex=0x282828),
    2: dict(name="yellow", ref=(248, 208, 16),  height=1.00, hex=0xF8D010),
    3: dict(name="blue",   ref=(0, 72, 168),    height=1.30, hex=0x0048A8),
    4: dict(name="red",    ref=(248, 48, 48),   height=0.70, hex=0xF83030),
    5: dict(name="orange", ref=(248, 112, 24),  height=0.90, hex=0xF87018),
    6: dict(name="green",  ref=(0, 184, 96),    height=1.10, hex=0x00B860),
    7: dict(name="floor",  ref=(248, 248, 248), height=0.05, hex=0xF8F8F8),
}
ALIVE_ALPHA_THRESHOLD = 0.1
CATEGORY_EMPTY = 0

REF_IDS = list(PALETTE.keys())
REF_COLORS = np.array([PALETTE[k]["ref"] for k in REF_IDS], dtype=np.float32)  # (K,3)


def classify_frame(rgb_premult, alpha):
    """
    rgb_premult: (3,H,W) float32, premultiplied by alpha (as stored in npz)
    alpha:       (H,W) float32
    returns: (H,W) uint8 category grid
    """
    alive = alpha > ALIVE_ALPHA_THRESHOLD
    alpha_safe = np.clip(alpha, 1e-3, None)
    true_rgb = (rgb_premult / alpha_safe[None, :, :]) * 255.0   # un-premultiply, scale to 0-255
    true_rgb = np.clip(true_rgb, 0, 255)
    true_rgb = np.transpose(true_rgb, (1, 2, 0))  # (H,W,3)

    # distance to every reference color -> nearest match
    diffs = true_rgb[None, :, :, :] - REF_COLORS[:, None, None, :]   # (K,H,W,3)
    dists = np.sum(diffs * diffs, axis=-1)                            # (K,H,W)
    nearest_idx = np.argmin(dists, axis=0)                            # (H,W) index into REF_IDS
    category = np.array(REF_IDS, dtype=np.uint8)[nearest_idx]

    category = np.where(alive, category, CATEGORY_EMPTY).astype(np.uint8)
    return category


def make_thumbnail_b64(rgb_premult, alpha, size=96):
    """Build a small PNG preview (data URI) of a settled frame for the UI thumbnails."""
    alpha_safe = np.clip(alpha, 1e-3, None)
    true_rgb = (rgb_premult / alpha_safe[None, :, :]) * 255.0
    true_rgb = np.clip(true_rgb, 0, 255).astype(np.uint8)
    true_rgb = np.transpose(true_rgb, (1, 2, 0))  # (H,W,3)
    alive = (alpha > ALIVE_ALPHA_THRESHOLD)
    canvas = np.full_like(true_rgb, 248)  # white background
    canvas[alive] = true_rgb[alive]
    im = Image.fromarray(canvas, mode="RGB").resize((size, size), Image.NEAREST)
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", type=str)
    ap.add_argument("--out", type=str, default="nca_room_3d_v2.html")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth timestep (default 1 = keep all)")
    args = ap.parse_args()

    print(f"Loading {args.npz_path} ...")
    data = np.load(args.npz_path, allow_pickle=True)

    states = data["states"]                  # (P,T,C,H,W)
    from_ids = data["from_layout_ids"]        # (P,)
    to_ids = data["to_layout_ids"]            # (P,)
    layout_names = data["layout_names"]       # (N,)
    switch_step = int(data["switch_step"])    # scalar

    P, T, C, H, W = states.shape
    N = len(layout_names)
    print(f"P={P} trajectories, T={T} steps, C={C} channels, {H}x{W} grid, N={N} layouts, switch_step={switch_step}")

    assert C >= 4, f"Expected at least 4 channels (RGBA), got {C}"
    if C != 4:
        print(f"Note: {C} channels found (hidden channels present) -- using first 4 as RGBA, ignoring the rest.")

    stride = max(1, args.stride)
    kept_steps = list(range(0, T, stride))
    if kept_steps[-1] != T - 1:
        kept_steps.append(T - 1)  # always keep the final settled frame
    T_kept = len(kept_steps)
    print(f"Keeping {T_kept}/{T} steps (stride={stride})")

    pairs_out = []
    thumbnails = {}  # layout_id -> data URI, filled from first available settled frame

    for i in range(P):
        traj = states[i]  # (T,C,H,W)
        cat_stack = np.zeros((T_kept, H, W), dtype=np.uint8)
        for k, t in enumerate(kept_steps):
            frame = traj[t]
            rgb = frame[0:3].astype(np.float32)
            alpha = frame[3].astype(np.float32)
            cat_stack[k] = classify_frame(rgb, alpha)

        fid, tid = int(from_ids[i]), int(to_ids[i])

        if fid not in thumbnails:
            thumbnails[fid] = make_thumbnail_b64(traj[0, 0:3].astype(np.float32), traj[0, 3].astype(np.float32))
        if tid not in thumbnails:
            thumbnails[tid] = make_thumbnail_b64(traj[T - 1, 0:3].astype(np.float32), traj[T - 1, 3].astype(np.float32))

        packed = base64.b64encode(cat_stack.tobytes()).decode("ascii")
        pairs_out.append({
            "from_id": fid,
            "to_id": tid,
            "from_name": str(layout_names[fid]),
            "to_name": str(layout_names[tid]),
            "steps_kept": kept_steps,
            "T": T_kept,
            "data_b64": packed,
        })
        print(f"  [{i+1}/{P}] {layout_names[fid]} -> {layout_names[tid]}  ({len(packed)/1024:.0f} KB base64)")

    for lid in range(N):
        if lid not in thumbnails:
            thumbnails[lid] = None  # no trajectory started/ended here; leave blank in UI

    payload = {
        "grid_size": H,
        "num_layouts": N,
        "layout_names": [str(x) for x in layout_names],
        "switch_step": switch_step,
        "switch_step_index_in_kept": min(range(T_kept), key=lambda k: abs(kept_steps[k] - switch_step)),
        "palette": {str(k): {"name": v["name"], "height": v["height"], "hex": v["hex"]} for k, v in PALETTE.items()},
        "thumbnails": thumbnails,
        "pairs": pairs_out,
    }

    payload_json = json.dumps(payload)
    print(f"Total embedded payload: {len(payload_json)/1024/1024:.2f} MB (as JSON text)")

    template_path = Path(__file__).parent / "viewer_template.html"
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("__PAYLOAD_JSON__", payload_json)

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nDone. Wrote {out_path.resolve()}  ({out_path.stat().st_size/1024/1024:.2f} MB)")
    print("Open it directly in a browser -- no server needed.")


if __name__ == "__main__":
    main()
