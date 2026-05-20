"""
Stitch periodic GSVR reconstruction snapshots into a GIF showing
axial/coronal/sagittal mid-slices over training time.

Snapshots are written by train.py when `visualization.snapshots.enabled: true`
in the config. Each snapshot is a NIfTI named `reconstruction_ttime=<seconds>s.nii.gz`.

Usage:
    python GS_SVR/scripts/make_reconstruction_gif.py \\
        --snapshots-dir <output_root>/snapshots \\
        [--output reconstruction_progress.gif] \\
        [--fps 5]
"""

import argparse
import glob
import os
import re

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


TTIME_RE = re.compile(r"reconstruction_ttime=([0-9]+(?:\.[0-9]+)?)s\.nii\.gz$")


def collect_snapshots(snapshots_dir):
    paths = sorted(glob.glob(os.path.join(snapshots_dir, "reconstruction_ttime=*s.nii.gz")))
    out = []
    for p in paths:
        m = TTIME_RE.search(os.path.basename(p))
        if m:
            out.append((float(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def mid_slices(vol):
    """Return (axial, coronal, sagittal) mid-slices, oriented for radiological display."""
    sx, sy, sz = vol.shape
    axial    = np.flip(np.rot90(vol[:, :, sz // 2]), axis=1)    # XY plane
    coronal  = np.flip(np.rot90(vol[:, sy // 2, :]), axis=1)     # XZ plane
    sagittal = np.flip(np.rot90(vol[sx // 2+4, :, :]), axis=1)     # YZ plane
    return axial, coronal, sagittal


def render_frame(slices, ttime, vmin, vmax):
    """Render a single frame as an RGB ndarray."""
    titles = ["", "", ""]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), dpi=100)
    for ax, img, title in zip(axes, slices, titles):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, color="white", pad=2)
        ax.set_axis_off()
    fig.suptitle(f"t = {ttime:2.2f} s", fontsize=14, y=0.99, color="limegreen")
    # explicit tight margins: tight_layout auto-pads around suptitle/titles and
    # leaves visible black bands top/bottom.
    fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.0, wspace=0.0)
    fig.patch.set_facecolor('black')
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshots-dir", required=True, help="Directory containing reconstruction_ttime=*s.nii.gz files")
    parser.add_argument("--output", default=None, help="Output GIF path (default: <snapshots-dir>/../reconstruction_progress.gif)")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second in the GIF, 1-50 (default: 5; GIF format clamps anything above 50)")
    args = parser.parse_args()

    if args.fps > 50:
        raise SystemExit(
            f"--fps {args.fps} exceeds the GIF format's practical ceiling. "
            "GIF stores per-frame delay as integer centiseconds (1/100 s), and "
            "all major viewers clamp delays below 2 cs up to ~10 cs as a "
            "safeguard against ancient delay=0 GIFs. So --fps > 50 either "
            "rounds to the same delay as --fps=50 or is silently slowed to "
            "~10 fps at playback. Use --fps <= 50, or switch to an MP4/WebM "
            "writer if you need higher framerates."
        )

    snaps = collect_snapshots(args.snapshots_dir)
    if not snaps:
        raise SystemExit(f"No reconstruction_ttime=*s.nii.gz files found in {args.snapshots_dir}")
    print(f"Found {len(snaps)} snapshots, ttime range {snaps[0][0]:.1f}s — {snaps[-1][0]:.1f}s")

    last_vol = nib.load(snaps[-1][1]).get_fdata()
    nz = last_vol[last_vol > 0]
    if nz.size:
        vmin, vmax = np.percentile(nz, [1, 99])
    else:
        vmin, vmax = float(last_vol.min()), float(last_vol.max())
    print(f"Display range fixed from final frame: vmin={vmin:.3f}, vmax={vmax:.3f}")

    frames = []
    for ttime, path in snaps:
        vol = nib.load(path).get_fdata()
        frames.append(render_frame(mid_slices(vol), ttime, vmin, vmax))

    output = args.output or os.path.join(os.path.dirname(args.snapshots_dir.rstrip("/")), "reconstruction_progress.gif")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    # fps capped at 50 upstream: GIF delay is integer centiseconds + 2-cs viewer clamp.
    imageio.mimsave(output, frames, format="GIF", fps=args.fps, loop=0)
    print(f"Wrote {len(frames)}-frame GIF to {output}")


if __name__ == "__main__":
    main()
