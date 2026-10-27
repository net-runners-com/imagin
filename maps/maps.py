"""Analysis maps from a single image: depth/height, normal, bump, value & detail heatmaps, palette.

usage: python maps/maps.py IMAGE OUTDIR [--height 760] [--no-depth]

Outputs (PNG):
  height.png       Depth Anything V2 relative depth, 1 = near (skipped with --no-depth)
  normal.png       tangent-space normal map derived from height (RGB = xyz*0.5+0.5)
  bump.png         high-pass of height (fine relief), mid-grey = 0
  lum_bump.png     high-pass of luminance (texture bump, works without depth)
  heat_value.png   CIELAB L* as a heatmap (dark = blue, bright = yellow)
  heat_detail.png  local high-frequency energy (where the detail lives)
  palette.png      k-means palette, swatch width proportional to area
  maps.json        summary numbers
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def srgb_to_lab(rgb):
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return L, a, b


def colormap(x):
    """0..1 -> perceptual blue-teal-yellow ramp (viridis-like, no matplotlib needed)."""
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float32)
    x = np.clip(x, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    t = (x - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)


def depth(im):
    import torch
    from transformers import pipeline
    est = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf',
                   device='mps' if torch.backends.mps.is_available() else 'cpu')
    d = np.asarray(est(im)["predicted_depth"].squeeze().float().cpu().numpy())
    d = np.asarray(Image.fromarray(d).resize(im.size), np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)


def normals_from_height(h, strength):
    gy, gx = np.gradient(ndi.gaussian_filter(h, 1.5))
    n = np.dstack([-gx * strength, -gy * strength, np.ones_like(h)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def palette(rgb, k=8, iters=12, seed=0):
    px = rgb.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    px = px[rng.choice(len(px), min(len(px), 40000), replace=False)]
    c = px[rng.choice(len(px), k, replace=False)]
    for _ in range(iters):
        lab = ((px[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([px[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    share = np.bincount(lab, minlength=k) / len(lab)
    order = np.argsort(-share)
    return c[order], share[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('outdir')
    ap.add_argument('--height', type=int, default=760)
    ap.add_argument('--no-depth', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    im = Image.open(a.image).convert('RGB')
    im = im.resize((int(im.width * a.height / im.height), a.height), Image.LANCZOS)
    rgb = np.asarray(im).astype(np.float32)
    L, _, _ = srgb_to_lab(rgb)
    out = lambda name, arr: Image.fromarray(arr).save(os.path.join(a.outdir, name))
    summary = {}

    if not a.no_depth:
        h = depth(im)
        n = normals_from_height(h, im.width * 0.25)
        bump = h - ndi.gaussian_filter(h, 3)
        out('height.png', (h * 255).astype(np.uint8))
        out('normal.png', ((n * 0.5 + 0.5) * 255).astype(np.uint8))
        out('bump.png', np.clip(128 + bump / (np.abs(bump).max() + 1e-6) * 127, 0, 255).astype(np.uint8))
        summary['bump_x1000'] = float(bump.std() * 1000)
        summary['normal_spread_deg'] = float(np.degrees(np.arccos(np.clip(n[..., 2], -1, 1))).mean())

    lb = L - ndi.gaussian_filter(L, 3)
    out('lum_bump.png', np.clip(128 + lb * 4, 0, 255).astype(np.uint8))
    out('heat_value.png', colormap(L / 100))
    det = ndi.gaussian_filter(np.abs(lb), 6)
    out('heat_detail.png', colormap(det / (np.percentile(det, 99) + 1e-6)))
    cols, share = palette(rgb)
    w = 600
    bar = np.zeros((80, w, 3), np.uint8)
    x = 0
    for c, s in zip(cols, share):
        x1 = min(w, x + max(1, int(round(s * w))))
        bar[:, x:x1] = c
        x = x1
    bar[:, x:] = cols[-1]
    out('palette.png', bar)
    summary.update(L_mean=float(L.mean()), L_p5=float(np.percentile(L, 5)), L_p95=float(np.percentile(L, 95)),
                   dark_frac=float((L < 30).mean()), detail_mean=float(det.mean()),
                   palette=[[int(v) for v in c] + [round(float(s), 3)] for c, s in zip(cols, share)])
    json.dump(summary, open(os.path.join(a.outdir, 'maps.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'palette'}))


if __name__ == '__main__':
    main()
