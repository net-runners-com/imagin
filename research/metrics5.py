"""Depth-banded value/contrast comparison: reference anime backgrounds vs our generations.

Depth: Depth Anything V2 Small (relative inverse depth, larger = nearer), normalised per image
to 0..1 by its 1st/99th percentile. Bands on that normalised depth:
  sky/atmo < 0.10 <= far < 0.35 <= mid < 0.65 <= near
Per band: L* (CIELAB lightness 0-100) mean / p5 (shadow floor) / p95 / contrast (p95-p5),
b* mean (negative = bluer), gradient magnitude of L* (edge/detail energy).
"""
import json, sys, glob, os
import numpy as np
from PIL import Image



SP = os.path.dirname(os.path.abspath(__file__))
OUT = "./output"
H = 760  # analyse everything at the same height

def white_spans(img, axis):
    a = np.asarray(img.convert("RGB")).astype(int)
    white = (a.min(axis=2) > 238)
    frac = white.mean(axis=0 if axis == "x" else 1)
    flags = list(frac > 0.9) + [True]
    out, s = [], None
    for i, f in enumerate(flags):
        if not f and s is None: s = i
        if f and s is not None:
            if i - s > 100: out.append((s, i))
            s = None
    return out

def panels(path):
    im = Image.open(path).convert("RGB")
    return [im.crop((x0, y0, x1, y1)) for (y0, y1) in white_spans(im, "y") for (x0, x1) in white_spans(im, "x")]

def insta_crop(path):  # original ai_anime phone screenshots: keep the longest non-white row run
    im = Image.open(path).convert("RGB"); a = np.asarray(im).astype(int)
    photo = list(((a > 235).all(axis=2)).mean(axis=1) < 0.3) + [False]
    best, s = (0, 0), None
    for y, v in enumerate(photo):
        if v and s is None: s = y
        if not v and s is not None:
            if y - s > best[1] - best[0]: best = (s, y)
            s = None
    return im.crop((0, best[0], im.width, best[1]))

def srgb_to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    b = 200 * (f[..., 1] - f[..., 2])
    return L, b

groups = {}
refA = [insta_crop(f"./data/refs/{n}.jpg") for n in range(202090, 202100)]
r1 = Image.open(f"{SP}/refs2/ref1.png").convert("RGB")
refB = [r1.crop((0, 0, 556, r1.height)), r1.crop((561, 0, r1.width, r1.height))] + \
       [Image.open(f"{SP}/refs2/ref{k}.png").convert("RGB") for k in (2, 3, 4, 5)]
groups["ref: ai_anime (10)"] = refA
groups["ref: new refs (6)"] = refB
groups["ours v3 anti-flat"] = panels(f"{OUT}/6aae0646-7ba4-83e8-bd7b-8096c83325ed-cec882309895.png")
groups["ours layered"] = panels(f"{OUT}/anime-crossing-layered.png")
groups["ours dynamic"] = panels(f"{OUT}/anime-crossing-dynamic.png")
groups["ours first"] = panels(f"{OUT}/6aadfb8d-1314-83ee-9f4d-b9dd85ecaeef-0-0.png")
groups["NG sample"] = [Image.open(f"{SP}/ng1.jpeg").convert("RGB")]
groups["ours v4 metric"] = panels(f"{OUT}/6aae0646-7ba4-83e8-bd7b-8096c83325ed-506c8206ac65.png")
groups["ours v5 contour"] = panels(f"{OUT}/6aae2119-6720-83e8-980f-44495600e1fb-69848209a2bd.png")
groups["ours v6 mid"] = panels(f"{OUT}/6aae2119-6720-83e8-980f-44495600e1fb-acac820691bf.png")
groups["ours v7"] = panels(f"{OUT}/6aae2119-6720-83e8-980f-44495600e1fb-5b908230b1bb.png")
groups["TARGET (19)"] = [Image.open(p).convert("RGB") for p in sorted(glob.glob(f"{SP}/target/t*.jpg"))]

from scipy import ndimage as ndi

def lab_full(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])

# GEOMETRY: height (depth) map, normal map, bump; how shading follows 3D form
import torch
from transformers import pipeline
est = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="mps" if torch.backends.mps.is_available() else "cpu")

def geo(im):
    im = im.resize((int(im.width * H / im.height), H)); rgb = np.asarray(im).astype(float)
    d = np.asarray(est(im)["predicted_depth"].squeeze().float().cpu().numpy()); d = np.asarray(Image.fromarray(d).resize(im.size))
    lo, hi = np.percentile(d, 1), np.percentile(d, 99); d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)   # 1 = near
    ds = ndi.gaussian_filter(d, 1.5); gy, gx = np.gradient(ds); k = im.width * 0.25
    n = np.dstack([-gx * k, -gy * k, np.ones_like(d)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    lin = np.where(rgb / 255 <= 0.04045, rgb / 255 / 12.92, ((rgb / 255 + 0.055) / 1.055) ** 2.4); Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    L, a, b = lab_full(rgb); m = d >= 0.10; r = {}
    nm = n[m]; mean_n = nm.mean(0); mean_n /= np.linalg.norm(mean_n)
    r["normal_spread_deg"] = float(np.degrees(np.arccos(np.clip(nm @ mean_n, -1, 1))).mean())
    best = (-1, 0, 0, None)
    for el in range(10, 90, 10):
        for az in range(0, 360, 30):
            l = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), -np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
            s = np.clip(nm @ l, 0, None); X = np.c_[np.ones_like(s), s]; c, *_ = np.linalg.lstsq(X, Y[m], rcond=None)
            res = Y[m] - X @ c; r2 = 1 - res.var() / (Y[m].var() + 1e-9)
            if r2 > best[0]: best = (r2, el, az, l)
    r["lambert_R2"], r["light_el"], r["light_az"] = float(best[0]), float(best[1]), float(best[2])
    ndl = nm @ best[3]; bins = np.linspace(-1, 1, 21); idx = np.digitize(ndl, bins)
    curve = np.array([L[m][idx == i].mean() if (idx == i).sum() > 50 else np.nan for i in range(1, 21)]); cv = curve[~np.isnan(curve)]
    r["cel_ness"] = float(np.abs(np.diff(cv)).max() / (np.abs(cv[-1] - cv[0]) / max(len(cv) - 1, 1) + 1e-6)) if len(cv) > 3 else np.nan
    r["shade_range"] = float(np.nanmax(cv) - np.nanmin(cv)) if len(cv) > 3 else np.nan                    # L* spread explained by orientation
    bump = d - ndi.gaussian_filter(d, 3); r["bump_x1000"] = float(bump[m].std() * 1000)
    Lh = L - ndi.gaussian_filter(L, 3)
    r["tex_follows_relief"] = float(np.corrcoef(Lh[m], bump[m])[0, 1])
    return r, n, d

per, viz = {}, {}
for g, ims in groups.items():
    rows = []
    for i, im in enumerate(ims):
        rr, n, d = geo(im); rows.append({k: float(v) for k, v in rr.items()})
        if i == 0: viz[g] = (im.resize((int(im.width * H / im.height), H)), n, d)
    per[g] = rows
json.dump(per, open(f"{SP}/geo_per_image.json", "w"))
# visual: RGB | height | normal for the first image of a few groups + the location photo
loc = Image.open("./data/refs/images (10).jpeg").convert("RGB"); _, nl, dl = geo(loc); viz["LOCATION"] = (loc.resize((int(loc.width * H / loc.height), H)), nl, dl)
cells = []
for g in ("LOCATION", "TARGET (19)", "ref: new refs (6)", "ours v7", "ours v6 mid"):
    im, n, d = viz[g]; t = 260; w = int(im.width * t / im.height)
    nrm = Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).resize((w, t)); hgt = Image.fromarray((d * 255).astype(np.uint8)).convert("RGB").resize((w, t))
    cells.append((g, im.resize((w, t)), hgt, nrm))
sheet = Image.new("RGB", (max(c[1].width for c in cells) * 3 + 40, len(cells) * 270), "white")
for i, (g, a, h2, nr) in enumerate(cells):
    sheet.paste(a, (0, i * 270)); sheet.paste(h2, (a.width + 10, i * 270)); sheet.paste(nr, (2 * a.width + 20, i * 270))
sheet.save("./output/analysis/normal-height-sheet.jpg", quality=85)
# encoding D control images for the location photo
Image.fromarray(((nl * 0.5 + 0.5) * 255).astype(np.uint8)).resize(loc.size).save("./output/encodings/normal_map.png")
Image.fromarray((dl * 255).astype(np.uint8)).resize(loc.size).save("./output/encodings/height_map.png")
print("ok")
