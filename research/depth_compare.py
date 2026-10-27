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
import torch
from transformers import pipeline

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

dev = "mps" if torch.backends.mps.is_available() else "cpu"
est = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)

BANDS = [("sky", 0.0, 0.10), ("far", 0.10, 0.35), ("mid", 0.35, 0.65), ("near", 0.65, 1.01)]

def analyse(im):
    im = im.resize((int(im.width * H / im.height), H))
    d = np.asarray(est(im)["predicted_depth"].squeeze().float().cpu().numpy())
    d = np.asarray(Image.fromarray(d).resize(im.size))
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    L, b = srgb_to_lab(np.asarray(im).astype(float))
    gy, gx = np.gradient(L); g = np.hypot(gx, gy)
    r = {"depth_std": float(d.std())}
    ground = d >= 0.10
    r["corr_depth_L"] = float(np.corrcoef(d[ground], L[ground])[0, 1]) if ground.sum() > 100 else float("nan")
    r["shadow_share"] = float((L[ground] < 25).mean()) if ground.any() else float("nan")
    for name, a, z in BANDS:
        m = (d >= a) & (d < z)
        if m.sum() < 200:
            for k in ("area", "L", "p5", "p95", "con", "b", "grad"): r[f"{name}_{k}"] = float("nan")
            continue
        Lm = L[m]
        r.update({f"{name}_area": float(m.mean()), f"{name}_L": float(Lm.mean()),
                  f"{name}_p5": float(np.percentile(Lm, 5)), f"{name}_p95": float(np.percentile(Lm, 95)),
                  f"{name}_con": float(np.percentile(Lm, 95) - np.percentile(Lm, 5)),
                  f"{name}_b": float(b[m].mean()), f"{name}_grad": float(g[m].mean())})
    return r, im, d

res, viz, per = {}, [], {}
for gname, ims in groups.items():
    rows = []
    for i, im in enumerate(ims):
        r, small, d = analyse(im); rows.append(r)
        if i < 3: viz.append((gname, small, d))
    per[gname] = rows
    res[gname] = {k: float(np.nanmean([x[k] for x in rows])) for k in rows[0]}
    res[gname]["n"] = len(rows)
json.dump(res, open(f"{SP}/depth_metrics.json", "w"), indent=1); json.dump(per, open(f"{SP}/depth_per_image.json", "w"))

# contact sheet: image + depth (grey, white = near) for up to 3 per group
th = 300
cells = []
for gname, small, d in viz:
    a = small.resize((int(small.width * th / small.height), th))
    dm = Image.fromarray((d * 255).astype(np.uint8)).resize(a.size).convert("RGB")
    cells.append((a, dm))
cw = max(a.width for a, _ in cells) * 2 + 12
per_row = 6
rows_n = (len(cells) + per_row - 1) // per_row
sheet = Image.new("RGB", (per_row * cw, rows_n * (th + 10)), "white")
for i, (a, dm) in enumerate(cells):
    x, y = (i % per_row) * cw, (i // per_row) * (th + 10)
    sheet.paste(a, (x, y)); sheet.paste(dm, (x + a.width + 4, y))
sheet.save(f"{OUT}/depth-compare-sheet.jpg", quality=85)
print("ok")
