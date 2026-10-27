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

def otsu(L):
    h, e = np.histogram(L, bins=100, range=(0, 100)); p = h / h.sum(); c = (e[:-1] + e[1:]) / 2
    best, t = -1, 50
    for i in range(1, 100):
        w0, w1 = p[:i].sum(), p[i:].sum()
        if w0 == 0 or w1 == 0: continue
        m0, m1 = (p[:i] * c[:i]).sum() / w0, (p[i:] * c[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best: best, t = v, c[i]
    return t, best / (L.var() + 1e-9)

def stats(im):
    im = im.resize((int(im.width * H / im.height), H))
    L, a, b = lab_full(np.asarray(im).astype(float)); C = np.hypot(a, b)
    r = {}
    # value distribution
    r["L_mean"], r["L_std"] = L.mean(), L.std()
    r["L_range"] = np.percentile(L, 99) - np.percentile(L, 1)
    r["dark%"], r["mid%"], r["light%"] = 100 * (L < 30).mean(), 100 * ((L >= 30) & (L <= 70)).mean(), 100 * (L > 70).mean()
    # notan: how cleanly the picture splits into 2 value masses, and how fragmented the masses are
    Ls = ndi.gaussian_filter(L, 4); t, r["notan_sep"] = otsu(Ls)
    for name, m in (("dark", Ls < t), ("light", Ls >= t)):
        lab, n = ndi.label(m); sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
        r[f"{name}_blobs"] = int((sizes > 0.001 * L.size).sum())
        r[f"{name}_top%"] = 100 * sizes.max() / max(m.sum(), 1)
    # highlight / shadow budget and focal concentration
    r["hi%"], r["clip%"], r["crush%"] = 100 * (L > 90).mean(), 100 * (L > 98).mean(), 100 * (L < 3).mean()
    ys, xs = np.nonzero(L >= np.percentile(L, 98))
    r["hi_spread"] = np.hypot(ys.std(), xs.std()) / np.hypot(*L.shape)
    # colour
    r["C_mean"] = C.mean()
    r["C_shadow"] = C[L < 30].mean() if (L < 30).sum() > 50 else np.nan
    r["C_light"] = C[L > 70].mean() if (L > 70).sum() > 50 else np.nan
    r["b_shadow"] = b[L < 30].mean() if (L < 30).sum() > 50 else np.nan
    r["b_light"] = b[L > 70].mean() if (L > 70).sum() > 50 else np.nan
    r["temp_split"] = r["b_light"] - r["b_shadow"]
    hue = np.degrees(np.arctan2(b, a))[C > 10]
    p = np.histogram(hue, bins=36, range=(-180, 180))[0]; p = p[p > 0] / p.sum()
    r["hue_entropy"] = float(-(p * np.log2(p)).sum())
    # detail vs mass: band energies of L (fine texture / medium / big shapes)
    g1, g4, g16 = ndi.gaussian_filter(L, 1.5), ndi.gaussian_filter(L, 6), ndi.gaussian_filter(L, 24)
    r["E_fine"], r["E_mid"], r["E_coarse"] = np.abs(L - g1).mean(), np.abs(g1 - g4).mean(), np.abs(g4 - g16).mean()
    r["fine/coarse"] = r["E_fine"] / r["E_coarse"]
    # quiet areas: local std under 2.5 L* in a 9x9 window
    m1 = ndi.uniform_filter(L, 9); m2 = ndi.uniform_filter(L * L, 9)
    r["quiet%"] = 100 * (np.sqrt(np.clip(m2 - m1 * m1, 0, None)) < 2.5).mean()
    return {k: float(v) for k, v in r.items()}

per = {g: [stats(im) for im in ims] for g, ims in groups.items()}
json.dump(per, open(f"{SP}/value_per_image.json", "w"))
day = {"ref: ai_anime (10)": [0, 2, 3, 5, 6, 8], "ref: new refs (6)": [1, 2, 3, 5],
       "ours v3 anti-flat": [0, 1], "ours layered": [0, 1], "ours dynamic": [0, 1], "ours first": [0, 1], "NG sample": [0], "ours v4 metric": [0, 1], "ours v5 contour": [0, 1], "ours v6 mid": [0, 1], "ours v7": [0, 1], "TARGET (19)": [0, 2, 4, 14, 16, 17]}
keys = list(per[next(iter(per))][0])
print("DAYTIME".ljust(12) + "".join(g.replace("ref: ", "").replace("ours ", "")[:11].rjust(12) for g in day))
for k in keys:
    vals = []
    for g, idx in day.items():
        v = [per[g][i][k] for i in idx if per[g][i][k] == per[g][i][k]]
        vals.append(sum(v) / len(v) if v else float("nan"))
    print(k.ljust(12) + "".join(f"{x:12.2f}" for x in vals))
