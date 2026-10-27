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

# LIGHTING / POST (bloom, vignette, black/white point, split-tone, haze, CA, dynamic range)
def stats4(im):
    im = im.resize((int(im.width * H / im.height), H))
    rgb = np.asarray(im).astype(float); L, a, b = lab_full(rgb); r = {}
    lin = np.where(rgb / 255 <= 0.04045, rgb / 255 / 12.92, ((rgb / 255 + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    r["DR_stops"] = float(np.log2((np.percentile(Y, 99.5) + 1e-4) / (np.percentile(Y, 0.5) + 1e-4)))
    r["black_pt"], r["white_pt"] = float(np.percentile(L, 0.5)), float(np.percentile(L, 99.5))
    hi = (L > 85).sum(); r["shoulder%"] = 100 * float((L > 97).sum() / max(hi, 1))   # share of bright px that clip
    # bloom: glow outside bright sources (distance rings from L>90 regions)
    src = L > 90
    if src.sum() > 30:
        dist = ndi.distance_transform_edt(~src)
        near, far = (dist > 1) & (dist <= 6), (dist > 20) & (dist <= 35)
        r["bloom"] = float(L[near].mean() - L[far].mean()) if near.any() and far.any() else np.nan
    else: r["bloom"] = np.nan
    # vignette: corners vs centre (radial)
    h, w = L.shape; yy, xx = np.mgrid[0:h, 0:w]
    rad = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2)) / np.sqrt(2)
    r["vignette"] = float(np.median(L[rad < 0.35]) - np.median(L[rad > 0.85]))     # + = darker corners
    top, bot = L[: h // 10], L[-h // 10:]
    r["top_vs_mid"] = float(np.median(L[h // 3: 2 * h // 3]) - np.median(top))    # sky darkening toward top (gradient overlay)
    # split toning: hue & chroma of shadows vs highlights
    for nm, m in (("sh", L < 25), ("hl", L > 75)):
        if m.sum() > 100:
            r[f"{nm}_hue"] = float(np.degrees(np.arctan2(b[m].mean(), a[m].mean())) % 360)
            r[f"{nm}_chroma"] = float(np.hypot(a[m].mean(), b[m].mean()))
        else: r[f"{nm}_hue"] = r[f"{nm}_chroma"] = np.nan
    # haze: dark-channel prior (He et al.), higher = more atmospheric veil
    dc = ndi.minimum_filter(rgb.min(axis=2), size=15) / 255
    r["haze_dc"] = float(np.median(dc[h // 4:]))                                    # below the top quarter (mostly ground)
    # chromatic aberration: R vs B edge offset near the borders
    gR, gB = ndi.gaussian_gradient_magnitude(rgb[..., 0], 1), ndi.gaussian_gradient_magnitude(rgb[..., 2], 1)
    band = rad > 0.7; best, bs = 0, -1
    for dx in (-2, -1, 0, 1, 2):
        c = np.corrcoef(gR[band], np.roll(gB, dx, axis=1)[band])[0, 1]
        if c > bs: bs, best = c, dx
    r["CA_px"] = float(abs(best))
    lamps = ndi.label(L > 95)[0]; sz = np.bincount(lamps.ravel())[1:]
    r["light_srcs"] = int(((sz > 20) & (sz < 0.02 * L.size)).sum())
    return {k: float(v) for k, v in r.items()}

per = {g: [stats4(im) for im in ims] for g, ims in groups.items()}
json.dump(per, open(f"{SP}/light_per_image.json", "w")); print("ok")
