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

# LINE / TOUCH / SHADER metrics (all on CIELAB L*, image scaled to height 760)
def coherence(L, s=1.5, r=4):
    gx, gy = ndi.sobel(ndi.gaussian_filter(L, s), 1), ndi.sobel(ndi.gaussian_filter(L, s), 0)
    jxx, jyy, jxy = (ndi.gaussian_filter(v, r) for v in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy; det = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return np.where(tr > 1e-6, det / (tr + 1e-6), 0)          # 0 isotropic … 1 one direction

def stats3(im):
    im = im.resize((int(im.width * H / im.height), H))
    rgb = np.asarray(im).astype(float); L, a, b = lab_full(rgb); C = np.hypot(a, b); r = {}
    coh = coherence(L)
    # LINE: thin dark strokes = black-hat response (closing minus image)
    bh5 = ndi.grey_closing(L, size=5) - L; bh9 = ndi.grey_closing(L, size=9) - L
    line = (bh5 > 12) & (L < 45); line9 = (bh9 > 12) & (L < 45)
    r["line%"] = 100 * line.mean()
    r["line_L"] = L[line].mean() if line.sum() > 50 else np.nan          # how dark the lines are
    r["line_C"] = C[line].mean() if line.sum() > 50 else np.nan          # coloured vs black lines
    r["line_width"] = line9.sum() / max(line.sum(), 1)                    # >1 = thicker strokes
    lab_, n = ndi.label(line); r["line_len"] = float(np.bincount(lab_.ravel())[1:].mean()) if n else 0.0
    r["line_straight"] = coh[line].mean() if line.sum() > 50 else np.nan
    # TOUCH
    m1 = ndi.uniform_filter(L, 9); m2 = ndi.uniform_filter(L * L, 9); lstd = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    quiet = lstd < 2.5
    r["grain"] = float(np.abs(L - ndi.median_filter(L, 3))[quiet].mean()) if quiet.sum() > 500 else np.nan
    E = [np.abs(ndi.gaussian_filter(L, s) - ndi.gaussian_filter(L, 2 * s)).mean() for s in (1, 2, 4, 8, 16)]
    r["tex_peak_sigma"] = (1, 2, 4, 8, 16)[int(np.argmax(np.array(E) / np.array([1, 1.1, 1.2, 1.3, 1.4])))]
    tex = (lstd > 4) & ~line
    r["stroke_dir"] = coh[tex].mean() if tex.sum() > 500 else np.nan    # directional brushwork
    q = (rgb // 16).astype(int); key = q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]
    cnt = np.bincount(key.ravel(), minlength=4096); r["colors"] = int((cnt > 0.0001 * key.size).sum())
    lap = np.abs(ndi.laplace(ndi.gaussian_filter(L, 0.7)))
    th, tw = L.shape[0] // 12, L.shape[1] // 8; sh, busy = [], []
    for i in range(12):
        for j in range(8):
            t = L[i*th:(i+1)*th, j*tw:(j+1)*tw]; sh.append(lap[i*th:(i+1)*th, j*tw:(j+1)*tw].var()); busy.append(t.std() > 6)
    sh, busy = np.array(sh), np.array(busy)
    r["dof%"] = 100 * float((sh[busy] < 0.15 * np.percentile(sh[busy], 90)).mean()) if busy.sum() > 5 else np.nan
    # SHADER
    g1 = ndi.gaussian_gradient_magnitude(L, 1); g3 = ndi.gaussian_gradient_magnitude(L, 3)
    top = g1 > np.percentile(g1, 95)
    r["edge_hard"] = float((g1[top] / (g3[top] + 1e-6)).mean())          # ~1 soft ramps … >2 hard steps
    sig = g1 > 2; r["hard_share%"] = 100 * float((g1[sig] > 8).mean()) if sig.any() else np.nan
    h = ndi.gaussian_filter1d(np.histogram(L, bins=100, range=(0, 100))[0].astype(float), 1.5)
    pk = [i for i in range(1, 99) if h[i] > h[i-1] and h[i] >= h[i+1] and h[i] > 0.004 * h.sum()]
    r["tone_steps"] = len(pk)
    br = L > 92; lb, nb = ndi.label(br); sizes = np.bincount(lb.ravel())[1:] if nb else np.array([])
    r["glints/Mpx"] = float(((sizes >= 3) & (sizes < 0.0005 * L.size)).sum() / (L.size / 1e6))
    return {k: float(v) for k, v in r.items()}

per = {g: [stats3(im) for im in ims] for g, ims in groups.items()}
json.dump(per, open(f"{SP}/shader_per_image.json", "w"))
print("ok")
