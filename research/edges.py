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

# Occlusion contours = depth discontinuities. For each contour pixel, step along the depth
# gradient (u points to the NEARER side) and compare the near side with the far side.
THR = 0.02   # |grad d| per px; a continuous ground slope is ~0.003, a silhouette jump ~0.1+

def edge_stats(im):
    im = im.resize((int(im.width * H / im.height), H))
    d = np.asarray(est(im)["predicted_depth"].squeeze().float().cpu().numpy())
    d = np.asarray(Image.fromarray(d).resize(im.size))
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    L, b = srgb_to_lab(np.asarray(im).astype(float))
    gy, gx = np.gradient(d); gm = np.hypot(gx, gy)
    Ly, Lx = np.gradient(L); Lg = np.hypot(Lx, Ly)
    h, w = d.shape
    # thin: keep local maxima of |grad d| along the gradient direction
    ys, xs = np.nonzero(gm > THR)
    ux, uy = gx[ys, xs] / gm[ys, xs], gy[ys, xs] / gm[ys, xs]
    def at(a, k, sign=1):
        yy = np.clip(np.rint(ys + sign * k * uy).astype(int), 0, h - 1)
        xx = np.clip(np.rint(xs + sign * k * ux).astype(int), 0, w - 1)
        return a[yy, xx]
    keep = (gm[ys, xs] >= at(gm, 1, 1)) & (gm[ys, xs] >= at(gm, 1, -1))
    ys, xs, ux, uy = ys[keep], xs[keep], ux[keep], uy[keep]
    edge_frac = len(ys) / (h * w)
    jump = at(d, 8, 1) - at(d, 8, -1)                     # depth step across the contour
    near3, far3 = at(L, 3, 1), at(L, 3, -1)
    near1, near7 = at(L, 1, 1), at(L, 7, 1)
    far1, far7 = at(L, 1, -1), at(L, 7, -1)
    dL = far3 - near3                                     # + = far side lighter than the object
    db = at(b, 3, -1) - at(b, 3, 1)                       # - = far side bluer
    rim = near1 - near7                                   # + = light rim on the object's edge
    halo = far1 - far7                                    # + = glow on the background next to it
    crisp = Lg[ys, xs].mean() / (Lg.mean() + 1e-6)        # contour gradient vs average texture gradient
    sky = at(d, 8, -1) < 0.10                             # contour against the sky vs against other objects
    obj = ~sky
    f = lambda a, m: float(a[m].mean()) if m.sum() > 30 else float("nan")
    return {"edge_frac%": 100 * edge_frac, "jump": f(jump, obj | sky),
            "sep|dL|": f(np.abs(dL), obj | sky), "dL_obj": f(dL, obj), "dL_sky": f(dL, sky),
            "db_obj": f(db, obj), "rim": f(rim, obj | sky), "halo": f(halo, obj | sky),
            "crisp": float(crisp), "obj_share": float(obj.mean()) if len(obj) else float("nan")}

per = {}
for g, ims in groups.items():
    per[g] = [edge_stats(im) for im in ims]
json.dump(per, open(f"{SP}/edge_per_image.json", "w"))
day = {"ref: ai_anime (10)": [0, 2, 3, 5, 6, 8], "ref: new refs (6)": [1, 2, 3, 5],
       "ours v3 anti-flat": [0, 1], "ours layered": [0, 1], "ours dynamic": [0, 1], "ours first": [0, 1], "NG sample": [0], "ours v4 metric": [0, 1], "ours v5 contour": [0, 1], "ours v6 mid": [0, 1], "ours v7": [0, 1], "TARGET (19)": [0, 2, 4, 14, 16, 17]}
keys = list(per[next(iter(per))][0])
for title, sel in (("ALL", {g: range(len(v)) for g, v in per.items()}), ("DAYTIME", day)):
    print(title.ljust(12) + "".join(g.replace("ref: ", "").replace("ours ", "")[:11].rjust(12) for g in sel))
    for k in keys:
        vals = []
        for g, idx in sel.items():
            v = [per[g][i][k] for i in idx if per[g][i][k] == per[g][i][k]]
            vals.append(sum(v) / len(v) if v else float("nan"))
        print(k.ljust(12) + "".join(f"{x:12.2f}" for x in vals))
