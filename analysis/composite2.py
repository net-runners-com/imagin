"""Composite v2: more operators, cached depth, coordinate-descent optimisation, full-resolution render."""
import sys, json
from composite import KEYS, RNG, score
from stylemetrics import *

def depth_near(im):
    d = np.asarray(est(im)["predicted_depth"].squeeze().float().cpu().numpy()); d = np.asarray(Image.fromarray(d).resize(im.size))
    lo, hi = np.percentile(d, 1), np.percentile(d, 99); return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)

def render(im, near, p):
    x = np.asarray(im).astype(float) / 255; lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    h, w = near.shape; k = h / 760
    t = np.clip((near - 0.55) / 0.45, 0, 1); t = t * t * (3 - 2 * t); lin *= (1 - p["fg"] * t)[..., None]
    yy, xx = np.mgrid[0:h, 0:w]; r2 = (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 2; lin *= (1 - p["v"] * r2)[..., None]
    if p["rim"] > 0:
        gy, gx = np.gradient(ndi.gaussian_filter(near, k)); edge = np.hypot(gx, gy) > 0.02 / k
        band = ndi.binary_dilation(edge, iterations=max(1, int(round(2 * k)))) & (near > ndi.uniform_filter(near, int(7 * k) | 1))
        lin[band] *= (1 - p["rim"])
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    if p["lc"] > 0:                                                                     # large-radius local contrast ("clarity")
        blur = ndi.gaussian_filter(s, (20 * k, 20 * k, 0)); s = s + p["lc"] * (s - blur)
    s = np.clip(s, 0, 1); s = np.clip(s - p["c"] * np.sin(2 * np.pi * s) / (2 * np.pi), 0, 1)
    if p["sat"] != 1:
        g = s.mean(2, keepdims=True); s = np.clip(g + p["sat"] * (s - g), 0, 1)
    return Image.fromarray((s * 255).astype(np.uint8))

GRID = {"fg": [0, .2, .35, .5, .65], "c": [0, .25, .5, .7, .9], "v": [0, .2, .35, .5, .65], "rim": [0, .2, .35, .5], "lc": [0, .15, .3, .5], "sat": [.85, .95, 1, 1.1]}
if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    full = Image.open(src).convert("RGB"); small = full.resize((int(full.width * H / full.height), H)); nearS = depth_near(small)
    ev = lambda p: score(measure(render(small, nearS, p)))
    key = lambda sc: (sc[0], -sc[1])
    p = {"fg": .45, "c": .5, "v": .35, "rim": .2, "lc": 0, "sat": 1}; best = ev(p); print("start", best, p, flush=True)
    for rnd in range(2):
        for name, vals in GRID.items():
            for v in vals:
                if v == p[name]: continue
                q = dict(p, **{name: v}); sc = ev(q)
                if key(sc) > key(best): best, p = sc, q; print("improved", best, p, flush=True)
    print("FINAL", best, p)
    nearF = depth_near(full); render(full, nearF, p).save(out)
    r = measure(render(small, nearS, p))
    for k in KEYS: print(f"{k:22}{RNG[k][0]:8.2f}–{RNG[k][1]:<8.2f}{r[k]:9.2f}  {'✓' if RNG[k][0] <= r[k] <= RNG[k][1] else '✗'}")
    json.dump({"src": src, "params": p, "score": best}, open(out + ".json", "w"))
