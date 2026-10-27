"""Composition (vanishing-point convergence) + terminator colour band + fog colour vs distance."""
import json, math, random
from stylemetrics import *
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

def comp(im):
    im = im.resize((int(im.width * H / im.height), H)); rgb = np.asarray(im).astype(float); L, a, b = lab_full(rgb); C = np.hypot(a, b)
    h, w = L.shape; r = {}
    segs = probabilistic_hough_line(canny(L, sigma=2), threshold=10, line_length=int(0.06 * h), line_gap=3, rng=0)
    S = []
    for (x0, y0), (x1, y1) in segs:
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
        if min(ang, 180 - ang) < 8 or abs(ang - 90) < 8: continue                     # drop near-horizontal / vertical
        S.append(((x0, y0), (x1, y1), math.hypot(x1 - x0, y1 - y0)))
    tot = sum(s[2] for s in S) + 1e-9; best = (0, None); random.seed(0)
    def inter(p, q):
        (x1, y1), (x2, y2) = p[0], p[1]; (x3, y3), (x4, y4) = q[0], q[1]; den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den; return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    for _ in range(400 if len(S) > 1 else 0):
        p, q = random.sample(S, 2); v = inter(p, q)
        if v is None or not (-0.5 * w < v[0] < 1.5 * w and -0.5 * h < v[1] < 1.5 * h): continue
        sup = 0
        for s in S:
            mx, my = (s[0][0] + s[1][0]) / 2, (s[0][1] + s[1][1]) / 2; dx, dy = s[1][0] - s[0][0], s[1][1] - s[0][1]
            ang = abs(math.degrees(math.atan2(dy, dx) - math.atan2(v[1] - my, v[0] - mx))) % 180
            if min(ang, 180 - ang) < 3: sup += s[2]
        if sup > best[0]: best = (sup, v)
    r["n_lines"] = len(S); r["vp_convergence"] = best[0] / tot if best[1] else float("nan")
    r["vp_x"], r["vp_y"] = (best[1][0] / w, best[1][1] / h) if best[1] else (float("nan"), float("nan"))
    # terminator colour band: chroma at strong light/shadow boundaries vs both sides
    g1 = ndi.gaussian_gradient_magnitude(L, 1.2); gy, gx = np.gradient(ndi.gaussian_filter(L, 1.2))
    ys, xs = np.nonzero(g1 > np.percentile(g1, 96)); gm = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-6; ux, uy = gx[ys, xs] / gm, gy[ys, xs] / gm
    at = lambda A, k: A[np.clip(np.rint(ys + k * uy).astype(int), 0, h - 1), np.clip(np.rint(xs + k * ux).astype(int), 0, w - 1)]
    cen, lit, sh = at(C, 0), at(C, 4), at(C, -4)
    r["terminator_C_boost"] = float((cen - np.maximum(lit, sh)).mean())                 # + = saturated band at the shadow edge
    hb = lambda k: np.degrees(np.arctan2(at(b, k).mean(), at(a, k).mean())) % 360
    r["terminator_hue"], r["shadow_side_hue"] = float(hb(0)), float(hb(-4))
    return r

if __name__ == "__main__":
    O = "./output/"
    tgt = [Image.open(p).convert("RGB") for i, p in enumerate(sorted(glob.glob("target/t*.jpg"))) if i in (0, 2, 4, 14, 16, 17)]
    sets = {"TARGET day": tgt, "v7 day": panels(O + "6aae2119-6720-83e8-980f-44495600e1fb-5b908230b1bb.png")[:2],
            "enc A": [Image.open(O + "encodings/6aae9c44-78e8-83ee-b33e-bb1c35a4cc62-b3108206a8c2.png").convert("RGB")],
            "enc B": [Image.open(O + "encodings/6aae9cd3-e3cc-83ee-a9e4-716b70d8f665-93548209831a.png").convert("RGB")],
            "location photo": [Image.open("./data/refs/images (10).jpeg").convert("RGB")]}
    res = {k: [comp(x) for x in v] for k, v in sets.items()}; json.dump(res, open("comp_per_image.json", "w"))
    for k in res["TARGET day"][0]:
        print(k.ljust(20) + "".join(f"{np.nanmean([x[k] for x in res[s]]):10.2f}" for s in sets), " | tgt each:", [round(x[k], 2) for x in res["TARGET day"]])
    print("cols:", list(sets))
    D = json.load(open("depth_per_image.json"))["TARGET (19)"]; print("target day near_b/far_b:", [(round(D[i]["near_b"], 1), round(D[i]["far_b"], 1)) for i in (0, 2, 4, 14, 16, 17)])
