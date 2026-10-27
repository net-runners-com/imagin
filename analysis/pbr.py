"""PBR-map-style decomposition estimated from a single 2D image (approximate)."""
import sys, json, math
from stylemetrics import *

def pbr(im):
    im = im.resize((int(im.width * H / im.height), H)); rgb = np.asarray(im).astype(float)
    L, a, b = lab_full(rgb); C = np.hypot(a, b)
    lin = np.where(rgb / 255 <= 0.04045, rgb / 255 / 12.92, ((rgb / 255 + 0.055) / 1.055) ** 2.4); Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    d = np.asarray(est(im)["predicted_depth"].squeeze().float().cpu().numpy()); d = np.asarray(Image.fromarray(d).resize(im.size))
    lo, hi = np.percentile(d, 1), np.percentile(d, 99); d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    sky, ground = d < 0.10, d >= 0.10; r = {}
    # ALBEDO vs SHADING (Retinex): large-scale log-luminance = illumination, residual = reflectance
    lY = np.log(Y + 1e-3); S = ndi.gaussian_filter(lY, 15); A = lY - S
    r["shading_share"] = float(S[ground].var() / (lY[ground].var() + 1e-9))          # 1 = lighting drives the image
    mx, mn = lin.max(2), lin.min(2); r["albedo_sat"] = float(((mx - mn) / (mx + 1e-4))[ground].mean())   # material colour, shading-free
    # ROUGHNESS / SPECULAR: luminance skew (Motoyoshi 2007) and desaturated highlight share
    g = L[ground]; r["lum_skew"] = float(((g - g.mean()) ** 3).mean() / (g.std() ** 3 + 1e-9))
    locL, locC = ndi.uniform_filter(L, 15), ndi.uniform_filter(C, 15)
    spec = ground & (L > locL + 15) & (C < locC * 0.8); r["spec_permil"] = float(1000 * spec.mean())
    # AO / CAVITY: darkening right behind occluding contours; small concave darkening
    gy, gx = np.gradient(ndi.gaussian_filter(d, 1)); gm = np.hypot(gx, gy); ys, xs = np.nonzero(gm > 0.02)
    if len(ys) > 50:
        ux, uy = gx[ys, xs] / gm[ys, xs], gy[ys, xs] / gm[ys, xs]
        at = lambda k: L[np.clip(np.rint(ys - k * uy).astype(int), 0, H - 1), np.clip(np.rint(xs - k * ux).astype(int), 0, im.width - 1)]
        r["AO"] = float((at(2) - at(10)).mean())                                        # negative = occlusion shadow behind objects
    else: r["AO"] = float("nan")
    r["cavity"] = float((ndi.grey_closing(L, size=7) - L)[ground].mean())
    # EMISSIVE: bright + saturated warm pixels (lamps, lit windows, sun glow)
    r["emissive_permil"] = float(1000 * ((L > 80) & (C > 35) & ((a > 0) | (b > 30))).mean())
    # TRANSLUCENCY / SSS: backlit glowing foliage share among foliage pixels
    hue = np.degrees(np.arctan2(b, a)) % 360; fol = ground & (hue > 95) & (hue < 170) & (C > 15)
    r["translucent%"] = float(100 * ((L > 65) & (C > 35))[fol].mean()) if fol.sum() > 200 else float("nan")
    # RAMP (toon): colour and softness of the lit→shadow transition along N·L
    gr, n, _ = geo(im); el, az = math.radians(gr["light_el"]), math.radians(gr["light_az"])
    l = np.array([math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), math.sin(el)]); ndl = (n @ l)[ground]
    litm, shm = ndl > np.percentile(ndl, 70), ndl < np.percentile(ndl, 30)
    La, aa, ba = L[ground], a[ground], b[ground]
    r["ramp_dL"] = float(La[litm].mean() - La[shm].mean())
    r["ramp_hue_shift"] = float((math.degrees(math.atan2(ba[shm].mean(), aa[shm].mean())) - math.degrees(math.atan2(ba[litm].mean(), aa[litm].mean())) + 180) % 360 - 180)
    # FLOW MAP: how strongly the sky/cloud strokes flow in one direction
    sx, sy = ndi.sobel(ndi.gaussian_filter(L, 1.5), 1), ndi.sobel(ndi.gaussian_filter(L, 1.5), 0); th = np.arctan2(sy, sx); w = np.hypot(sx, sy)
    msk = sky & (w > np.percentile(w[sky], 60)) if sky.sum() > 500 else sky
    r["sky_flow"] = float(np.hypot((w[msk] * np.cos(2 * th[msk])).sum(), (w[msk] * np.sin(2 * th[msk])).sum()) / (w[msk].sum() + 1e-9)) if msk.sum() > 200 else float("nan")
    # DISPLACEMENT: skyline roughness (raw perimeter / smoothed perimeter)
    per = lambda m: float((m ^ ndi.binary_erosion(m)).sum())
    r["silhouette_rough"] = per(sky) / max(per(ndi.gaussian_filter(sky.astype(float), 4) > 0.5), 1) if sky.sum() > 500 else float("nan")
    # OPACITY / ALPHA: see-through gaps in foliage/fences (small sky holes)
    lab_, nlab = ndi.label(sky); sz = np.bincount(lab_.ravel())[1:] if nlab else np.array([])
    r["alpha_gaps/Mpx"] = float(((sz > 3) & (sz < 0.001 * L.size)).sum() / (L.size / 1e6))
    return r

if __name__ == "__main__":
    SP = "."; T = "TARGET (19)"
    tgt = [Image.open(p).convert("RGB") for i, p in enumerate(sorted(glob.glob("target/t*.jpg"))) if i in (0, 2, 4, 14, 16, 17)]
    O = "./output/"
    sets = {"TARGET day": tgt, "v6 day": panels(O + "6aae2119-6720-83e8-980f-44495600e1fb-acac820691bf.png")[:2],
            "v7 day": panels(O + "6aae2119-6720-83e8-980f-44495600e1fb-5b908230b1bb.png")[:2],
            "enc A": [Image.open(O + "encodings/6aae9c44-78e8-83ee-b33e-bb1c35a4cc62-b3108206a8c2.png").convert("RGB")],
            "enc B": [Image.open(O + "encodings/6aae9cd3-e3cc-83ee-a9e4-716b70d8f665-93548209831a.png").convert("RGB")]}
    res = {k: [pbr(x) for x in v] for k, v in sets.items()}; json.dump(res, open("pbr_per_image.json", "w"))
    keys = list(res["TARGET day"][0])
    print("metric".ljust(18) + "tgt mean".rjust(9) + "tgt range".rjust(16) + "".join(k.rjust(9) for k in list(sets)[1:]))
    for k in keys:
        tv = [x[k] for x in res["TARGET day"] if x[k] == x[k]]; mu = sum(tv) / len(tv)
        vals = [np.nanmean([x[k] for x in res[s]]) for s in list(sets)[1:]]
        print(k.ljust(18) + f"{mu:9.2f}" + f"{min(tv):8.2f}–{max(tv):<7.2f}" + "".join(f"{v:9.2f}" for v in vals))
