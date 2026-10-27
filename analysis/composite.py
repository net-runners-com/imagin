"""Local 2D composite pass driven by the depth map (never sent anywhere), grid-searched against the target metrics."""
import sys, json, math, itertools
from stylemetrics import *
T, DAY = "TARGET (19)", [0, 2, 4, 14, 16, 17]
src = {"d": "depth_per_image.json", "e": "edge_per_image.json", "v": "value_per_image.json", "s": "shader_per_image.json", "l": "light_per_image.json", "g": "geo_per_image.json"}
tgt = [{} for _ in DAY]
for p, f in src.items():
    J = json.load(open(f))[T]
    for j, i in enumerate(DAY): tgt[j].update({f"{p}_{k}": v for k, v in J[i].items()})
for t in tgt: t["rise"] = t["d_far_L"] - t["d_near_L"]; t["lift"] = t["d_far_p5"] - t["d_near_p5"]
KEYS = ["l_haze_dc","v_E_fine","v_quiet%","s_line%","s_line_width","s_line_C","s_glints/Mpx","v_L_std","v_dark%","v_C_light","v_hue_entropy","e_rim",
        "l_top_vs_mid","l_vignette","l_bloom","l_sh_chroma","s_hard_share%","g_cel_ness","g_normal_spread_deg","g_bump_x1000","rise","lift"]   # light_el dropped (estimator unreliable)
RNG = {k: (min(t[k] for t in tgt), max(t[k] for t in tgt), sum(t[k] for t in tgt) / len(tgt)) for k in KEYS}

def score(r):
    hits = sum(RNG[k][0] <= r[k] <= RNG[k][1] for k in KEYS)
    dist = sum(abs(r[k] - RNG[k][2]) / (RNG[k][1] - RNG[k][0] + 1e-9) for k in KEYS) / len(KEYS)
    return hits, dist

def composite(im, fg, c, v, rim):
    im = im.resize((int(im.width * H / im.height), H)); x = np.asarray(im).astype(float) / 255
    d = np.asarray(est(im)["predicted_depth"].squeeze().float().cpu().numpy()); d = np.asarray(Image.fromarray(d).resize(im.size))
    lo, hi = np.percentile(d, 1), np.percentile(d, 99); near = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    t = np.clip((near - 0.55) / 0.45, 0, 1); t = t * t * (3 - 2 * t); lin *= (1 - fg * t)[..., None]           # darken the nearest band
    h, w = near.shape; yy, xx = np.mgrid[0:h, 0:w]; r2 = (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 2
    lin *= (1 - v * r2)[..., None]                                                                               # vignette V(r)=1-v r^2
    if rim:                                                                                                      # knock back bright rims on the near side of depth contours
        gy, gx = np.gradient(ndi.gaussian_filter(near, 1)); edge = np.hypot(gx, gy) > 0.02
        band = ndi.binary_dilation(edge, iterations=2) & (near > ndi.uniform_filter(near, 7))
        lin[band] *= 0.8
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    s = np.clip(s - c * np.sin(2 * np.pi * s) / (2 * np.pi), 0, 1)                                               # S-curve contrast
    return Image.fromarray((s * 255).astype(np.uint8))

if __name__ == "__main__":
    base = Image.open(sys.argv[1]).convert("RGB"); out = sys.argv[2]
    r0 = measure(base.resize((int(base.width * H / base.height), H))); print("base", score(r0))
    best = None
    for fg, c, v, rim in itertools.product((0, 0.25, 0.45), (0, 0.25, 0.5), (0, 0.2, 0.35), (0, 1)):
        im = composite(base, fg, c, v, rim); sc = score(measure(im))
        if best is None or (sc[0], -sc[1]) > (best[0][0], -best[0][1]): best = (sc, (fg, c, v, rim), im)
    print("best", best[0], "params fg,c,v,rim =", best[1]); best[2].save(out)
    rb = measure(best[2])
    for k in KEYS: print(f"{k:22}{RNG[k][0]:8.2f}–{RNG[k][1]:<8.2f}{r0[k]:9.2f}{rb[k]:9.2f}  {'✓' if RNG[k][0] <= rb[k] <= RNG[k][1] else '✗'}")
