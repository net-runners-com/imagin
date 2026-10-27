"""All style metrics in one module (assembled from depth_compare/edges/metrics2-5)."""

import json, os, sys, math, glob

import numpy as np

from PIL import Image

from scipy import ndimage as ndi

import torch

from transformers import pipeline

H = 760

est = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device='mps' if torch.backends.mps.is_available() else 'cpu')

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

def srgb_to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    b = 200 * (f[..., 1] - f[..., 2])
    return L, b

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

THR = 0.02

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

def coherence(L, s=1.5, r=4):
    gx, gy = ndi.sobel(ndi.gaussian_filter(L, s), 1), ndi.sobel(ndi.gaussian_filter(L, s), 0)
    jxx, jyy, jxy = (ndi.gaussian_filter(v, r) for v in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy; det = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return np.where(tr > 1e-6, det / (tr + 1e-6), 0)

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


def measure(im):
    r = {}
    a, _, _ = analyse(im); r.update({f"d_{k}": v for k, v in a.items()})
    r.update({f"e_{k}": v for k, v in edge_stats(im).items()})
    r.update({f"v_{k}": v for k, v in stats(im).items()})
    r.update({f"s_{k}": v for k, v in stats3(im).items()})
    r.update({f"l_{k}": v for k, v in stats4(im).items()})
    g, _, _ = geo(im); r.update({f"g_{k}": float(v) for k, v in g.items()})
    r["rise"] = r["d_far_L"] - r["d_near_L"]; r["lift"] = r["d_far_p5"] - r["d_near_p5"]
    return r
