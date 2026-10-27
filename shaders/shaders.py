"""Procedural 2D shaders for top-down / painted terrain (numpy only, CPU).

Extracted from the golf-course renderer (net-runners-com/golf-course-photoreal).
Everything works on float32 arrays shaped (H, W) or (H, W, 3) in 0..1.

Building blocks
  fbm(shape, scale)          multi-octave value noise, zero-mean / unit-std
  blades(shape, length)      anisotropic "grass blade" noise (white noise smeared along random directions)
  domain_warp(mask, ...)     organic boundaries: resample a label map through a noise displacement field
  soft_upscale(labels, ...)  low-res label map -> smooth high-res label map (per-class blur + argmax)
  height_to_normal(h)        height map -> unit normals (bump / normal map)
  lambert(n, light)          N.L shading
  hillshade(h, light)        height -> shade, normalised so the median stays ~1
  stripes(shape, axis, period)  mowing stripes along an axis
  tint(base, warm, cool, t)  hue drift between a warm and a cool variant of a colour
  grass / sand / tree_canopy  material shaders built from the above
  cast_shadow(alpha, height, offset)  soft directional shadow of an object layer
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

LIGHT_UPPER_LEFT = np.array([-0.55, -0.6, 0.58]) / np.linalg.norm([-0.55, -0.6, 0.58])


def fbm(shape, scale, octaves=4, seed=None, gain=0.55):
    H, W = shape
    r = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = scale / (2 ** o)
        k = max(int(s / 3), 1)
        n = r.standard_normal((H // k + 2, W // k + 2)).astype(np.float32)
        n = np.asarray(Image.fromarray(n).resize((W, H), Image.BICUBIC), np.float32)
        out += amp * ndi.gaussian_filter(n, max(s / 6, 0.6))
        tot += amp
        amp *= gain
    out /= tot
    return (out - out.mean()) / (out.std() + 1e-6)


def blades(shape, length=3.0, n_dirs=4, seed=None):
    r = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    for _ in range(n_dirs):
        w = r.standard_normal(shape).astype(np.float32)
        ang = r.uniform(0, 180)
        w = ndi.rotate(w, ang, reshape=False, order=1, mode='wrap')
        w = ndi.gaussian_filter(w, (length, 0.5))
        out += ndi.rotate(w, -ang, reshape=False, order=1, mode='wrap')
    return (out - out.mean()) / (out.std() + 1e-6)


def soft_upscale(labels, scale, sigma=9, boost=None):
    """labels: int (h, w) -> int (h*scale, w*scale) with smooth boundaries. boost: {class: weight}."""
    h, w = labels.shape
    H, W = h * scale, w * scale
    classes = np.unique(labels)
    soft = []
    for c in classes:
        big = np.asarray(Image.fromarray(((labels == c) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255
        soft.append(ndi.gaussian_filter(big, sigma) * (boost or {}).get(int(c), 1.0))
    return classes[np.stack(soft).argmax(0)]


def domain_warp(labels, amp=6, scale=90, seed=101, keep=()):
    """Displace a label map by low-frequency noise. Classes in `keep` keep their exact shape.
    Keep `scale` large: a fine warp (~8px) eats the edges and makes regions unreadable."""
    H, W = labels.shape
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    wy, wx = amp * fbm((H, W), scale, 3, seed), amp * fbm((H, W), scale, 3, seed + 2)
    out = ndi.map_coordinates(labels, [yy + wy, xx + wx], order=0, mode='nearest')
    if keep:
        k = np.isin(labels, keep)
        moved = np.isin(out, keep) & ~k
        out[moved] = labels[moved]
        out[k] = labels[k]
    return out


def height_to_normal(h, strength=1.0):
    gy, gx = np.gradient(h * strength)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def normal_map_rgb(n):
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def lambert(n, light=LIGHT_UPPER_LEFT):
    return np.clip((n * light).sum(-1), 0, 1)


def hillshade(h, light=LIGHT_UPPER_LEFT, strength=1.8, mask=None):
    """Note: z of the normal is fixed at 1/strength-scaled slope; returns shade ~1 at the median."""
    s = lambert(height_to_normal(h, strength), light)
    ref = np.percentile(s[mask] if mask is not None else s, 60)
    return s / (ref + 1e-6)


def stripes(shape, axis_dxdy, period, soft=3, sharp=2.5, wobble=None):
    H, W = shape
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    ax = np.asarray(axis_dxdy, np.float32)
    ax /= np.linalg.norm(ax)
    perp = -xx * ax[1] + yy * ax[0]
    ph = perp / period * np.pi + (0 if wobble is None else wobble)
    return np.tanh(ndi.gaussian_filter(np.sin(ph), soft) * sharp)


def tint(base, warm, cool, t):
    base, warm, cool = (np.asarray(c, np.float32) / 255 for c in (base, warm, cool))
    t = np.clip(t, -1, 1)[..., None]
    return base + np.clip(t, 0, 1) * (warm - base) + np.clip(-t, 0, 1) * (cool - base)


def grass(shape, base, warm, cool, hue_t, big, mid, blade, var=(0.04, 0.035, 0.07)):
    """hue_t, big, mid, blade: noise fields (H, W). var = amplitudes for big / mid / blade."""
    col = tint(base, warm, cool, hue_t)
    return col * (1 + var[0] * big + var[1] * mid + var[2] * blade)[..., None]


def sand(shape, color=(232, 220, 186), seed=20, rake_axis=(0.9, 0.3)):
    H, W = shape
    g1 = np.random.default_rng(seed).standard_normal((H, W)).astype(np.float32)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    rake = np.sin((xx * rake_axis[0] + yy * rake_axis[1]) / 4.5 + 3 * fbm(shape, 70, 2, seed + 1))
    c = np.asarray(color, np.float32) / 255
    return c * (1 + 0.05 * g1 + 0.04 * fbm(shape, 90, 3, seed + 2) + 0.02 * rake)[..., None]


def tree_canopy(mask, r_range=(24, 48), seed=7, light=LIGHT_UPPER_LEFT,
                species=((40, 74, 30), (52, 84, 30), (34, 64, 34), (62, 90, 36), (46, 70, 40))):
    """Pack crowns (each = dome + 8-14 leaf clusters) into `mask`. Returns color (H,W,3), alpha, height."""
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    tdist = ndi.distance_transform_edt(mask)
    color = np.zeros((H, W, 3), np.float32)
    height = np.zeros((H, W), np.float32)
    alpha = np.zeros((H, W), np.float32)
    cand = np.argwhere(mask & (tdist > 6))
    rng.shuffle(cand)
    placed = []
    for y, x in cand[:8000]:
        r = min(rng.uniform(*r_range), tdist[y, x] + 14)
        if any((y - py) ** 2 + (x - px) ** 2 < (0.6 * (r + pr)) ** 2 for py, px, pr in placed):
            continue
        placed.append((y, x, r))
    for y, x, r in placed:
        base = np.array(species[rng.integers(len(species))], np.float32) / 255 * rng.uniform(0.9, 1.1)
        subs = [(y, x, r * 0.75)] + [(y + r * 0.55 * np.sin(a), x + r * 0.55 * np.cos(a), r * rng.uniform(0.35, 0.55))
                                     for a in rng.uniform(0, 2 * np.pi, rng.integers(8, 15))]
        for sy, sx, sr in subs:
            ya, yb = int(max(sy - sr - 2, 0)), int(min(sy + sr + 3, H))
            xa, xb = int(max(sx - sr - 2, 0)), int(min(sx + sr + 3, W))
            if ya >= yb or xa >= xb:
                continue
            py, px = np.mgrid[ya:yb, xa:xb].astype(np.float32)
            d = np.sqrt((py - sy) ** 2 + (px - sx) ** 2) / sr
            inside = d < 1
            z = np.sqrt(np.clip(1 - d ** 2, 0, 1))
            dome = np.sqrt(np.clip(1 - ((py - y) ** 2 + (px - x) ** 2) / (r * 1.15) ** 2, 0, 1))
            nx = (px - sx) / sr * 0.6 + (px - x) / r * 0.4
            ny = (py - sy) / sr * 0.6 + (py - y) / r * 0.4
            nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0.05, 1))
            lam = np.clip(-light[0] * nx - light[1] * ny + light[2] * nz, 0, 1)
            hz = r * (0.7 + 0.5 * dome) + sr * 0.4 * z
            upd = inside & (hz > height[ya:yb, xa:xb])
            col = base[None, None] * (0.38 + 0.95 * lam[..., None])
            color[ya:yb, xa:xb][upd] = col[upd]
            height[ya:yb, xa:xb][upd] = hz[upd]
            alpha[ya:yb, xa:xb][upd] = 1
    leaf, leaf2 = fbm((H, W), 3, 1, seed + 23), fbm((H, W), 9, 2, seed + 24)
    color *= (1 + 0.16 * leaf + 0.14 * leaf2)[..., None]
    lm = leaf2 + 0.6 * leaf
    color[(lm > 1.6) & (alpha > 0)] *= 1.15
    color[(lm < -1.7) & (alpha > 0)] *= 0.55
    cav = ndi.gaussian_filter(height, 8) - height
    color *= np.clip(1 - 0.015 * np.clip(cav, 0, None), 0.45, 1)[..., None]
    return color, ndi.gaussian_filter(alpha, 0.8), height


def cast_shadow(alpha, height, offset=(34, 38), blur=4, ref_h=55):
    th = ndi.gaussian_filter(alpha * np.clip(height / ref_h, 0.5, 1.3), blur)
    return np.clip(ndi.shift(th, offset, order=1), 0, 1) * (1 - alpha)
