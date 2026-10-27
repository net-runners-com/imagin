"""Render a small synthetic golf hole with shaders.py -> examples/shader_demo*.png

usage: python shaders/demo.py
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shaders as sh  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples')
os.makedirs(OUT, exist_ok=True)

# low-res label map: 0 rough, 1 fairway, 2 green, 3 bunker, 4 trees
h, w = 120, 80
yy, xx = np.mgrid[:h, :w]
lab = np.zeros((h, w), np.int32)
lab[((xx - 40) / 16) ** 2 + ((yy - 70) / 40) ** 2 < 1] = 1
lab[((xx - 42) / 12) ** 2 + ((yy - 18) / 10) ** 2 < 1] = 2
lab[((xx - 60) / 6) ** 2 + ((yy - 30) / 4) ** 2 < 1] = 3
lab[((xx - 10) / 12) ** 2 + ((yy - 60) / 30) ** 2 < 1] = 4

S = 8
L = sh.soft_upscale(lab, S, sigma=7, boost={3: 1.25})
L = sh.domain_warp(L, amp=5, scale=80, keep=(2, 3))
H, W = L.shape
M = {k: L == v for k, v in dict(rough=0, fairway=1, green=2, bunker=3, tree=4).items()}

# height: raised green, sunken bunker with lip, gentle undulation
dist = lambda m: ndi.distance_transform_edt(m).astype(np.float32)
bd, bout = dist(M['bunker']), dist(~M['bunker'])
hgt = 0.7 * sh.fbm((H, W), 600, 3, 1) + 3 * np.clip(dist(M['green']) / 40, 0, 1) ** 0.7
hgt -= 5 * np.clip(bd / 25, 0, 1) ** 0.8
hgt += 1.6 * np.exp(-((bout - 5) / 8) ** 2) * (bout > 0)
hgt = ndi.gaussian_filter(hgt, 2)
bl = sh.blades((H, W), 3.0, 4, 12)
shade = sh.hillshade(hgt * 18 + 0.8 * bl * (M['rough'] | M['tree']), strength=0.1)

big, mid, hue = sh.fbm((H, W), 500, 3, 10), sh.fbm((H, W), 90, 3, 11), 0.3 * sh.fbm((H, W), 250, 3, 15)
img = sh.grass((H, W), (64, 96, 38), (90, 106, 44), (50, 88, 44), hue, big, mid, bl, (0.06, 0.06, 0.09))
fair = sh.grass((H, W), (106, 156, 62), (124, 158, 64), (88, 148, 70), hue, big, mid, sh.blades((H, W), 1.2, 3, 14), (0.035, 0.03, 0.04))
fair *= (1 + 0.085 * sh.stripes((H, W), (0, -1), 30))[..., None]
img[M['fairway']] = fair[M['fairway']]
green = sh.grass((H, W), (100, 164, 68), (116, 164, 68), (84, 156, 76), hue, big, mid, bl, (0.02, 0.02, 0.025))
img[M['green']] = green[M['green']]
sd = sh.sand((H, W))
img[M['bunker']] = sd[M['bunker']]
img *= np.where(M['bunker'][..., None], 0.72 + 0.38 * shade[..., None], 0.55 + 0.5 * shade[..., None])

col, a, th = sh.tree_canopy(M['tree'])
s = sh.cast_shadow(a, th)
img *= (1 - 0.55 * s)[..., None]
img = img * (1 - a[..., None]) + col * a[..., None]

Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, 'shader_demo.png'))
Image.fromarray(sh.normal_map_rgb(sh.height_to_normal(hgt, 4))).save(os.path.join(OUT, 'shader_demo_normal.png'))
Image.fromarray(((hgt - hgt.min()) / (np.ptp(hgt) + 1e-6) * 255).astype(np.uint8)).save(os.path.join(OUT, 'shader_demo_height.png'))
print('ok', img.shape)
