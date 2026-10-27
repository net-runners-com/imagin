"""Usage: measure.py IMAGE [--panels]  -> key metrics vs TARGET daytime (mean, min-max)."""
import sys, json, math, os
sys.argv, args = sys.argv[:1], sys.argv[1:]
from stylemetrics import *
SP = os.path.dirname(os.path.abspath(__file__)); T = "TARGET (19)"; DAY = [0, 2, 4, 14, 16, 17]
src = {"d": "depth_per_image.json", "e": "edge_per_image.json", "v": "value_per_image.json", "s": "shader_per_image.json", "l": "light_per_image.json", "g": "geo_per_image.json"}
tgt = [{} for _ in DAY]
for p, f in src.items():
    J = json.load(open(f"{SP}/{f}"))[T]
    for j, i in enumerate(DAY): tgt[j].update({f"{p}_{k}": v for k, v in J[i].items()})
for t in tgt: t["rise"] = t["d_far_L"] - t["d_near_L"]; t["lift"] = t["d_far_p5"] - t["d_near_p5"]
KEYS = [("haze","l_haze_dc"),("fine detail","v_E_fine"),("quiet%","v_quiet%"),("line%","s_line%"),("line width","s_line_width"),("line chroma","s_line_C"),
        ("glints/Mpx","s_glints/Mpx"),("L std","v_L_std"),("dark%","v_dark%"),("C light","v_C_light"),("hue entropy","v_hue_entropy"),("rim","e_rim"),
        ("sky top dark","l_top_vs_mid"),("vignette","l_vignette"),("bloom","l_bloom"),("sh chroma","l_sh_chroma"),("hard edge%","s_hard_share%"),
        ("cel-ness","g_cel_ness"),("light elev","g_light_el"),("normal spread","g_normal_spread_deg"),("bump","g_bump_x1000"),("rise","rise"),("far lift","lift")]
path = args[0]; im = Image.open(path).convert("RGB"); ims = panels(path) if "--panels" in args else [im]
rs = [measure(x) for x in ims]; hits = 0
print(f"{'metric':14}{'target mean':>12}{'target range':>18}{'this':>9}  ok")
for name, k in KEYS:
    tv = [t[k] for t in tgt if not math.isnan(t[k])]; mu = sum(tv) / len(tv); lo, hi = min(tv), max(tv)
    x = sum(r[k] for r in rs) / len(rs); ok = lo <= x <= hi; hits += ok
    print(f"{name:14}{mu:12.2f}{lo:9.2f}–{hi:<8.2f}{x:9.2f}  {'✓' if ok else ('↑' if x > hi else '↓')}")
print(f"in target range: {hits}/{len(KEYS)}")
