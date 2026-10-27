"""Full per-image + per-cluster analysis of the 19 target images (all metric families)."""
import json, math, glob
from stylemetrics import *
from pbr import pbr
from comp import comp
T = "TARGET (19)"
NAMES = ["train-day","pink-dusk-street","mirror-day","red-sunset-street","town-day","hoodie-night","fuji-sunset","rooftop-sunset","whitehair-night",
         "snow-dusk-road","girl-dusk","boy-dusk-bokeh","boy-bridge-night","headphones-night","station-day","sunset-hill","cherry-path","fuji-day","bob-girl-dusk"]
CL = {"DAY landscape": [0, 2, 4, 14, 17], "DUSK landscape": [1, 3, 6, 9, 15, 16], "CHARACTER night/dusk": [5, 7, 8, 10, 11, 12, 13, 18]}
src = {"d": "depth_per_image.json", "e": "edge_per_image.json", "v": "value_per_image.json", "s": "shader_per_image.json", "l": "light_per_image.json", "g": "geo_per_image.json"}
rows = [{} for _ in range(19)]
for p, f in src.items():
    for i, r in enumerate(json.load(open(f))[T]): rows[i].update({f"{p}_{k}": v for k, v in r.items()})
ims = [Image.open(p).convert("RGB") for p in sorted(glob.glob("target/t*.jpg"))]
for i, im in enumerate(ims):
    small = im.resize((int(im.width * H / im.height), H))
    rows[i].update({f"p_{k}": v for k, v in pbr(small).items()}); rows[i].update({f"c_{k}": v for k, v in comp(small).items()})
    rows[i]["rise"] = rows[i]["d_far_L"] - rows[i]["d_near_L"]; rows[i]["lift"] = rows[i]["d_far_p5"] - rows[i]["d_near_p5"]
json.dump({"names": NAMES, "clusters": CL, "rows": rows}, open("target_full.json", "w"))
KEYS = ["v_L_mean","v_L_std","v_dark%","l_black_pt","l_white_pt","l_DR_stops","v_C_mean","v_C_light","v_C_shadow","l_sh_hue","l_hl_hue","l_sh_chroma","l_hl_chroma","v_hue_entropy",
        "l_haze_dc","rise","lift","v_E_fine","v_E_mid","v_quiet%","s_line%","s_line_width","s_line_C","s_glints/Mpx","s_hard_share%","g_cel_ness","g_light_el","g_normal_spread_deg",
        "e_rim","e_halo","l_bloom","l_vignette","l_top_vs_mid","s_dof%","s_grain","s_stroke_dir","v_light_top%","v_light_blobs","e_edge_frac%","e_jump","e_dL_sky",
        "p_shading_share","p_lum_skew","p_spec_permil","p_emissive_permil","p_translucent%","p_sky_flow","c_n_lines","c_vp_convergence","c_vp_y","l_light_srcs"]
def stat(idx, k):
    v = [rows[i][k] for i in idx if not (isinstance(rows[i][k], float) and math.isnan(rows[i][k]))]
    return (sum(v) / len(v), min(v), max(v)) if v else (float("nan"),) * 3
print("metric".ljust(20) + "".join(c[:14].rjust(22) for c in CL))
for k in KEYS: print(k.ljust(20) + "".join(f"{stat(ix,k)[0]:8.2f} ({stat(ix,k)[1]:6.1f}–{stat(ix,k)[2]:<6.1f})" for ix in CL.values()))
