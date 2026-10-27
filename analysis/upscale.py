import sys, torch, numpy as np
from PIL import Image
from spandrel import ModelLoader
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = ModelLoader().load_from_file("models/RealESRGAN_x4plus_anime_6B.pth").eval().to(dev)
src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB"); x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255)[None]
T, P, s = 256, 16, m.scale; _, _, h, w = x.shape; y = torch.zeros(1, 3, h * s, w * s)
with torch.no_grad():
    for i in range(0, h, T):
        for j in range(0, w, T):
            i0, j0, i1, j1 = max(i - P, 0), max(j - P, 0), min(i + T + P, h), min(j + T + P, w)
            t = m(x[:, :, i0:i1, j0:j1].to(dev)).cpu()
            y[:, :, i * s:min(i + T, h) * s, j * s:min(j + T, w) * s] = t[:, :, (i - i0) * s:(min(i + T, h) - i0) * s, (j - j0) * s:(min(j + T, w) - j0) * s]
up = Image.fromarray((y[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
up.resize((2160, 3840), Image.LANCZOS).save(out.replace(".png", "-4k.png"))
up.resize((1080, 1920), Image.LANCZOS).save(out)
print("ok", up.size)
