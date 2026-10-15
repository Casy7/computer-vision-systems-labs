"""Лаб 1, частина C: геометрія та радіометрія (кроки 12-14)."""
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import *

log = []


def say(s):
    print(s)
    log.append(s)


# ============ Крок 12: аліасинг ============
cv2.setNumThreads(1)
amap = imread(os.path.join(LAB_DATA, "C", "texture.png"))   # дрібна текстура штукатурки/паперу
F = 4
sz = (amap.shape[1] // F, amap.shape[0] // F)
nearest = cv2.resize(amap, sz, interpolation=cv2.INTER_NEAREST)
SIGMA = 0.5 * F                      # типове правило: sigma ~ F/2 для зменшення в F разів
blur_then = cv2.resize(cv2.GaussianBlur(amap, (0, 0), SIGMA), sz, interpolation=cv2.INTER_NEAREST)
ref_area = cv2.resize(amap, sz, interpolation=cv2.INTER_AREA)   # еталон "без аліасингу"
p_nn, p_bl = psnr(nearest, ref_area), psnr(blur_then, ref_area)
say(f"STEP12 texture {amap.shape[:2]} -> {sz[::-1]}: PSNR(nearest vs INTER_AREA)={p_nn:.2f} dB; "
    f"PSNR(blur+nearest vs INTER_AREA)={p_bl:.2f} dB")

sd = lambda im: float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std())
say(f"STEP12 texture grey std: INTER_AREA={sd(ref_area):.2f}, nearest={sd(nearest):.2f}, blur+nearest={sd(blur_then):.2f}")

# синтетична регулярна текстура: вертикальні смуги з періодом 6 px (1/6 цикл/піксель)
H0, W0, T = 400, 1000, 6
x = np.arange(W0)
stripes = np.tile((0.5 + 0.5 * np.sign(np.sin(2 * np.pi * x / T + 0.1)))[None, :] * 255, (H0, 1)).astype(np.uint8)
s_nn = cv2.resize(stripes, (W0 // F, H0 // F), interpolation=cv2.INTER_NEAREST)
s_bl = cv2.resize(cv2.GaussianBlur(stripes, (0, 0), SIGMA), (W0 // F, H0 // F), interpolation=cv2.INTER_NEAREST)


def spectrum(row):
    r = row.astype(np.float64) - row.mean()
    s = np.abs(np.fft.rfft(r * np.hanning(len(r)))) / len(r)
    return np.fft.rfftfreq(len(r), 1.0), s


f_o, S_o = spectrum(stripes[0])
f_n, S_n = spectrum(s_nn[0])
f_b, S_b = spectrum(s_bl[0])
# нова "хибна" частота: |0.2*4 - 1| = 0.2 цикл на піксель зменшеного зображення... знайдемо пік
pk_n, pk_b = f_n[np.argmax(S_n)], f_b[np.argmax(S_b)]
say(f"STEP12 stripes: original period {T} px (f={1/T:.3f} c/px). After x{F}: nearest peak f={pk_n:.3f} c/px "
    f"(period {1/pk_n:.1f} px), amplitude {S_n.max():.2f}; blur+nearest peak amplitude {S_b.max():.3f} "
    f"(ratio {S_n.max()/S_b.max():.1f}x)")
say(f"STEP12 true content at x{F}: frequency {F/T:.2f} c/px exceeds Nyquist 0.5 -> alias at {abs(F/T-1):.2f} c/px")

fig, ax = plt.subplots(2, 3, figsize=(10, 5.4))
# однаковий масштаб: показуємо всі три у піксельній сітці зменшеного зображення
for a, im, t in zip(ax[0], (ref_area, nearest, blur_then),
                    ("еталон (INTER_AREA)", f"INTER_NEAREST (PSNR {p_nn:.1f} дБ)", f"GaussianBlur σ={SIGMA:g} + NEAREST ({p_bl:.1f} дБ)")):
    a.imshow(rgb(im), interpolation="nearest")
    a.set_title(t, fontsize=8)
    a.set_xlabel("x, піксель")
ax[0, 0].set_ylabel("y, піксель")
# збільшений фрагмент текстури
cr = (slice(200, 300), slice(400, 560))
for a, im in zip(ax[1], (ref_area, nearest, blur_then)):
    a.imshow(rgb(im[cr]), interpolation="nearest")
    a.set_xlabel("x, піксель (фрагмент)")
ax[1, 0].set_ylabel("y, піксель (фрагмент)")
fig.savefig(os.path.join(FIGURES, "c_aliasing_map.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.semilogy(f_o, S_o + 1e-6, label=f"оригінал (період {T} px)", alpha=.6)
ax.semilogy(f_n * 1.0, S_n + 1e-6, label="NEAREST ×1/4 (частоти у цикл/піксель нового зобр.)")
ax.semilogy(f_b, S_b + 1e-6, label="Gauss + NEAREST ×1/4")
ax.set_xlabel("просторова частота, цикл/піксель відповідного зображення")
ax.set_ylabel("амплітуда спектра, відн. од.")
ax.set_ylim(1e-4, 1)
ax.legend(fontsize=7)
ax.grid(alpha=.3, which="both")
fig.savefig(os.path.join(FIGURES, "c_aliasing_spectrum.png"))
plt.close(fig)

# ============ Крок 13: інтерполяція ============
big = imread(os.path.join(LAB_DATA, "A", "photo.jpg"))
say(f"STEP13 A/photo.jpg shape {big.shape}")
K = 8
methods = {"NEAREST": cv2.INTER_NEAREST, "LINEAR": cv2.INTER_LINEAR,
           "CUBIC": cv2.INTER_CUBIC, "LANCZOS4": cv2.INTER_LANCZOS4}
# 3 фрагменти 512x512 з різних частин великого знімка (нормований набір, не один кадр)
spots = [(50, 250), (800, 2200), (1500, 200)]   # (y0, x0): дроти на тлі неба, суцвіття, гілки
rows, store = [], {}
for i, (y0, x0) in enumerate(spots):
    hr = big[y0:y0 + 512, x0:x0 + 512].copy()
    lr = cv2.resize(hr, (512 // K, 512 // K), interpolation=cv2.INTER_AREA)
    for nm, m in methods.items():
        up = cv2.resize(lr, (512, 512), interpolation=m)
        rows.append((i, nm, round(float(psnr(hr, up)), 2)))
        store[(i, nm)] = up
    store[(i, "HR")] = hr
    store[(i, "LR")] = lr
mean_rows = []
for nm in methods:
    v = [r[2] for r in rows if r[1] == nm]
    mean_rows.append((nm, round(float(np.mean(v)), 2), round(float(np.std(v)), 2)))
    say(f"STEP13 {nm}: PSNR mean={np.mean(v):.2f} dB, sd={np.std(v):.2f} dB, per-crop {v}")
save_csv(os.path.join(RESULTS, "c_interpolation.csv"), ["crop", "method", "psnr_db"], rows)
save_csv(os.path.join(RESULTS, "c_interpolation_mean.csv"), ["method", "psnr_mean_db", "psnr_sd_db"], mean_rows)

fig, ax = plt.subplots(1, 5, figsize=(11, 2.6))
sub = (slice(150, 278), slice(150, 278))      # 128x128 зі 512x512 (піксельна сітка HR)
ax[0].imshow(rgb(store[(0, "HR")][sub]), interpolation="nearest")
ax[0].set_title("оригінал HR", fontsize=8)
for a, nm in zip(ax[1:], methods):
    a.imshow(rgb(store[(0, nm)][sub]), interpolation="nearest")
    a.set_title(f"{nm}\n{[r[2] for r in rows if r[0]==0 and r[1]==nm][0]:.2f} дБ", fontsize=8)
for a in ax:
    a.set_xlabel("x, піксель HR")
ax[0].set_ylabel("y, піксель HR")
fig.savefig(os.path.join(FIGURES, "c_interpolation.png"))
plt.close(fig)

# ============ Крок 14: гамма ============
photo = imread(os.path.join(LAB_DATA, "A", "photo.jpg"))
gammas = (0.5, 1.0, 2.2)


def apply_gamma(img, g):
    lut = (255.0 * (np.arange(256) / 255.0) ** g).round().astype(np.uint8)
    return cv2.LUT(img, lut)


fig, ax = plt.subplots(1, 4, figsize=(12, 2.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
for a, g in zip(ax[:3], gammas):
    a.imshow(rgb(apply_gamma(photo, g)))
    a.set_title(f"γ = {g}", fontsize=9)
    a.set_xlabel("x, піксель")
ax[0].set_ylabel("y, піксель")
t = np.linspace(0, 1, 256)
for g in gammas:
    ax[3].plot(t, t ** g, label=f"γ = {g}")
ax[3].plot(t, t, "k:", lw=.7)
ax[3].set_xlabel("вхід, норм. значення")
ax[3].set_ylabel("вихід, норм. значення")
ax[3].legend(fontsize=8)
ax[3].grid(alpha=.3)
ax[3].set_aspect("equal")
fig.savefig(os.path.join(FIGURES, "c_gamma.png"))
plt.close(fig)

G = 2.2
to_lin = lambda v: (np.asarray(v, np.float64) / 255.0) ** G
to_code = lambda L: 255.0 * np.asarray(L, np.float64) ** (1 / G)
a_, b_ = 0.0, 255.0
naive = (a_ + b_) / 2
correct = float(to_code((to_lin(a_) + to_lin(b_)) / 2))
say(f"STEP14 avg of codes 0 and 255: naive {naive:.1f} (luminance {float(to_lin(naive)):.3f}); "
    f"correct {correct:.1f} (luminance 0.500)")

# графік: усереднення (0, v)
v = np.arange(256)
nv = (0 + v) / 2
cv_ = to_code((to_lin(0) + to_lin(v)) / 2)
fig, ax = plt.subplots(figsize=(5, 3.3))
ax.plot(v, nv, label="наївне (середнє кодів)")
ax.plot(v, cv_, label="коректне (через лінійну освітленість)")
ax.set_xlabel("код другого пікселя (перший = 0)")
ax.set_ylabel("код результату усереднення")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.savefig(os.path.join(FIGURES, "c_gamma_average.png"))
plt.close(fig)

# зменшення в 4 рази: усереднення кодів vs усереднення в лінійному просторі
ph = cv2.resize(photo, (photo.shape[1] // F, photo.shape[0] // F), interpolation=cv2.INTER_AREA)
lin = (photo.astype(np.float64) / 255.0) ** G
lin_small = cv2.resize(lin, (photo.shape[1] // F, photo.shape[0] // F), interpolation=cv2.INTER_AREA)
ph_lin = np.clip(to_code(lin_small), 0, 255)
d = np.abs(ph.astype(np.float64) - ph_lin)
say(f"STEP14 downscale x{F} photo: mean|naive-linear|={d.mean():.2f}, max={d.max():.1f}; "
    f"mean luminance naive={to_lin(ph).mean():.4f}, linear-space={to_lin(ph_lin).mean():.4f}, "
    f"original={lin.mean():.4f}")
# шахова дошка 1x1: зменшення вдвічі
cb = ((np.indices((256, 256)).sum(0) % 2) * 255).astype(np.uint8)
cb_naive = cv2.resize(cb, (128, 128), interpolation=cv2.INTER_AREA)
cb_lin = float(to_code(0.5))   # еталон: середня освітленість L = 0.5
say(f"STEP14 checkerboard 1px x0.5: naive code={cb_naive.mean():.1f} (L={float(to_lin(cb_naive.mean())):.3f}) "
    f"vs correct {cb_lin:.1f} (L=0.500)")

fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
ax[0].imshow(rgb(ph), interpolation="nearest")
ax[0].set_title("усереднення кодів (INTER_AREA)", fontsize=8)
ax[1].imshow(rgb(ph_lin.round().astype(np.uint8)), interpolation="nearest")
ax[1].set_title("усереднення в лінійному просторі", fontsize=8)
for a in ax:
    a.set_xlabel("x, піксель")
ax[0].set_ylabel("y, піксель")
fig.savefig(os.path.join(FIGURES, "c_gamma_downscale.png"))
plt.close(fig)

with open(os.path.join(RESULTS, "c_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log) + "\n")
