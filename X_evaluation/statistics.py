# Standard Library
import numpy as np
import cv2

import dill
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


# Custom
from tools import load_image, save_image, convert_envmap
PATH_ROOT = "generated_panorama_run_2"

regex_ldr_input = PATH_ROOT+'/hdr/ldr_input_*'
paths_ldr_input = glob(regex_ldr_input)

regex_SRiTMO = PATH_ROOT+'/hdr/hdr_SRiTMO_[*'
paths_SRiTMO = glob(regex_SRiTMO)

regex_SRiTMO_boosted = PATH_ROOT+'/hdr/hdr_SRiTMO_boosted_[*'
paths_SRiTMO_boosted = glob(regex_SRiTMO_boosted)

print(f"Found {len(paths_ldr_input)} LDR input images")
print(f"Found {len(paths_SRiTMO)} SRiTMO images")
print(f"Found {len(paths_SRiTMO_boosted)} SRiTMO boosted images")
assert len(paths_ldr_input) == len(paths_SRiTMO) == len(paths_SRiTMO_boosted)
paths_zipped = zip(paths_ldr_input, paths_SRiTMO, paths_SRiTMO_boosted)


# def load_image_set(paths):
#     img_set = []
#     for p in tqdm(paths):
#         img = load_image(p)
#         img = convert_envmap(img, format_dst='skylatlong', size_dst=img.shape[0]//2)
#         img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1]
#         img_set.append(np.expand_dims(img, 0))
#     return np.concatenate(img_set, axis=0)

# ldr_images_all = load_image_set(paths_ldr_input)
# dill.dump(ldr_images_all, open("skyLatlong_ldr_images_all.pkl", "wb"))
# hdr_images_all = load_image_set(paths_SRiTMO)
# dill.dump(hdr_images_all, open("skyLatlong_hdr_images_all.pkl", "wb"))
# hdr_images_boosted_all = load_image_set(paths_SRiTMO_boosted)
# dill.dump(hdr_images_boosted_all, open("skyLatlong_hdr_images_boosted_all.pkl", "wb"))

ldr_images_all = dill.load(open("skyLatlong_ldr_images_all.pkl", "rb")).astype(np.float64)
hdr_images_all = dill.load(open("skyLatlong_hdr_images_all.pkl", "rb")).astype(np.float64)
hdr_images_boosted_all = dill.load(open("skyLatlong_hdr_images_boosted_all.pkl", "rb")).astype(np.float64)

def calc_statistics(img_set):
    f_shape = img_set.shape[1] * img_set.shape[2]
    img_set_flat = img_set.reshape(-1, f_shape)
    mean = np.nanmean(img_set_flat, axis=1).mean()
    std = np.nanstd(img_set_flat, axis=1).mean()
    var = np.nanvar(img_set_flat, axis=1).mean()
    return f"\t min={img_set_flat.min()}, max={img_set_flat.max()}, mean={mean}, std={std}, var={var}"

print('---- Original ----')
print(ldr_images_all.shape, calc_statistics(ldr_images_all))
print(hdr_images_all.shape, calc_statistics(hdr_images_all))
print(hdr_images_boosted_all.shape, calc_statistics(hdr_images_boosted_all))

def normalize_meanStd(img):
    # Mean 0, std 1
    mean = img.mean()
    std = img.std()
    return (img - mean) / std

def normalize_minMax(img):
    # Min 0, Max 1
    return (img - img.min()) / (img.max() - img.min())

def normalize_minMax(img):
    # Min 0, Max 1
    mean = img.mean()
    std = img.std()
    mean = img[img < (mean+std)].mean()
    return (img - img.min()) / (img.max() - img.min())

def match_exposure(ref, a):
    f_shape = ref.shape[1] * ref.shape[2]
    ref = ref - ref.min()
    a = a - a.min()
    ratio = np.reshape(ref / a, (-1, f_shape))
    ratio = np.nan_to_num(ratio, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))

    # # Filter
    # u = np.nanmean(ratio)
    # std = np.nanstd(ratio)
    # ratio[ratio > (u + 2*std)] = np.nan
    # ratio[ratio < (u - 2*std)] = np.nan

    # Get final Ratio
    ratio = np.nanmean(ratio, axis=1)
    ratio = np.repeat(ratio, f_shape).reshape(a.shape)

    # Apply Ratio
    a = (a * ratio)
    return np.nan_to_num(a, nan=0, posinf=0, neginf=0)

# ldr_images_all = normalize_minMax(ldr_images_all)
ldr_images_all = ldr_images_all - ldr_images_all.min()
hdr_images_all = match_exposure(ldr_images_all, hdr_images_all)
hdr_images_boosted_all = match_exposure(ldr_images_all, hdr_images_boosted_all)

print('---- Matched Exposure ----')
print(ldr_images_all.shape, calc_statistics(ldr_images_all))
print(hdr_images_all.shape, calc_statistics(hdr_images_all))
print(hdr_images_boosted_all.shape, calc_statistics(hdr_images_boosted_all))

def tm_logN(img, base=2, selective=False):
    # log_b (a) = log_k (a) / log_k (b)
    if selective:
        t = img
        t[t > base] = np.log2(t[t > base]+base**base) / np.log2(base)
    else:
        t = np.log2(img+1) / np.log2(base)
    t = np.nan_to_num(t, nan=0, posinf=0, neginf=0)
    return t

# Tonemap
base=4
selective=True
ldr_images_all = tm_logN(ldr_images_all, base, selective)
hdr_images_all = tm_logN(hdr_images_all, base, selective)
hdr_images_boosted_all = tm_logN(hdr_images_boosted_all, base)

print('---- Tonemapped ----')
print(ldr_images_all.shape, calc_statistics(ldr_images_all))
print(hdr_images_all.shape, calc_statistics(hdr_images_all))
print(hdr_images_boosted_all.shape, calc_statistics(hdr_images_boosted_all))

# Histogram
bins=10000
range=(hdr_images_boosted_all.min(),hdr_images_boosted_all.max())
plot_shape=(2048,1024)

# Get Histogram
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, ax = plt.subplots(figsize=(plot_shape[0]*px, plot_shape[1]*px))

hist_ldr, bins_edges = np.histogram(ldr_images_all, bins=bins, range=range, density=False, weights=None)
plt.hist(bins_edges[:-1], bins_edges, weights=(hist_ldr),
        label='LDR', color='blue', alpha=1., edgecolor='blue', histtype='step')

hist_hdr, bins_edges = np.histogram(hdr_images_all, bins=bins_edges, range=range, density=False, weights=None)
plt.hist(bins_edges[:-1], bins_edges, weights=(hist_hdr),
        label='HDR SRiTMO', color='green', alpha=1., edgecolor='green', histtype='step')

hist_boosted, bins_edges = np.histogram(hdr_images_boosted_all, bins=bins_edges, range=range, density=False, weights=None)
plt.hist(bins_edges[:-1], bins_edges, weights=(hist_boosted),
        label='HDR SRiTMO Boosted', color='red', alpha=1., edgecolor='red', histtype='step')

ax.set_yscale('log', base=10)
# ax.set_xscale('log', base=4)

plt.legend(loc='upper right')
ax.set_xlabel(f'Log{base}* Intensity (bins={bins})')
ax.set_ylabel('Count')
ax.set_title('Intensity')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
fig.canvas.draw()
image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

# RGBA[0,255] to RGB [0,1]
image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)/255.
plt.close('all')
save_image("histogram.png", image)
