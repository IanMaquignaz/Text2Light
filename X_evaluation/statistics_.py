# Standard Library
import numpy as np
import cv2

import dill
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


# Custom
from tools import match_grayscale_exposure as match_exposure
from tools import load_image, save_image, convert_envmap



# PATH_ROOT = "generated_panorama_run_2"

# regex_ldr_input = PATH_ROOT+'/hdr/ldr_input_*'
# paths_ldr_input = glob(regex_ldr_input)

# regex_SRiTMO = PATH_ROOT+'/hdr/hdr_SRiTMO_[*'
# paths_SRiTMO = glob(regex_SRiTMO)

# regex_SRiTMO_boosted = PATH_ROOT+'/hdr/hdr_SRiTMO_boosted_[*'
# paths_SRiTMO_boosted = glob(regex_SRiTMO_boosted)

# print(f"Found {len(paths_ldr_input)} LDR input images")
# print(f"Found {len(paths_SRiTMO)} SRiTMO images")
# print(f"Found {len(paths_SRiTMO_boosted)} SRiTMO boosted images")
# assert len(paths_ldr_input) == len(paths_SRiTMO) == len(paths_SRiTMO_boosted)

# Matches ONLY
# Get HDRDB pano paths
DIR_panos_outdoor="./outdoorPanosExr/"
TAIL_panos_outdoor=" Panorama_hdr.exr"

# Get generated pano paths
DIR_panos_generated="./generated_panorama_run_2/"
NAME_MATCH_FILE="run_2_matches_confirmed.txt"

# Load unique filenames from unique.txt
def load_match_list():
    with open(NAME_MATCH_FILE, 'r') as f:
        lines = f.readlines()
    matches = [ l.strip().split("; ") for l in lines]
    return matches
matches = load_match_list()

paths_ldr_input = [ DIR_panos_generated+'/hdr/ldr_input_'+a+'.exr' for _,a in matches ]
paths_SRiTMO = [ DIR_panos_generated+'/hdr/hdr_SRiTMO_'+a+'.exr' for _,a in matches ]
paths_SRiTMO_boosted = [ DIR_panos_generated+'/hdr/hdr_SRiTMO_boosted_'+a+'.exr' for _,a in matches ]
paths_GT_HDR = [ DIR_panos_outdoor+a+TAIL_panos_outdoor for a,_ in matches ]

print(f"Found {len(paths_ldr_input)} LDR input images")
print(f"Found {len(paths_SRiTMO)} SRiTMO images")
print(f"Found {len(paths_SRiTMO_boosted)} SRiTMO boosted images")
print(f"Found {len(paths_GT_HDR)} GT HDR images")
assert len(paths_ldr_input) \
    == len(paths_SRiTMO) \
        == len(paths_SRiTMO_boosted) \
            == len(paths_GT_HDR)

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
# hdr_images_gt = load_image_set(paths_GT_HDR)
# dill.dump(hdr_images_gt, open("skyLatlong_hdr_images_gt_all.pkl", "wb"))

ldr_images_all = dill.load(open("skyLatlong_ldr_images_all.pkl", "rb")).astype(np.float64)
hdr_images_all = dill.load(open("skyLatlong_hdr_images_all.pkl", "rb")).astype(np.float64)
hdr_images_boosted_all = dill.load(open("skyLatlong_hdr_images_boosted_all.pkl", "rb")).astype(np.float64)
hdr_images_gt = dill.load(open("skyLatlong_hdr_images_gt_all.pkl", "rb")).astype(np.float64)

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
print(hdr_images_gt.shape, calc_statistics(hdr_images_gt))

print('---- Reshaped ----')
hdr_images_gt = hdr_images_gt[:,:-800,:]
tmp_list = []
for i in tqdm(range(len(ldr_images_all))):
    t = cv2.resize(hdr_images_gt[i], dsize=(hdr_images_all.shape[-1], hdr_images_all.shape[-2]), interpolation=cv2.INTER_LINEAR)
    tmp_list.append(np.expand_dims(t, 0))
hdr_images_gt = np.concatenate(tmp_list, axis=0)
print(hdr_images_gt.shape, calc_statistics(hdr_images_gt))


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

# ldr_images_all = normalize_minMax(ldr_images_all)
ldr_images_all = match_exposure(ldr_images_all, ldr_images_all)
hdr_images_all = match_exposure(ldr_images_all, hdr_images_all)
hdr_images_boosted_all = match_exposure(ldr_images_all, hdr_images_boosted_all)
hdr_images_gt = match_exposure(ldr_images_all, hdr_images_gt)

print('---- Matched Exposure ----')
print(ldr_images_all.shape, calc_statistics(ldr_images_all))
print(hdr_images_all.shape, calc_statistics(hdr_images_all))
print(hdr_images_boosted_all.shape, calc_statistics(hdr_images_boosted_all))
print(hdr_images_gt.shape, calc_statistics(hdr_images_gt))

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
hdr_images_gt = tm_logN(hdr_images_gt, base)

print('---- Tonemapped ----')
print(ldr_images_all.shape, calc_statistics(ldr_images_all))
print(hdr_images_all.shape, calc_statistics(hdr_images_all))
print(hdr_images_boosted_all.shape, calc_statistics(hdr_images_boosted_all))
print(hdr_images_gt.shape, calc_statistics(hdr_images_gt))

# Histogram
bins=10000
range=(hdr_images_boosted_all.min(),hdr_images_boosted_all.max())
plot_shape=(2048,1024)

# Get Histogram
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, (ax1,ax2) = plt.subplots(2,1,sharex='all',figsize=(plot_shape[0]*px, plot_shape[1]*px))

hist_ldr, bins_edges = np.histogram(ldr_images_all, bins=bins, range=range, density=False, weights=None)
ax1.hist(bins_edges[:-1], bins_edges, weights=(hist_ldr),
        label='LDR', color='blue', alpha=1., edgecolor='blue', histtype='step')

hist_hdr, bins_edges = np.histogram(hdr_images_all, bins=bins_edges, range=range, density=False, weights=None)
ax1.hist(bins_edges[:-1], bins_edges, weights=(hist_hdr),
        label='HDR SRiTMO', color='green', alpha=1., edgecolor='green', histtype='step')

hist_boosted, bins_edges = np.histogram(hdr_images_boosted_all, bins=bins_edges, range=range, density=False, weights=None)
ax2.hist(bins_edges[:-1], bins_edges, weights=(hist_boosted),
        label='HDR SRiTMO Boosted', color='black', alpha=1., edgecolor='black', histtype='step')

hist_hdr_gt, bins_edges = np.histogram(hdr_images_gt, bins=bins_edges, range=range, density=False, weights=None)
ax2.hist(bins_edges[:-1], bins_edges, weights=(hist_hdr_gt),
        label='HDR Outdoor GT', color='red', alpha=1., edgecolor='red', histtype='step')

ax1.set_yscale('log', base=10)
# ax1.set_xscale('log', base=4)
# ax1.set_xlabel(f'Log{base}* Intensity (bins={bins})')
ax1.set_ylabel('Count')
ax1.set_title('Intensity')
ax1.legend(loc='upper right')

ax2.set_yscale('log', base=10)
# ax2.set_xscale('log', base=4)
ax2.set_xlabel(f'Log{base}* Intensity (bins={bins})')
ax2.set_ylabel('Count')
# ax2.set_title('Intensity')
ax2.legend(loc='upper right')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
fig.canvas.draw()
image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

# RGBA[0,255] to RGB [0,1]
image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)/255.
plt.close('all')
save_image("histogram.png", image)
