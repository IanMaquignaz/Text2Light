# Standard Library
import os
import csv
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

# Machine Vision/Learning
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
toTensor = ToTensor()
from torchvision.transforms import functional as F
from torchvision.utils import make_grid, save_image

# Custom
from tools import load_image, load_render
from tools import match_shape, match_exposure
from tools import tm_gamma, exposure
# from tools import plot_histogram
from envmap import EnvironmentMap
from utils_ml.metrics import EarthMoversDistance as EMD # EMD
from utils_cv.io.opencv import save_image as cv_save_image, load_image as cv_load_image


PATH_panos_generated = "./generated_panorama_run_2"
PATH_panos_outdoor = "./outdoorPanosExr"
PATH_output = "./run_2_matches_confirmed_grid"
# shutil.rmtree(PATH_output, ignore_errors=True)
# os.makedirs(PATH_output, exist_ok=True)

FILE_MATCHES="run_2_matches_confirmed.txt"

# Load unique filenames from unique.txt
def load_match_list(file):
    matches = {}
    with open(file, 'r') as f:
        lines = f.readlines()
    pairs = [ l.strip().split("; ") for l in lines]
    for r, a in pairs:
        if r in matches.keys():
            matches[r].append(a)
        else:
            matches[r] = [a]
    return matches


def boost_ldr2HDR(img):
    # img  = (img - img.min()) / (img.max() - img.min())
    # img = img * 2.0 - 1.0
    if img.min()*8 < -1.0 and img.max()*8 > 1.0:
        img = img * 7.0
    else:
        img  = (img - img.min()) / (img.max() - img.min())
        img = img * 2.0 - 1.0
        img = img*8
    print(img.min(), img.max())
    gamma = 0.5
    boost = 8
    balance = 0.2
    sun_thresh = 0.83
    hdr_output = toTensor(img).unsqueeze(0)
    luma = torch.mean(hdr_output, dim=1, keepdim=True)
    mask = torch.clip(luma / 8 - sun_thresh, 0, 1)
    mean = hdr_output.mean()
    hdr_output += hdr_output * mask * boost
    hdr_output = torch.exp((hdr_output - mean) * gamma - balance)
    return hdr_output[0].numpy().transpose(1,2,0)

    # sun_thresh = 0.95
    # if img.min() < 0.0:
    #     luma = cv2.cvtColor(img.astype(np.float32)-img.min(), cv2.COLOR_RGB2XYZ)[:,:,1]
    # else:
    #     luma = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1]
    # mask_sun = (luma / luma.max()) > sun_thresh
    # mask_sky = np.logical_not(mask_sun)
    # luma_mean_sun = luma[mask_sun].mean()
    # luma_mean_sky = luma[mask_sky].mean()
    # img[mask_sun] = np.power(2.2,(img[mask_sun]))
    # img[mask_sky] = np.power(2.2, (img[mask_sky]) -0.3)-0.7
    # return img

def plot_histogram_HDR(
    img_gn_ldr,
    img_gn_hdr,
    img_gt_hdr,
    title="",
    shape:tuple[int,int]=(2048,1024),
    dpi:float=180,
):
    img_gn_ldr = cv2.cvtColor(img_gn_ldr.copy().astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1]
    img_gn_hdr = cv2.cvtColor(img_gn_hdr.copy().astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1]
    img_gt_hdr = cv2.cvtColor(img_gt_hdr.copy().astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1]

    shape_inches = (shape[0]/float(dpi), shape[1]/float(dpi))
    fig = plt.figure(constrained_layout=False, figsize=shape_inches, dpi=dpi)
    ax = fig.subplots()
    ax.set_title(title)

    # Histogram
    hist_bins=500
    hist_range=(
        min(img_gn_ldr.min(), img_gn_hdr.min(), img_gt_hdr.min()),
        max(img_gn_ldr.max(), img_gn_hdr.max(), img_gt_hdr.max())
    )

    # bins = np.linspace(hist_range[0], hist_range[1], hist_bins)
    # print(bins)
    base = 2
    bins = np.logspace(-np.sqrt(np.log2(hist_range[1])), np.log2(hist_range[1]), base=base, num=hist_bins)
    bins2 = bins[bins <=base]
    bins2 = np.linspace(0, base, len(bins2))
    bins[bins <=base] = bins2
    # bins = np.concatenate((bins[bins <=4], bins2))
    # print(bins)

    hist_hdr_gt, bins_edges = np.histogram(img_gt_hdr, bins=bins, range=hist_range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_hdr_gt),
            label='HDR Outdoor GT', color='red', alpha=0.3, edgecolor='red', histtype='step')

    hist_gn_ldr, bins_edges = np.histogram(img_gn_ldr, bins=bins_edges, range=hist_range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_gn_ldr),
            label='LDR Boosted', color='green', alpha=1., edgecolor='green', histtype='step')

    # hist_tmo, bins_edges = np.histogram(img_tmo, bins=bins_edges, range=hist_range, density=False, weights=None)
    # ax.hist(bins_edges[:-1], bins_edges, weights=(hist_tmo),
    #         label='HDR SRiTMO', color='green', alpha=1., edgecolor='green', histtype='step')

    hist_gn_hdr, bins_edges = np.histogram(img_gn_hdr, bins=bins_edges, range=hist_range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_gn_hdr),
            label='HDR SRiTMO Boosted', color='blue', alpha=0.3, edgecolor='blue', histtype='step')


    ax.set_yscale('symlog', base=20)
    ax.set_xscale('symlog', base=2)
    ax.set_xlabel(f'Intensity (Symlog$_{base}$ bins={hist_bins})')
    ax.set_ylabel('Mean Count')
    ax.legend(loc='upper right', prop={'size': 8})

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # RGBA[0,255] to RGB [0,1]
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)/255.
    plot = np.zeros((shape[1], shape[0], 3), dtype=np.float32)
    plot[:image.shape[0], :image.shape[1]] = image
    plt.close('all')

    return plot

trim=-76
images = []
txt_thickness = 2
txt_color = (255,255,255)
matches = load_match_list(FILE_MATCHES)
for k, v in tqdm(matches.items()):

    # Generated LDR
    p_ldr = os.path.join(PATH_panos_generated, f'ldr/ldr_{v[0]}.png')
    img_ldr = load_image(p_ldr)

    # Ground Truth HDR
    p_gt = os.path.join(PATH_panos_outdoor, k+' Panorama_hdr.exr')
    img_gt_hdr = match_shape(img_ldr,load_image(p_gt))
    img_gt_hdr = match_exposure(img_ldr.copy(), img_gt_hdr, trim=-76)

    # Generated input to SRiTMO
    p_gn_ldr = os.path.join(PATH_panos_generated, "hdr", f"ldr_input_{v[0]}.exr")
    img_gn_ldr = load_image(p_gn_ldr)
    img_gn_ldr_boosted = boost_ldr2HDR(img_gn_ldr)
    img_gn_ldr_boosted = match_exposure(img_gt_hdr.copy(), img_gn_ldr_boosted, trim=-76)
    # cv_save_image(f'boosted_{k}.exr', img_gn_ldr_boosted.astype(np.float32))

    # Generated HDR from SRiTMO
    p_gn_HDR = p_a = os.path.join(PATH_panos_generated, "hdr", f"hdr_SRiTMO_boosted_{v[0]}.exr")
    img_gn_hdr = load_image(p_gn_HDR)
    img_gn_hdr = match_exposure(img_gt_hdr.copy(), img_gn_hdr, trim=-76)

    img_gt_hdr = img_gt_hdr[:trim,:,:]
    img_gn_ldr_boosted = img_gn_ldr_boosted[:trim,:,:]
    img_gn_hdr = img_gn_hdr[:trim,:,:]

    plot = plot_histogram_HDR(
        img_gn_ldr=img_gn_ldr_boosted.copy(),
        img_gn_hdr=img_gn_hdr.copy(),
        img_gt_hdr=img_gt_hdr.copy(),
        title="HDR Cumulative Intensity",
        shape=(img_gt_hdr.shape[1],img_gt_hdr.shape[0])
    )
    img_gt_hdr = tm_gamma(img_gt_hdr, 2.2)
    img_gt_hdr = cv2.putText(img_gt_hdr.astype(np.float32).copy(), text=f"Ground Truth: {k}", org=(10, img_gt_hdr.shape[0]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=txt_color, thickness=txt_thickness, lineType=cv2.FILLED)
    img_gn_ldr_boosted = tm_gamma(img_gn_ldr_boosted, 2.2)
    img_gn_ldr_boosted = cv2.putText(img_gn_ldr_boosted.astype(np.float32).copy(), text=f"{v[0]}", org=(10, img_gn_ldr_boosted.shape[0]-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=txt_color, thickness=txt_thickness, lineType=cv2.FILLED)

    grid = make_grid([ toTensor(img_gt_hdr), toTensor(img_gn_ldr_boosted), toTensor(plot) ], nrow=1)
    images.append(grid)

    # if len(images) == 3:
    #     break

master_grid = make_grid(images[:len(images)//2], nrow=3, padding=4, pad_value=0)
save_image(master_grid, os.path.join(PATH_output, 'matches_ldrBoosted_1.png'))
master_grid = make_grid(images[len(images)//2:], nrow=3, padding=4, pad_value=0)
save_image(master_grid, os.path.join(PATH_output, 'matches_ldrBoosted_2.png'))