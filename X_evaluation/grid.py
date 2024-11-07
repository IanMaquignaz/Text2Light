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

def get_skydome(img):
    count = 4
    e = EnvironmentMap(img.copy(), 'latlong')
    e = e.convertTo('skyangular', int(img.shape[1]/count))
    sample = e.data.copy()
    return sample

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


def plot_histogram_LDR(
    img_ldr,
    img_tmo,
    gt_img_ldr,
    title="",
    shape:tuple[int,int]=(2048,1024),
    dpi:float=180,
):
    assert len(img_ldr) == len(img_tmo), \
         f"All lists must have the same length, got {len(img_ldr)}, {len(img_tmo)}"

    count = float(len(img_ldr))
    img_ldr = sum(
        [ cv2.cvtColor(i.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1] for i in img_ldr ]
    )/count
    img_tmo = sum(
        [ cv2.cvtColor(i.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1] for i in img_tmo ]
    )/count
    gt_img_ldr = cv2.cvtColor(gt_img_ldr.copy().astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1]

    # Normalize
    # gt_img_ldr = cv2.normalize(gt_img_ldr, None, norm_type=cv2.NORM_MINMAX)
    # img_tmo = cv2.normalize(img_tmo, None, norm_type=cv2.NORM_MINMAX)
    # img_ldr = cv2.normalize(img_ldr, None, norm_type=cv2.NORM_MINMAX)

    shape_inches = (shape[0]/float(dpi), shape[1]/float(dpi))
    fig = plt.figure(constrained_layout=False, figsize=shape_inches, dpi=dpi)
    ax = fig.subplots()
    ax.set_title(title)

    # Histogram
    bins=1000
    range=(
        min(img_ldr.min(), img_tmo.min(), gt_img_ldr.min()),
        max(img_ldr.max(), img_tmo.max(), gt_img_ldr.max())
    )

    # Get LDR Histogram
    hist_ldr, bins_edges = np.histogram(img_ldr, bins=bins, range=range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_ldr),
            label='LDR', color='blue', alpha=1., edgecolor='blue', histtype='step')

    hist_tmo, bins_edges = np.histogram(img_tmo, bins=bins_edges, range=range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_tmo),
            label='HDR SRiTMO', color='green', alpha=1., edgecolor='green', histtype='step')

    hist_ldr_gt, bins_edges = np.histogram(gt_img_ldr, bins=bins_edges, range=range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_ldr_gt),
            label='LDR Outdoor GT', color='red', alpha=1., edgecolor='red', histtype='step')

    ax.set_yscale('symlog', base=10)
    # ax.set_xscale('symlog', base=2)
    ax.set_xlabel(f'Intensity (linear bins={bins})')
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


def plot_histogram_HDR(
    img_tmo,
    img_tmo_boosted,
    gt_img_hdr,
    title="",
    shape:tuple[int,int]=(2048,1024),
    dpi:float=180,
):
    assert len(img_tmo) == len(img_tmo_boosted), \
            f"All lists must have the same length, got {len(img_ldr)}, {len(img_tmo)}, {len(img_tmo_boosted)}"
    count = len(img_tmo)

    img_tmo = sum(
        [ cv2.cvtColor(i.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1] for i in img_tmo ]
    )/count
    img_tmo_boosted = sum(
        [ cv2.cvtColor(i.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1] for i in img_tmo_boosted ]
    )/count
    gt_img_hdr = cv2.cvtColor(gt_img_hdr.copy().astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1]

    shape_inches = (shape[0]/float(dpi), shape[1]/float(dpi))
    fig = plt.figure(constrained_layout=False, figsize=shape_inches, dpi=dpi)
    ax = fig.subplots()
    ax.set_title(title)

    # Histogram
    hist_bins=500
    hist_range=(
        min(img_tmo_boosted.min(), gt_img_hdr.min()),
        max(img_tmo_boosted.max(), gt_img_hdr.max())
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

    hist_hdr_gt, bins_edges = np.histogram(gt_img_hdr, bins=bins, range=hist_range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_hdr_gt),
            label='HDR Outdoor GT', color='red', alpha=1., edgecolor='red', histtype='step')

    hist_tmo, bins_edges = np.histogram(img_tmo, bins=bins_edges, range=hist_range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_tmo),
            label='HDR SRiTMO', color='green', alpha=1., edgecolor='green', histtype='step')

    hist_boosted, bins_edges = np.histogram(img_tmo_boosted, bins=bins_edges, range=hist_range, density=False, weights=None)
    ax.hist(bins_edges[:-1], bins_edges, weights=(hist_boosted),
            label='HDR SRiTMO Boosted', color='blue', alpha=1., edgecolor='blue', histtype='step')


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


matches = load_match_list(FILE_MATCHES)
for k, v in tqdm(matches.items()):

    img_ldr = []
    img_tmo = []
    img_tmo_hdr = []
    img_tmo_boosted = []

    trim=-76
    # Generated LDR
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+v[0]+'.png')
    img_gn_ldr = load_image(p_gn_ldr)
    img_gn_ldr[trim:,:,:]=0
    img_gn_ldr = match_exposure(img_gn_ldr,img_gn_ldr)

    # Ground Truth HDR
    p_gt = os.path.join(PATH_panos_outdoor, k+' Panorama_hdr.exr')
    img_gt_hdr = load_image(p_gt)

    # Match shape and exposure
    img_gt_hdr = match_shape(img_gn_ldr, img_gt_hdr)
    img_gn_ldr = img_gn_ldr[:trim,:,:]
    img_gt_hdr = img_gt_hdr[:trim,:,:]
    img_gt_hdr = match_exposure(img_gn_ldr.copy(), img_gt_hdr)

    img_gt_ldr = tm_gamma(img_gt_hdr.copy(), 2.2)

    images = []
    txt_thickness = 2
    txt_color = (1,1,1)
    for m in v:
        p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+m+'.png')
        img_gn_ldr = load_image(p_gn_ldr)
        img_gn_ldr = img_gn_ldr[:trim,:,:]
        img_gn_ldr = match_exposure(img_gt_ldr.copy(),img_gn_ldr)
        img_ldr.append(img_gn_ldr.copy())

        p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_SRiTMO_'+m+'.exr')
        img_gn_hdr = load_image(p_gn_hdr)
        img_gn_hdr = img_gn_hdr[:trim,:,:]
        img_tmo.append(match_exposure(img_gt_ldr.copy(), img_gn_hdr.copy()))
        img_tmo_hdr.append(match_exposure(img_gt_hdr.copy(), img_gn_hdr.copy()))

        # HDR EMD
        p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_SRiTMO_boosted_'+m+'.exr')
        img_gn_hdr = load_image(p_gn_hdr)
        img_gn_hdr = img_gn_hdr[:trim,:,:]
        img_gn_hdr = match_exposure(img_gt_hdr.copy(), img_gn_hdr.copy())
        img_tmo_boosted.append(img_gn_hdr.copy())

        img_gn_hdr = tm_gamma(img_gn_hdr, 2.2)
        img_gn_hdr = cv2.putText(img_gn_hdr, m, (10, img_gn_hdr.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, 1, txt_color, txt_thickness, cv2.FILLED)
        images.append(toTensor(img_gn_hdr))

    img_gt_ldr = cv2.putText(img_gt_ldr, f'Ground Truth: {k}', (10, img_gt_ldr.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, 1, txt_color, txt_thickness, cv2.FILLED)
    images.insert(0, toTensor(img_gt_ldr.copy()))

    if len(images) % 2 != 0:
        images.append(toTensor(np.ones_like(img_gt_ldr)))

    plot = plot_histogram_LDR(
        img_ldr=img_ldr.copy(),
        img_tmo=img_tmo.copy(),
        gt_img_ldr=tm_gamma(img_gt_hdr.copy(), 2.2),
        title="LDR Cumulative Intensity",
        shape=(img_gt_ldr.shape[1],img_gt_ldr.shape[0])
    )
    images.append(toTensor(plot))

    plot = plot_histogram_HDR(
        img_tmo=img_tmo_hdr.copy(),
        img_tmo_boosted=img_tmo_boosted.copy(),
        gt_img_hdr=img_gt_hdr.copy(),
        title="HDR Cumulative Intensity",
        shape=(img_gt_ldr.shape[1],img_gt_ldr.shape[0])
    )
    images.append(toTensor(plot))
    # for i in images:
    #     print(i.shape, i.min(), i.max())

    # Made image grid
    grid = make_grid(images, nrow=2)
    # print('grid', grid.shape, grid.min(), grid.max())

    txt_color = (0,0,0)
    file_gt = f"run_2_matches_confirmed_grid/renders/{k} Panorama_hdr.png"
    render_gt = load_render(file_gt)
    render_gt = cv2.resize(render_gt, None, fx=img_gt_ldr.shape[1]/render_gt.shape[1], fy=img_gt_ldr.shape[1]/render_gt.shape[1])
    render_gt = cv2.putText(render_gt, k, (10, render_gt.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, 1, txt_color, txt_thickness, cv2.FILLED)
    render_gt = toTensor(render_gt)

    file_gn = f"run_2_matches_confirmed_grid/renders/hdr_SRiTMO_boosted_{v[0]}.png"
    render_gn = load_render(file_gn)
    render_gn = cv2.resize(render_gn, None, fx=img_gt_ldr.shape[1]/render_gn.shape[1], fy=img_gt_ldr.shape[1]/render_gn.shape[1])
    render_gn = cv2.putText(render_gn, v[0], (10, render_gn.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, 1, txt_color, txt_thickness, cv2.FILLED)
    render_gn = toTensor(render_gn)
    render_grid = make_grid([render_gt, render_gn], nrow=2)
    # print('render_grid', render_grid.shape, render_grid.min(), render_grid.max())

    grid = torch.cat([grid, render_grid], dim=1)
    # print('grid', grid.shape, grid.min(), grid.max())

    assert grid.min() >= 0.0 and grid.max() <= 1.0, \
        f"Grid must be in range [0,1], got {grid.min()}, {grid.max()}"
    save_image(grid, os.path.join(PATH_output, k+'_matches.png'))
