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
from tools import match_shape
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

FILE_MATCHES="run_2_matches_confirmed.txt"
matches = load_match_list(FILE_MATCHES)

def get_skydome(img):
    count = 4
    e = EnvironmentMap(img.copy(), 'latlong')
    e = e.convertTo('skyangular', int(img.shape[1]/count))
    return e.data.copy()


def match_exposure_(ref, a, trim):
    assert ref.shape == a.shape, f"Shapes do not match: {ref.shape} != {a.shape}"
    assert ref.ndim == 3 or ref.ndim==2, f"Only 2D and 3D tensors are supported"

    if a.min() < 0:
        a = a - a.min()
    if ref.min() < 0:
        ref = ref - ref.min()

    if ref.ndim == 3:
        assert trim < 0
        a_mean = cv2.cvtColor(a[:trim,:,:].astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1].mean()
        ref_mean = cv2.cvtColor(ref[:trim,:,:].astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1].mean()

        # Get final Ratio
        ratio = np.nan_to_num(
            ref_mean / a_mean,
            nan=float('nan'),
            posinf=float('nan'),
            neginf=float('nan')
        )
        ratio = np.nan_to_num(ratio, nan=1., posinf=1., neginf=1.)
    else:
        raise "Not implemented"

    # Apply Ratio
    a = (a * ratio)
    return np.nan_to_num(a, nan=0., posinf=0., neginf=0.)


master_grid = []
for k, v in tqdm(matches.items()):
    images = []
    trim=-76
    # Generated LDR
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+v[0]+'.png')
    img_gn_ldr = load_image(p_gn_ldr)
    img_gn_ldr = match_exposure_(img_gn_ldr,img_gn_ldr, trim=trim)

    # Ground Truth HDR
    p_gt = os.path.join(PATH_panos_outdoor, k+' Panorama_hdr.exr')
    img_gt_hdr = load_image(p_gt)

    # Match shape and exposure
    img_gt_hdr = match_shape(img_gn_ldr, img_gt_hdr)
    img_gt_hdr = match_exposure_(img_gn_ldr.copy(), img_gt_hdr, trim=trim)
    img_gt_ldr = tm_gamma(img_gt_hdr.copy(), 2.2)

    txt_size=0.5
    txt_thickness = 1
    txt_color = (255,255,255)
    max_duplicates=1
    for m in v:
        p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_SRiTMO_boosted_'+m+'.exr')
        img_gn_hdr = load_image(p_gn_hdr)
        img_gn_hdr = match_exposure_(img_gt_hdr.copy(), img_gn_hdr, trim=trim)
        img_gn_hdr = get_skydome(img_gn_hdr)
        img_gn_ldr = tm_gamma(img_gn_hdr, 2.2)
        img_gn_ldr = cv2.putText(img_gn_ldr, m, (10, img_gn_hdr.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, txt_size, txt_color, txt_thickness, cv2.FILLED)

        images.append(toTensor(img_gn_ldr))
        if len(images) == max_duplicates:
            break

    img_gt_hdr = get_skydome(img_gt_hdr)
    img_gt_ldr = tm_gamma(img_gt_hdr, 2.2)
    img_gt_ldr = cv2.putText(img_gt_ldr, k, (10, img_gt_ldr.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, txt_size, txt_color, txt_thickness, cv2.FILLED)
    images.insert(0, toTensor(img_gt_ldr))

    # Pad (if needed)
    while len(images) < max_duplicates+1:
        images.append(toTensor(np.zeros_like(img_gt_ldr)))

    grid = make_grid(images, nrow=max_duplicates+1)
    assert grid.min() >= 0.0 and grid.max() <= 255.0, \
        f"Grid must be in range [0,1], got {grid.min()}, {grid.max()}"
    master_grid.append(grid.detach().clone())

    # if len(master_grid) >=2:
    #     break

final_grid = make_grid(master_grid, nrow=3, padding=2, pad_value=1)
save_image(final_grid, os.path.join(PATH_output, 'matches_skyangular.png'))
