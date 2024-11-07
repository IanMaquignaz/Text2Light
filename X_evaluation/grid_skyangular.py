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
    e = EnvironmentMap(img.copy(), 'latlong')
    e = e.convertTo('skyangular', img.shape[1])
    return e.data.copy()


txt_size=2
txt_thickness = 2
txt_max_lines = 3
def get_text_width(txt):
    (txt_width, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_TRIPLEX, txt_size, txt_thickness)
    return txt_width
(txt_width, txt_height), baseline = cv2.getTextSize('AAA', cv2.FONT_HERSHEY_TRIPLEX, txt_size, txt_thickness)
txt_color = (255,255,255)
master_grid = []
for k, v in tqdm(matches.items()):
    images = []
    trim=-76
    # Generated LDR
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+v[0]+'.png')
    img_gn_ldr = load_image(p_gn_ldr)
    img_gn_ldr = match_exposure(img_gn_ldr,img_gn_ldr, trim=trim)

    # Ground Truth HDR
    p_gt = os.path.join(PATH_panos_outdoor, k+' Panorama_hdr.exr')
    img_gt_hdr = load_image(p_gt)

    # Match shape and exposure
    img_gt_hdr = match_shape(img_gn_ldr, img_gt_hdr)
    img_gt_hdr = match_exposure(img_gn_ldr.copy(), img_gt_hdr, trim=trim)
    img_gt_ldr = tm_gamma(img_gt_hdr.copy(), 2.2)

    max_duplicates=1
    for m in v:
        p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_SRiTMO_boosted_'+m+'.exr')
        img_gn_hdr = load_image(p_gn_hdr)
        img_gn_hdr = match_exposure(img_gt_hdr.copy(), img_gn_hdr, trim=trim)
        img_gn_hdr = get_skydome(img_gn_hdr)
        img_gn_ldr = tm_gamma(img_gn_hdr, 2.2)
        img_gn_ldr = cv2.copyMakeBorder(img_gn_ldr, baseline,(txt_height+baseline)*(txt_max_lines),0,0, cv2.BORDER_CONSTANT, 0)

        m = m.replace('[', '').replace(']', '')
        if get_text_width(m) > img_gn_ldr.shape[1]:
            m_split = m.split(' ')
            lines = []
            while len(m_split) > 0:
                m_line = ''
                while len(m_split) > 0 and get_text_width(m_line+m_split[0]) < img_gn_ldr.shape[1]:
                    m_line += m_split.pop(0) + ' '
                lines.append(m_line)
            for i, m_line in enumerate(lines):
                img_gn_ldr = cv2.putText(
                    img_gn_ldr, m_line,
                    (10, img_gn_ldr.shape[0]-10 -((txt_height+baseline)*(len(lines)-1-i))),
                    cv2.FONT_HERSHEY_TRIPLEX, txt_size, txt_color, txt_thickness, cv2.FILLED
                )
        else:
            img_gn_ldr = cv2.putText(
                img_gn_ldr, m,
                (10, img_gn_ldr.shape[0]-10),
                cv2.FONT_HERSHEY_TRIPLEX, txt_size, txt_color, txt_thickness, cv2.FILLED
            )

        images.append(toTensor(img_gn_ldr))
        if len(images) == max_duplicates:
            break

    img_gt_hdr = get_skydome(img_gt_hdr)
    img_gt_ldr = tm_gamma(img_gt_hdr, 2.2)
    img_gt_ldr = cv2.copyMakeBorder(img_gt_ldr, baseline,(txt_height+baseline)*(txt_max_lines),0,0, cv2.BORDER_CONSTANT, 0)
    img_gt_ldr = cv2.putText(img_gt_ldr, f"Ground Truth: {k}", (10, img_gt_ldr.shape[0]-10), cv2.FONT_HERSHEY_TRIPLEX, txt_size, txt_color, txt_thickness, cv2.FILLED)
    images.insert(0, toTensor(img_gt_ldr))

    # Pad (if needed)
    while len(images) < max_duplicates+1:
        images.append(toTensor(np.zeros_like(img_gt_ldr)))

    grid = make_grid(images, nrow=max_duplicates+1)
    assert grid.min() >= 0.0 and grid.max() <= 255.0, \
        f"Grid must be in range [0,1], got {grid.min()}, {grid.max()}"
    master_grid.append(grid.detach().clone())

    # if len(master_grid) >=1:
    #     break

final_grid = make_grid(master_grid, nrow=3, padding=2, pad_value=1)
save_image(final_grid, os.path.join(PATH_output, 'matches_skyangular.png'))
