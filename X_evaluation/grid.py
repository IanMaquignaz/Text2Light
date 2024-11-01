# Standard Library
import os
import csv
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path

# Machine Vision/Learning
import cv2
import numpy as np
from torchvision.transforms import ToTensor
toTensor = ToTensor()
from torchvision.transforms import functional as F

# Custom
from tools import load_image
from tools import match_shape, match_exposure
from tools import tm_gamma, exposure
from tools import plot_histogram
from envmap import EnvironmentMap
from utils_ml.metrics import EarthMoversDistance as EMD # EMD


PATH_panos_generated = "./generated_panorama_run_2"
PATH_panos_outdoor = "./outdoorPanosExr"
PATH_output = "./run_2_matches_confirmed_grid"

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


matches = load_match_list(FILE_MATCHES)

for k, v in tqdm(matches.items()):

    trim=-76
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+v[0]+'.png')
    img_gn_ldr = load_image(p_gn_ldr)
    img_gn_ldr[trim:,:,:]=0
    img_gn_ldr = match_exposure(img_gn_ldr,img_gn_ldr)

    p_gt = os.path.join(PATH_panos_outdoor, k+' Panorama_hdr.exr')
    img_gt = load_image(p_gt)

    # Match shape and exposure
    img_gt = match_shape(img_gn_ldr, img_gt)
    img_gn_ldr = img_gn_ldr[:trim,:,:]
    img_gt = img_gt[:trim,:,:]
    img_gn_ldr_ = img_gn_ldr.copy()

    # from utils_ml.metrics import EarthMoversDistance as EMD # EMD
    emd_LDR = EMD(bins=1000, range=[0,1])
    emd_HDR = EMD(bins=1000, range=[0,20000])


    images = []
    for m in v:
        p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+m+'.png')
        img_gn_ldr = load_image(p_gn_ldr)
        img_gn_ldr = img_gn_ldr[:trim,:,:]
        img_gn_ldr = match_exposure(img_gn_ldr,img_gn_ldr)

        # LDR EMD
        img_gt_tmp_HDR = match_exposure(img_gn_ldr.copy(), img_gt.copy())
        img_gt_tmp_LDR = tm_gamma(img_gt_tmp_HDR, 2.2)
        emd_LDR.update(toTensor(img_gn_ldr).unsqueeze(0),toTensor(img_gt_tmp_LDR).unsqueeze(0))

        # HDR EMD
        p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_SRiTMO_boosted_'+m+'.exr')
        img_gn_hdr = load_image(p_gn_hdr)
        img_gn_hdr = img_gn_hdr[:trim,:,:]
        img_gn_hdr = match_exposure(img_gn_ldr.copy(), img_gn_hdr.copy())

        emd_HDR.update(toTensor(img_gn_hdr.copy()).unsqueeze(0), toTensor(img_gt_tmp_HDR.copy()).unsqueeze(0))

        img_gn_hdr = tm_gamma(img_gn_hdr, 2.2)
        img_gn_hdr = cv2.putText(img_gn_hdr, m, (10, img_gn_hdr.shape[0]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        images.append(toTensor(img_gn_hdr))

    img_gt = match_exposure(img_gn_ldr_, img_gt.copy())
    img_gt = tm_gamma(img_gt, 2.2)
    img_gt = cv2.putText(img_gt, 'Ground Truth', (10, img_gt.shape[0]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    images.insert(0, toTensor(img_gt.copy()))

    print(f"EMD_LDR: {emd_LDR.compute()}")
    img_EMD_LDR = emd_LDR.plot_cummulative_histogram_error(include_error=False).numpy().transpose(1,2,0)
    img_EMD_LDR = cv2.cvtColor(img_EMD_LDR, cv2.COLOR_RGBA2RGB)
    img_EMD_LDR = cv2.resize(img_EMD_LDR, (img_gt.shape[1], img_gt.shape[0]))
    img_EMD_LDR = cv2.putText(img_EMD_LDR, 'LDR', (10, img_gt.shape[0]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    images.append(toTensor(img_EMD_LDR))

    print(f"EMD_HDR: {emd_HDR.compute()}")
    img_EMD_HDR = emd_HDR.plot_cummulative_histogram_error(include_error=False).numpy().transpose(1,2,0)
    img_EMD_HDR = cv2.cvtColor(img_EMD_HDR, cv2.COLOR_RGBA2RGB)
    img_EMD_HDR = cv2.resize(img_EMD_HDR, (img_gt.shape[1], img_gt.shape[0]))
    img_EMD_HDR = cv2.putText(img_EMD_HDR, 'HDR', (10, img_gt.shape[0]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    images.append(toTensor(img_EMD_HDR))

    from torchvision.utils import make_grid, save_image
    if len(images) % 3 == 0:
        grid = make_grid(images, nrow=3)
    else:
        grid = make_grid(images, nrow=2)
    save_image(grid, os.path.join(PATH_output, k+'_matches.png'))
