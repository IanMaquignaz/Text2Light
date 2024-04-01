# Standard Library
import os
import csv
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path

# Machine Vision/Learning
import numpy as np

# Custom
from tools import load_image, save_image
from tools import match_shape, match_exposure
from tools import tm_gamma, exposure
from tools import plot_histogram
from envmap import EnvironmentMap

# Clean matches
shutil.rmtree("matches_confirmed", ignore_errors=True)
os.makedirs("matches_confirmed", exist_ok=True)

PATH_panos_generated = "./generated_panorama"
PATH_panos_outdoor = "./outdoorPanosExr"
PATH_output = "./matches_confirmed"

matches = []
FILE_MATCHES="matches.txt"
with open(FILE_MATCHES, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        matches.append(row)


def get_skydome(img):
    count = 4
    e = EnvironmentMap(img.copy(), 'latlong')
    e = e.convertTo('skyangular', int(img.shape[1]/count))
    sample = e.data.copy()
    row = np.concatenate(
        (
            sample.copy()*0.1,
            sample.copy()*0.01,
            sample.copy()*0.001,
            sample.copy()*0.00001,
        ), axis=1)
    return row

# matches = matches[:5]
for match in tqdm(matches):
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+match[0]+'.png')
    img_gn_ldr = load_image(p_gn_ldr)

    p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_'+match[0]+'.exr')
    img_gn_hdr = load_image(p_gn_hdr)

    p_gt = os.path.join(PATH_panos_outdoor, match[1]+' Panorama_hdr.exr')
    img_gt = load_image(p_gt)

    # Match shape and exposure
    trim=-76
    assert img_gn_ldr.shape == img_gn_hdr.shape
    img_gt = match_shape(img_gn_ldr, img_gt)

    # Make black border uniform
    img_gt[trim:,:,:]=0
    img_gn_ldr[trim:,:,:]=0
    img_gn_hdr[trim:,:,:]=0

    # Exposure
    # img_gt = exposure(img_gt, 12)
    img_gt = match_exposure(img_gn_ldr, img_gt, offset=0.4)
    img_gn_hdr = match_exposure(img_gt, img_gn_hdr)

    row_gt = get_skydome(img_gt)
    row_gn_ldr = get_skydome(img_gn_ldr)
    row_gn_hdr = get_skydome(img_gn_hdr)

    # Trim black border
    img_gt = img_gt[:trim,:,:]
    img_gn_ldr = img_gn_ldr[:trim,:,:]
    img_gn_hdr = img_gn_hdr[:trim,:,:]

    # Histogram
    hist = plot_histogram(img_gt, img_gn_hdr, plot_shape=(img_gt.shape[1],500))
    save_image(f'{PATH_output}/{match[1]}_{match[0]}_hist.jpeg', hist)
    # import cv2
    # hist = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f'{PATH_output}/{match[1]}_{match[0]}_hist.jpeg', hist)

    # Save
    gt = np.concatenate((row_gt,img_gt), axis=0)
    gt = tm_gamma(gt, 2.2)
    gn_ldr = np.concatenate((row_gn_ldr,img_gn_ldr), axis=0)
    gn_hdr = np.concatenate((row_gn_hdr,img_gn_hdr), axis=0)
    gn_hdr = tm_gamma(gn_hdr, 2.2)
    pair = np.concatenate((gt, gn_ldr, gn_hdr, hist), axis=0)
    save_image(f'{PATH_output}/{match[1]}_{match[0]}.jpeg', pair)
