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


# Custom
from tools import load_image, save_image
from tools import match_shape, match_exposure
from tools import tm_gamma, exposure
from tools import plot_histogram
from envmap import EnvironmentMap

PATH_panos_generated = "./generated_panorama"
PATH_panos_outdoor = "./outdoorPanosExr"
PATH_output = "./matches_confirmed"

FILE_MATCHES="matches.txt"

def get_skydome(img):
    count = 4
    e = EnvironmentMap(img.copy(), 'latlong')
    e = e.convertTo('skyangular', int(img.shape[1]/count))
    sample = e.data.copy()
    return sample


TARGET='9C4A7199'

# All Matches
matches = []
with open(FILE_MATCHES, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[1] == TARGET:
            matches.append(row[0])
    print('All Matches:', matches)

p_gt = os.path.join(PATH_panos_outdoor, TARGET+' Panorama_hdr.exr')
img_gt = load_image(p_gt)

images = []
for m in tqdm(matches):
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+m+'.png')
    img_gn_ldr = load_image(p_gn_ldr)

    p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_'+m+'.exr')
    img_gn_hdr = load_image(p_gn_hdr)


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
    img_gt = match_exposure(img_gn_ldr.copy(), img_gt.copy(), offset=0.4)
    img_gn_hdr = match_exposure(img_gt.copy(), img_gn_hdr.copy())
    img_gn_hdr = tm_gamma(img_gn_hdr, 2.2)
    # img_gn_hdr = get_skydome(img_gn_hdr)
    img_gn_hdr = img_gn_hdr[:trim,:,:]
    img_gn_hdr = cv2.putText(img_gn_hdr, m, (10, img_gn_hdr.shape[0]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    images.append(toTensor(img_gn_hdr))

from torchvision.utils import make_grid, save_image
grid = make_grid(images, nrow=3)
save_image(grid, os.path.join(TARGET+'_matches.png'))
