# Standard Library
import os
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path

# Machine Vision/Learning
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import torch
from torchvision.transforms import ToTensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
toTensor = ToTensor()
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)

# Custom
from tools import load_image, save_image
from tools import match_shape, match_exposure
from tools import L1, SIL

# Get HDRDB pano paths
DIR_panos_outdoor="./outdoorPanosExr"
REGEX_panos_outdoor=DIR_panos_outdoor+"/*.exr"
PATHS_panos_outdoor=glob(REGEX_panos_outdoor)

# Get generated pano paths
DIR_panos_generated="./generated_panorama/holistic"
REGEX_panos_generated=DIR_panos_generated+"/*.png"
PATHS_panos_generated=glob(REGEX_panos_generated)

# Clean matches
shutil.rmtree("matches", ignore_errors=True)
os.makedirs("matches", exist_ok=True)
if os.path.exists("matches.txt"):
    os.remove("matches.txt")

# p_ref = PATHS_panos_generated[0]
# ref=load_image(p_ref)
# ref=match_shape(ref,ref)
# print(p_ref, ref.shape, ref.min(),ref.max())
# save_image('matches/test_ref.jpeg', ref)

# p_a = PATHS_panos_outdoor[0]
# a = load_image(p_a)
# print(p_a, a.shape, a.min(), a.max())
# a = match_shape(ref, a)
# a = match_exposure(ref, a)
# print(p_a, a.shape, a.min(), a.max())

# save_image('matches/test_a.jpeg', a.astype(np.uint8))

# refa = np.concatenate((ref,a), axis=0)
# save_image('matches/test_cat.jpeg', refa)
# exit()

# PATHS_panos_generated = [
#     'holistic_[purple petaled flower].png',
#     'holistic_[green leaves tree under blue sky during golden hour].png',
#     'holistic_[orange petaled flowers near green trees at daytime].png',
#     'holistic_[green leafed plants during daytime].png',
#     'holistic_[green and white leafed plants].png',
#     'holistic_[green-leafed trees near mountain at daytime].png',
#     'holistic_[closeup photo of green leafed plants].png',
#     'holistic_[brown tree trunks lot].png',
#     'holistic_[purple petaled flower].png',
#     'holistic_[comulus clouds].png',
# ]
# PATHS_panos_generated = [ './generated_panorama/holistic/'+p for p in PATHS_panos_generated]

# PATHS_panos_outdoor = [
#     '9C4A7199 Panorama_hdr.exr',
#     '9C4A0307 Panorama_hdr.exr',
# ]
# PATHS_panos_outdoor = [ './outdoorPanosExr/'+p for p in PATHS_panos_outdoor]

# Save test log
with open("matches.txt", "a") as f:
    f.write(f"outdoorPanosExr, generated_panorama, loss\n")

cache={}
for p_ref in tqdm(PATHS_panos_generated):
    print(p_ref)
    ref=load_image(p_ref)
    ref=match_shape(ref,ref)

    best_match_loss = 999999
    best_match_path = None
    for p_a in PATHS_panos_outdoor:
        if p_a not in cache.keys():
            a=load_image(p_a)
            a=match_shape(ref,a)
            cache[p_a]=a.copy()
        else:
            a=cache[p_a].copy()
        a=match_exposure(ref,a)

        a_ = toTensor(a).unsqueeze(axis=0).float().cuda()
        ref_ = toTensor(ref).unsqueeze(axis=0).float().cuda()
        loss=0
        loss+=lpips(a_,ref_)
        # loss += SIL(ref,a)
        # loss += (1-PSNR(ref, a, data_range=1)/32)*.10
        # loss = MSE(ref,a)
        if loss < best_match_loss:
            best_match_loss = loss
            best_match_path = p_a
        print('\t', p_a, ' :: ', loss)


    # Save match
    print('Done! ', p_ref, '--> ', best_match_path, ' :: ', best_match_loss)
    a=cache[best_match_path]
    a=match_exposure(ref,a)
    best_img=np.concatenate((ref, a), axis=0)
    po_a = Path(best_match_path).stem.replace(' Panorama_hdr', '')
    po_ref = Path(p_ref).stem.replace('holistic_', '')
    p_out = 'matches/'+ po_a +'_'+ po_ref +'.jpeg'
    save_image(p_out, best_img)

    # Save test log
    with open("matches.txt", "a") as f:
        f.write(f"{po_a}, {po_ref}, {best_match_loss}\n")
