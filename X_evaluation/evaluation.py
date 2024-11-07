# Standard Library
import os
import csv
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path

# Machine Vision/Learning
import numpy as np
from torchvision.transforms import ToTensor
toTensor = ToTensor()

# Custom
from tools import load_image, save_image
from tools import match_shape, match_exposure
from tools import tm_gamma, exposure
from envmap import EnvironmentMap

PATH_panos_generated = "./generated_panorama_run_2"
PATH_panos_outdoor = "./outdoorPanosExr"
PATH_output = "./matches_confirmed"

# Clean matches
shutil.rmtree(PATH_output, ignore_errors=True)
os.makedirs(PATH_output, exist_ok=True)

FILE_MATCHES="run_2_matches_confirmed.txt"

# Unique Only
matches = {}
with open(FILE_MATCHES, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        matches[row[1]] = row
    matches = matches.values()
    print('Unique Matches:', len(matches))

# All Matches
matches = []
with open(FILE_MATCHES, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        matches.append(row)
    print('All Matches:', len(matches))

# Count Duplicates
matches = {}
with open(FILE_MATCHES, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[1] in matches:
            matches[row[1]] += 1
        else:
            matches[row[1]] = 1
    # matches = matches.values()
    print('Count Duplicates Matches:')
    true_unique=0
    for k,v in matches.items():
        if v > 1:
            print(k, v)
        else:
            true_unique += 1
    print('True Unique Matches:', true_unique)
# exit()


# Load unique filenames from unique.txt
def load_match_list(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    pairs = [ l.strip().split("; ") for l in lines]
    return pairs

FILE_MATCHES="run_2_matches_confirmed.txt"
matches = load_match_list(FILE_MATCHES)


from torchmetrics.multimodal import CLIPImageQualityAssessment
clipIQA_fake = CLIPImageQualityAssessment(model_name_or_path='openai/clip-vit-large-patch14-336', data_range=1.0, prompts=('quality',))
clipIQA_real = CLIPImageQualityAssessment(model_name_or_path='openai/clip-vit-large-patch14-336', data_range=1.0, prompts=('quality',))

from torchmetrics.image import UniversalImageQualityIndex
uqi = UniversalImageQualityIndex()

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
lpips = LPIPS(net_type='vgg', normalize=True)

from torchmetrics import MeanAbsoluteError as L1
l1 = L1()

from torchmetrics import MeanSquaredError as L2
l2 = L2()

from utils_ml.metrics import DynamicRange as DR # Dynamic Range
dr_real = DR(result_format='real')
dr_fake = DR(result_format='fake')

from utils_ml.metrics import EarthMoversDistance as EMD # EMD
emd = EMD(bins=1000, range=[0,20000])

from utils_ml.metrics import IntegratedIllumination as II # Integrated Illumination
ii_real = II(result_format='real', shape=512, envmap_format='latlong')
ii_fake = II(result_format='fake', shape=512, envmap_format='latlong')


def to_skyangular(img):
    e = EnvironmentMap(img, 'latlong')
    e = e.convertTo('skyangular', img.shape[1])
    return e.data

# def get_skydome(img):
#     count = 4
#     e = EnvironmentMap(img.copy(), 'latlong')
#     e = e.convertTo('skyangular', int(img.shape[1]/count))
#     sample = e.data.copy()
#     row = np.concatenate(
#         (
#             sample.copy()*0.1,
#             sample.copy()*0.01,
#             sample.copy()*0.001,
#             sample.copy()*0.00001,
#         ), axis=1)
#     return row

def get_skydome(img):
    e = EnvironmentMap(img.copy(), 'latlong')
    e = e.convertTo('skyangular', img.shape[1])
    return e.data.copy()

import torch
def boost_ldr2HDR(img):
    # img  = (img - img.min()) / (img.max() - img.min())
    # img = img * 2.0 - 1.0
    if img.min()*8 < -1.0 and img.max()*8 > 1.0:
        img = img * 7.0
    else:
        img  = (img - img.min()) / (img.max() - img.min())
        img = img * 2.0 - 1.0
        img = img*8
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

MATCHES_LDR = False
if MATCHES_LDR == False:
    l1_ldr = L1()
    l2_ldr = L2()
print('Matches:', len(matches))
for gt, gn in tqdm(matches):
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr/ldr_'+gn+'.png')
    img_gn_ldr = load_image(p_gn_ldr)

    # p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr/hdr_SRiTMO_boosted_'+gn+'.exr')
    # img_gn_hdr = load_image(p_gn_hdr)

    # Generated input to SRiTMO
    p_gn_ldr_raw = os.path.join(PATH_panos_generated, "hdr", f"ldr_input_{gn}.exr")
    img_gn_ldr_raw = load_image(p_gn_ldr_raw)
    img_gn_hdr = boost_ldr2HDR(img_gn_ldr_raw)

    p_gt_hdr = os.path.join(PATH_panos_outdoor, gt+' Panorama_hdr.exr')
    img_gt_hdr = load_image(p_gt_hdr)

    # Match shape and exposure
    trim=-76
    assert img_gn_ldr.shape == img_gn_hdr.shape
    img_gt_hdr = match_shape(img_gn_ldr, img_gt_hdr)

    # Make black border uniform
    img_gt_hdr[trim:,:,:]=0
    img_gn_ldr[trim:,:,:]=0
    img_gn_hdr[trim:,:,:]=0

    # Exposure
    # img_gt = exposure(img_gt, 12)
    img_gt_hdr = match_exposure(img_gn_ldr.copy(), img_gt_hdr.copy(), trim=trim)
    img_gn_hdr = match_exposure(img_gt_hdr.copy(), img_gn_hdr.copy(), trim=trim)

    if MATCHES_LDR == True:
        # Metrics LDR and GT_HDR
        img_gt_tm = tm_gamma(img_gt_hdr.copy(), 2.2, [0., 1.])
        img_gt_tm = toTensor(img_gt_tm).unsqueeze(0).float()
        img_gn_ldr_t = toTensor(img_gn_ldr.copy()).unsqueeze(0).float()
        l1.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        l2.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        dr_real.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        dr_fake.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        ii_real.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        ii_fake.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        emd.update(img_gn_ldr_t.clone(), img_gt_tm.clone())

        # LDR Metrics LDR and GT_LDR
        img_gt_tm = tm_gamma(to_skyangular(img_gt_hdr.copy()), 2.2, [0., 1.])
        img_gt_tm = toTensor(img_gt_tm).unsqueeze(0).float()
        img_gn_ldr_t = toTensor(to_skyangular(img_gn_ldr.copy())).unsqueeze(0).float()
        clipIQA_real.update(img_gt_tm.clone())
        clipIQA_fake.update(img_gn_ldr_t.clone())
        uqi.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        lpips.update(img_gn_ldr_t.clone(), img_gt_tm.clone())

    elif MATCHES_LDR == False:
        # Metrics HDR and GT_HDR
        img_gt_t = toTensor(img_gt_hdr.copy()).unsqueeze(0).float()
        img_gn_hdr_t = toTensor(img_gn_hdr.copy()).unsqueeze(0).float()
        l1.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        l2.update(img_gn_hdr_t.clone(), img_gt_t.clone())

        # For LDR L1 and L2 of HDR samples
        ref = toTensor(tm_gamma(img_gt_hdr.copy(), 2.2, [0., 1.])).unsqueeze(0).float()
        target = toTensor(tm_gamma(img_gn_hdr.copy(), 2.2, [0., 1.])).unsqueeze(0).float()
        l1_ldr.update(ref, target)
        l2_ldr.update(ref, target)

        dr_real.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        dr_fake.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        ii_real.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        ii_fake.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        emd.update(img_gn_hdr_t.clone(), img_gt_t.clone())

        img_gt_tm = tm_gamma(to_skyangular(img_gt_hdr.copy()), 2.2, [0., 1.])
        img_gt_tm = toTensor(img_gt_tm).unsqueeze(0).float()
        img_gn_hdr_tm = tm_gamma(to_skyangular(img_gn_hdr.copy()), 2.2, [0., 1.])
        img_gn_hdr_tm = toTensor(img_gn_hdr_tm).unsqueeze(0).float()
        clipIQA_real.update(img_gt_tm.clone())
        clipIQA_fake.update(img_gn_hdr_tm.clone())
        uqi.update(img_gn_hdr_tm.clone(), img_gt_tm.clone())
        lpips.update(img_gn_hdr_tm.clone(), img_gt_tm.clone())

    # row_gt = get_skydome(img_gt.copy())
    # row_gn_ldr = get_skydome(img_gn_ldr.copy())
    # row_gn_hdr = get_skydome(img_gn_hdr.copy())

    # # Trim black border
    # img_gt = img_gt[:trim,:,:]
    # img_gn_ldr = img_gn_ldr[:trim,:,:]
    # img_gn_hdr = img_gn_hdr[:trim,:,:]

    # # # Histogram
    # # hist = plot_histogram(img_gt, img_gn_hdr, plot_shape=(img_gt.shape[1],500))
    # # save_image(f'{PATH_output}/{match[1]}_{match[0]}_hist.jpeg', hist)
    # # import cv2
    # # hist = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)
    # # cv2.imwrite(f'{PATH_output}/{match[1]}_{match[0]}_hist.jpeg', hist)

    # # Save
    # gt = np.concatenate((row_gt,img_gt), axis=0)
    # gt = tm_gamma(gt, 2.2)
    # gn_ldr = np.concatenate((row_gn_ldr,img_gn_ldr), axis=0)
    # gn_hdr = np.concatenate((row_gn_hdr,img_gn_hdr), axis=0)
    # gn_hdr = tm_gamma(gn_hdr, 2.2)
    # # pair = np.concatenate((gt, gn_ldr, gn_hdr, hist), axis=0)
    # pair = np.concatenate((gt, gn_hdr, hist), axis=0) # No LDR
    # save_image(f'{PATH_output}/{match[1]}_{match[0]}.jpeg', pair)


if MATCHES_LDR == True:
    print("LDR Metrics")
else:
    print("HDR Metrics")
print(f"L1: {l1.compute()}")
print(f"L2: {l2.compute()}")
if MATCHES_LDR == False:
    print(f"L1 LDR: {l1_ldr.compute()}")
    print(f"L2 LDR: {l2_ldr.compute()}")
print(f"DR real: {dr_real.compute()}")
print(f"DR fake: {dr_fake.compute()}")
print(f"II real: {ii_real.compute()}")
print(f"II fake: {ii_fake.compute()}")
print(f"EMD: {emd.compute()}")

# image = emd.plot_cummulative_histogram_error()
# from torchvision.utils import save_image as tv_save_image
# tv_save_image(image, 'loss_EMD.png')

print(f"CLIP_IQA real: {clipIQA_real.compute().mean()}")
print(f"CLIP_IQA fake: {clipIQA_fake.compute().mean()}")
print(f"UQI: {uqi.compute()}")
print(f"LPIPS: {lpips.compute()}")