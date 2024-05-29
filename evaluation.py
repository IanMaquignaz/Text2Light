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
from tools import plot_histogram
from envmap import EnvironmentMap

# Clean matches
shutil.rmtree("matches_confirmed", ignore_errors=True)
os.makedirs("matches_confirmed", exist_ok=True)

PATH_panos_generated = "./generated_panorama"
PATH_panos_outdoor = "./outdoorPanosExr"
PATH_output = "./matches_confirmed"

FILE_MATCHES="matches.txt"

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
exit()

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
dr = DR(result_format='ratio')

from utils_ml.metrics import EarthMoversDistance as EMD # EMD
emd = EMD(bins=1000, range=[0,20000])

from utils_ml.metrics import IntegratedIllumination as II # Integrated Illumination
ii = II(result_format='ratio', shape=512, envmap_format='latlong')

def to_skyangular(img):
    e = EnvironmentMap(img, 'latlong')
    e = e.convertTo('skyangular', img.shape[1])
    return e.data


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


MATCHES_LDR = False
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
    img_gt = match_exposure(img_gn_ldr.copy(), img_gt.copy(), offset=0.4)
    img_gn_hdr = match_exposure(img_gt.copy(), img_gn_hdr.copy())

    if MATCHES_LDR == True:
        # Metrics LDR and GT_HDR
        img_gt_tm = tm_gamma(img_gt.copy(), 2.2, [0., 1.])
        img_gt_tm = toTensor(img_gt_tm).unsqueeze(0).float()
        img_gn_ldr_t = toTensor(img_gn_ldr.copy()).unsqueeze(0).float()
        l1.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        l2.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        dr.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        ii.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        emd.update(img_gn_ldr_t.clone(), img_gt_tm.clone())

        # LDR Metrics LDR and GT_LDR
        img_gt_tm = tm_gamma(img_gt.copy(), 2.2, [0., 1.])
        img_gt_tm = toTensor(to_skyangular(img_gt_tm)).unsqueeze(0).float()
        img_gn_ldr_t = toTensor(to_skyangular(img_gn_ldr.copy())).unsqueeze(0).float()
        clipIQA_real.update(img_gt_tm.clone())
        clipIQA_fake.update(img_gn_ldr_t.clone())
        uqi.update(img_gn_ldr_t.clone(), img_gt_tm.clone())
        lpips.update(img_gn_ldr_t.clone(), img_gt_tm.clone())

    elif MATCHES_LDR == False:
        # Metrics HDR and GT_HDR
        img_gt_t = toTensor(img_gt.copy()).unsqueeze(0).float()
        img_gn_hdr_t = toTensor(img_gn_hdr.copy()).unsqueeze(0).float()
        l1.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        l2.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        dr.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        ii.update(img_gn_hdr_t.clone(), img_gt_t.clone())
        emd.update(img_gn_hdr_t.clone(), img_gt_t.clone())

        img_gt_tm = tm_gamma(img_gt.copy(), 2.2, [0., 1.])
        img_gt_tm = toTensor(to_skyangular(img_gt_tm)).unsqueeze(0).float()
        img_gn_hdr_tm = tm_gamma(img_gn_hdr.copy(), 2.2, [0., 1.])
        img_gn_hdr_tm = toTensor(to_skyangular(img_gn_hdr_tm)).unsqueeze(0).float()
        clipIQA_real.update(img_gt_tm.clone())
        clipIQA_fake.update(img_gn_hdr_tm.clone())
        uqi.update(img_gn_hdr_tm.clone(), img_gt_tm.clone())
        lpips.update(img_gn_hdr_tm.clone(), img_gt_tm.clone())

    row_gt = get_skydome(img_gt.copy())
    row_gn_ldr = get_skydome(img_gn_ldr.copy())
    row_gn_hdr = get_skydome(img_gn_hdr.copy())

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
    # pair = np.concatenate((gt, gn_ldr, gn_hdr, hist), axis=0)
    pair = np.concatenate((gt, gn_hdr, hist), axis=0) # No LDR
    save_image(f'{PATH_output}/{match[1]}_{match[0]}.jpeg', pair)


print(f"L1: {l1.compute()}")
print(f"L2: {l2.compute()}")
print(f"DR: {dr.compute()}")
print(f"II: {ii.compute()}")
print(f"EMD: {emd.compute()}")

image = emd.plot_cummulative_histogram_error()
from torchvision.utils import save_image as tv_save_image
tv_save_image(image, 'loss_EMD.png')

print(f"CLIP_IQA real: {clipIQA_real.compute().mean()}")
print(f"CLIP_IQA fake: {clipIQA_fake.compute().mean()}")
print(f"UQI: {uqi.compute()}")
print(f"LPIPS: {lpips.compute()}")