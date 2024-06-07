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
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()

# Custom
from tools import load_image, save_image
from tools import tm_logN, tm_gamma, exposure
from tools import match_shape, match_exposure
from tools import L1, SIL
from tools import convert_envmap

NAME_MATCH_FILE='matches.txt'
NAME_MATCH_DIR='matches/'

NAME_DUPLICATES_FILE='duplicates.txt'
NAME_DUPLICATES_DIR='duplicates/'


# Get HDRDB pano paths
DIR_panos_outdoor="./outdoorPanosExr"
REGEX_panos_outdoor=DIR_panos_outdoor+"/*.exr"
PATHS_panos_outdoor=glob(REGEX_panos_outdoor)

# Get generated pano paths
DIR_panos_generated="./generated_panorama_run_2/holistic"
REGEX_panos_generated=DIR_panos_generated+"/*.png"
PATHS_panos_generated=glob(REGEX_panos_generated)


def find_duplicates(paths, thresh=0.1, lsg=-1, clean=False):
    # Clean Duplicates
    if clean:
        shutil.rmtree(NAME_DUPLICATES_DIR, ignore_errors=True)
        os.makedirs(NAME_DUPLICATES_DIR, exist_ok=True)
        if os.path.exists(NAME_DUPLICATES_FILE):
            os.remove(NAME_DUPLICATES_FILE)

    def flip_image(k):
        if lsg==0:
            k=k[::-1,:,:] #Toggle to compare sky or ground
        if lsg >= 0:
            k = convert_envmap(k, format_dst='skylatlong', size_dst=k.shape[0]//2)
        return k

    scale = (1/thresh)*10
    def load_image_cached(p_a):
        if p_a not in cache.keys():
            a = load_image(p_a)
            a = flip_image(a)
            cache[p_a]=a.copy()
        else:
            a=cache[p_a].copy()
        return a

    cache={}
    thresh_content = 0.01
    paths_to_search = paths.copy()

        # Init log
    with open(NAME_DUPLICATES_FILE, "a") as f:
        f.write(f"generated_panorama; generated_panorama; loss ({len(paths)} Total)\n")

        while len(paths_to_search) > 0:
            p_ref = paths_to_search[0]
            ref=load_image_cached(p_ref)

            if np.mean(ref) > thresh_content:
                paths_maybe_duplicates = paths_to_search.copy()
                for p_a in tqdm(paths_maybe_duplicates):
                    a = load_image_cached(p_a)

                    # compute loss
                    loss=MSE(a,ref)
                    if loss < thresh:
                        # remove key
                        paths_to_search.remove(p_a)

                        # Save match
                        # print('Done! ', p_ref, '--> ', p_a, ' :: ', loss)
                        best_img=np.concatenate((load_image(p_ref), load_image(p_a)), axis=0)
                        po_a = Path(p_a).stem.replace('holistic_', '')
                        po_ref = Path(p_ref).stem.replace('holistic_', '')
                        p_out = NAME_DUPLICATES_DIR+ po_ref +'_'+ po_a +f'_{int(loss*scale)}_'+'.jpeg'
                        save_image(p_out, best_img)

                        # log
                        f.write(f"{po_ref}; {po_a}; {loss}\n")
            else:
                paths_to_search.pop()
                print(f"Failed threshold({thresh_content}) mean={np.mean(ref)}")
            # log
            f.write(f"")

find_duplicates(PATHS_panos_generated, thresh=0.0001, lsg=0, clean=True)
find_duplicates(PATHS_panos_generated, thresh=0.0001, lsg=1, clean=False)


# Load unique filenames from unique.txt
def load_unique_list():
    with open('unique.txt', 'r') as f:
        lines = f.readlines()
    return [ DIR_panos_generated+'/holistic_'+l.strip()+'.png' for l in lines]
PATHS_panos_generated_unique = load_unique_list()
print(len(PATHS_panos_generated), len(PATHS_panos_generated_unique))


def find_matches(paths_LDR, paths_HDR, lsg=-1, LDR2HDR=True, clean=False):
    # Clean matches
    if clean:
        shutil.rmtree(NAME_MATCH_DIR, ignore_errors=True)
        os.makedirs(NAME_MATCH_DIR, exist_ok=True)
        if os.path.exists(NAME_MATCH_FILE):
            os.remove(NAME_MATCH_FILE)

    exp = 16
    shape = 256
    format_dst='latlong'
    if lsg >= 0:
        format_dst='skylatlong'

    def flip_image(k):
        if lsg==0:
            k=k[::-1,:,:] #Toggle to compare sky or ground
        if lsg >= 0:
            k = convert_envmap(k, format_dst='skylatlong', size_dst=k.shape[0]//2)
        return k

    cache_LDR={}
    def load_image_LDR(p_i):
        if p_i not in cache_LDR.keys():
            i = load_image(p_i)
            i = flip_image(i)
            cache_LDR[p_i]=i.copy()
        else:
            i=cache_LDR[p_i].copy()
        return i

    cache_HDR={}
    def load_image_HDR(p_j):
        if p_j not in cache_HDR.keys():
            j = load_image(p_j)
            j = flip_image(j)
            j = exposure(j, exp)
            j = tm_logN(j, base=2, clip=[0,1])
            cache_HDR[p_j]=j.copy()
        else:
            j=cache_HDR[p_j].copy()
        return j

    if LDR2HDR:
        paths_ref = paths_LDR
        paths_a = paths_HDR
        load_image_ref = load_image_LDR
        load_image_a = load_image_HDR
    else:
        paths_ref = paths_HDR
        paths_a = paths_LDR
        load_image_ref = load_image_HDR
        load_image_a = load_image_LDR

    for p_ref in tqdm(paths_ref):
        ref=load_image_ref(p_ref)
        ref=match_shape(shape,ref,format=format_dst)

        best_match_loss = 999999
        best_match_path = None
        for p_a in paths_a:
            a = load_image_a(p_a)
            a=match_shape(shape,a,format=format_dst)
            if LDR2HDR:
                a_ = a
                # a_ = match_exposure(ref, a)
                # a_ = tm_logN(a_, base=2, clip=[0,1])
                ref_ = ref
            else:
                a_ = a
                ref_ = ref
                # ref_ = match_exposure(a, ref)
                # ref_ = tm_logN(ref_, base=2, clip=[0,1])

            # compute loss
            # loss=lpips(a_,ref_)
            # loss=SIL(a,ref)
            loss=MSE(a_,ref_)
            if loss < best_match_loss:
                best_match_loss = loss
                best_match_path = p_a
            # print('\t', p_a, ' :: ', loss)


        # Save match
        print('Done! ', p_ref, '--> ', best_match_path, ' :: ', best_match_loss)
        if best_match_path is not None:
            ref = load_image(p_ref)
            ref = match_shape(shape,ref)
            a = load_image(best_match_path)
            a = match_shape(shape,a)

            if LDR2HDR:
                # a = match_exposure(ref, a)
                a = exposure(a, exp)
                a = tm_logN(a, base=2, clip=[0,1])
                best_img = np.concatenate((ref, a), axis=0)
                po_ref = Path(p_ref).stem.replace('holistic_', '')
                po_a = Path(best_match_path).stem.replace(' Panorama_hdr', '')
                p_out = NAME_MATCH_DIR+ po_a +'_'+ po_ref +'.jpeg'
            else:
                # ref = match_exposure(a, ref)
                ref = exposure(ref, exp)
                ref = tm_logN(ref, base=2, clip=[0,1])
                best_img = np.concatenate((a,ref), axis=0)
                po_a = Path(best_match_path).stem.replace('holistic_', '')
                po_ref = Path(p_ref).stem.replace(' Panorama_hdr', '')
                p_out = NAME_MATCH_DIR+ po_ref +'_'+ po_a +'.jpeg'

            save_image(p_out, best_img)

            # Save test log
            with open(NAME_MATCH_FILE, "a") as f:
                f.write(f"{po_a}; {po_ref}; {best_match_loss}\n")


find_matches(PATHS_panos_generated, PATHS_panos_outdoor, lsg=0, clean=True)
find_matches(PATHS_panos_generated, PATHS_panos_outdoor, lsg=1, clean=False)
find_matches(PATHS_panos_generated, PATHS_panos_outdoor, lsg=0, clean=False, LDR2HDR=False)
find_matches(PATHS_panos_generated, PATHS_panos_outdoor, lsg=1, clean=False, LDR2HDR=False)
