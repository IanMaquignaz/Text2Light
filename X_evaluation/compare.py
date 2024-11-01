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
from torch.nn.functional import l1_loss, mse_loss as l2_loss
from torchvision.utils import save_image as tv_save_image
from torchvision.transforms import ToTensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
toTensor = ToTensor()
device = np.random.randint(0, torch.cuda.device_count())
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).eval().to(device)

# Custom
from tools import load_image, save_image
from tools import tm_logN, tm_logN_torch, tm_gamma, exposure
from tools import match_shape, match_exposure, match_exposure_torch
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
DIR_panos_generated="./generated_panorama_run_2/ldr"
REGEX_panos_generated=DIR_panos_generated+"/ldr_*.png"
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

# find_duplicates(PATHS_panos_generated, thresh=0.0001, lsg=0, clean=True)
# find_duplicates(PATHS_panos_generated, thresh=0.0001, lsg=1, clean=False)


# Load unique filenames from unique.txt
def load_unique_list():
    with open('run_2_unique.txt', 'r') as f:
        lines = f.readlines()
    return [ DIR_panos_generated+'/ldr_'+l.strip()+'.png' for l in lines]
PATHS_panos_generated_unique = load_unique_list()
print(len(PATHS_panos_generated), len(PATHS_panos_generated_unique))

LSG=0
def find_matches(path_reference, paths_LDR=None, paths_HDR=None, lsg=None, LDR2HDR=False, clean=False, to_cuda=False):
    print('Finding matches...')

    if paths_LDR is None:
        paths_LDR = PATHS_panos_generated_unique
    if paths_HDR is None:
        paths_HDR = PATHS_panos_outdoor
    if lsg is None:
        lsg = LSG

    clip=[0,1]

    # Clean matches
    if clean:
        shutil.rmtree(NAME_MATCH_DIR, ignore_errors=True)
        os.makedirs(NAME_MATCH_DIR, exist_ok=True)
        if os.path.exists(NAME_MATCH_FILE):
            os.remove(NAME_MATCH_FILE)

    exp = 12
    shape = 256; trim=64
    format_dst='latlong'
    if lsg == 0 or lsg == 1:
        format_dst='skylatlong'
    elif lsg == 2:
        shape = 256
        format_dst='skyangular'

    def flip_image(k):
        if lsg==0:
            k=k[::-1,:,:] #Toggle to compare sky or ground
        if lsg == 0 or lsg == 1:
            k = convert_envmap(k, format_dst='skylatlong', size_dst=k.shape[0]//2)
        elif lsg == 2:
            k = convert_envmap(k, format_dst='skyangular', size_dst=k.shape[0])
        return k

    def trim_skydome(img):
        if lsg == -1: # latlong
            img = img[:-trim,:,:]
        elif lsg == 0: # skylatlong, ground up
            img = img[trim:,:,:]
        elif lsg == 1: # skylatlong, sky up
            img = img[:-trim,:,:]
        # lsg==2 == Skyangular, no trim
        return img

    cache_LDR={}
    def load_image_LDR(p_i):
        if p_i not in cache_LDR.keys():
            i = load_image(p_i)
            i = flip_image(i)
            i = match_shape(shape,i,format=format_dst)
            i = trim_skydome(i)
            i = i.astype(np.float32)
            i = toTensor(i).unsqueeze(0)
            if to_cuda: i = i.to(lpips.device)
            cache_LDR[p_i]=i.clone()
        else:
            i=cache_LDR[p_i].clone()
        return i

    cache_HDR={}
    def load_image_HDR(p_j):
        if p_j not in cache_HDR.keys():
            j = load_image(p_j)
            j = flip_image(j)
            j = match_shape(shape,j,format=format_dst)
            j = trim_skydome(j)
            j = exposure(j, exp)
            j = j.astype(np.float32)
            # j = tm_logN(j, base=2, clip=[0,1])
            j = toTensor(j).unsqueeze(0)
            if to_cuda: j = j.to(lpips.device)
            cache_HDR[p_j]=j.clone()
        else:
            j=cache_HDR[p_j].clone()
        return j


    if LDR2HDR:
        p_ref = path_reference
        paths_a = paths_HDR
        load_image_ref = load_image_LDR
        load_image_a = load_image_HDR
    else:
        p_ref = path_reference
        paths_a = paths_LDR
        load_image_ref = load_image_HDR
        load_image_a = load_image_LDR

    ref=load_image_ref(p_ref)
    best_match_loss = 999999
    best_match_path = None
    # for p_a in tqdm(paths_a):
    for p_a in paths_a:
        a = load_image_a(p_a)
        a_ = a

        if LDR2HDR:
            a_ = match_exposure_torch(ref, a)
            a_ = tm_logN_torch(a_, base=2, clip=clip)
            ref_ = ref
        else:
            ref_ = match_exposure_torch(a, ref)
            ref_ = tm_logN_torch(ref_, base=2, clip=clip)

        # compute loss
        # loss=lpips(a_,ref_)
        loss = l2_loss(a_, ref_)
        # loss=L1(a_,ref_); threshold=0.1
        # loss=SIL(a_,ref_)
        if loss < best_match_loss and a_.mean() > 0.01 and ref_.mean() > 0.01:
            best_match_loss = loss
            best_match_path = p_a
            tmp = torch.cat((a_,ref_), dim=-2)
            tv_save_image(tmp, 'temp.png')
            print('\t', p_ref, '->', p_a, ' :: ', loss)


    # Save match
    print('Done! ', p_ref, '--> ', best_match_path, ' :: ', best_match_loss)
    if best_match_path is not None:
        ref = load_image(p_ref)
        ref = match_shape(shape,ref,format='latlong')
        ref = toTensor(ref).unsqueeze(0)

        a = load_image(best_match_path)
        a = match_shape(shape,a,format='latlong')
        a = toTensor(a).unsqueeze(0)

        if LDR2HDR:
            a = exposure(a, exp)
            a = match_exposure_torch(ref, a)
            a = tm_logN_torch(a, base=2, clip=[0,1])
            po_ref = Path(p_ref).stem.replace(' Panorama_hdr', '').replace('ldr_', '')
            po_a = Path(best_match_path).stem.replace(' Panorama_hdr', '').replace('ldr_', '')
        else:
            ref = exposure(ref, exp)
            ref = match_exposure_torch(a, ref)
            ref = tm_logN_torch(ref, base=2, clip=[0,1])
            po_ref = Path(best_match_path).stem.replace(' Panorama_hdr', '').replace('ldr_', '')
            po_a = Path(p_ref).stem.replace(' Panorama_hdr', '').replace('ldr_', '')

        scale = 1000
        best_img = torch.cat((a,ref), dim=-2)
        p_out = NAME_MATCH_DIR+ po_a +'_'+ po_ref +'_'+f'{int(best_match_loss*scale)}'+'.jpeg'

        # save_image(p_out, best_img)
        tv_save_image(best_img, p_out)

        # Save test log
        with open(NAME_MATCH_FILE, "a") as f:
            f.write(f"{po_a}; {po_ref}; {best_match_loss}\n")


# find_matches(PATHS_panos_generated_unique, PATHS_panos_outdoor, lsg=0, clean=True)
# find_matches(PATHS_panos_generated_unique, PATHS_panos_outdoor, lsg=1, clean=False)
# find_matches(PATHS_panos_generated_unique, PATHS_panos_outdoor, lsg=0, clean=False, LDR2HDR=False)
# find_matches(PATHS_panos_generated_unique, PATHS_panos_outdoor, lsg=1, clean=False, LDR2HDR=False)
# find_matches(PATHS_panos_generated_unique, PATHS_panos_outdoor, lsg=-1, clean=True)
# find_matches(PATHS_panos_generated_unique, PATHS_panos_outdoor, lsg=2, clean=False, LDR2HDR=False)
import multiprocessing as mp

if __name__ == '__main__':
    # Clean matches
    shutil.rmtree(NAME_MATCH_DIR, ignore_errors=True)
    os.makedirs(NAME_MATCH_DIR, exist_ok=True)
    if os.path.exists(NAME_MATCH_FILE):
        os.remove(NAME_MATCH_FILE)

    # # find_matches(PATHS_panos_generated_unique[0], LDR2HDR=True)
    # find_matches(PATHS_panos_outdoor[0], LDR2HDR=False, lsg=-1)
    # find_matches(PATHS_panos_outdoor[0], LDR2HDR=False, lsg=0)
    # find_matches(PATHS_panos_outdoor[0], LDR2HDR=False, lsg=1)
    # find_matches(PATHS_panos_outdoor[0], LDR2HDR=False, lsg=2)


    pids = 6
    with mp.get_context('spawn').Pool(pids) as pool:
    #     # Parallelize find_matches by splitting PATHS_panos_generated_unique into chunks
    #     pool.map(find_matches, PATHS_panos_generated_unique, chunksize=50)
        LSG=0
        pool.map(find_matches, PATHS_panos_outdoor, chunksize=50)
        LSG=1
        pool.map(find_matches, PATHS_panos_outdoor, chunksize=50)
        LSG=2
        pool.map(find_matches, PATHS_panos_outdoor, chunksize=50)
