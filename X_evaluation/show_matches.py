# Standard Library
import os
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path

# Torch
import torch
import torchvision.transforms.functional as F
from torchvision.utils import save_image as tv_save_image
from torchvision.transforms import ToTensor; toTensor = ToTensor()

# Custom
from tools import load_image, save_image
from tools import tm_logN_torch
from tools import match_shape, match_exposure_torch

# Matches
DIR_MATCHES_CONFIRMED="./run_2_matches_confirmed/"
NAME_MATCH_FILE="run_2_matches_confirmed.txt"

# Get HDRDB pano paths
DIR_panos_outdoor="./outdoorPanosExr/"
TAIL_panos_outdoor=" Panorama_hdr.exr"

# Get generated pano paths
DIR_panos_generated="./generated_panorama_run_2/ldr/"
HEADER_panos_generated="ldr_"
TAIL_panos_generated=".png"

# Load unique filenames from unique.txt
def load_match_list(fullpath=False):
    with open(NAME_MATCH_FILE, 'r') as f:
        lines = f.readlines()
    matches = [ l.strip().split("; ") for l in lines]
    if fullpath:
        matches = [
            [
                DIR_panos_outdoor+r+TAIL_panos_outdoor,
                DIR_panos_generated+HEADER_panos_generated+a+TAIL_panos_generated
            ] for r,a in matches
        ]
    return matches

def match_scale(ref, gen):
    # Match scale
    ref = F.resize(ref, gen.shape[-2:])
    return ref

if __name__ == '__main__':
    # Clean matches
    shutil.rmtree(DIR_MATCHES_CONFIRMED, ignore_errors=True)
    os.makedirs(DIR_MATCHES_CONFIRMED, exist_ok=True)

    # Load matches
    matches = load_match_list()
    matches_fullPath = load_match_list(fullpath=True)
    for m, M in tqdm(zip(matches, matches_fullPath)):
        ref, gen = M
        ref = load_image(ref)
        gen = load_image(gen)
        # ref = match_shape(gen,ref,format='latlong')
        ref = ref[:-800, :, :]
        ref = toTensor(ref).unsqueeze(0)
        gen = toTensor(gen).unsqueeze(0)
        ref = match_scale(ref, gen)

        ref = match_exposure_torch(gen,ref)
        ref = tm_logN_torch(ref, base=2, clip=[0,1])

        # img = torch.cat((gen,ref), dim=-2)
        diff = torch.abs(gen-ref)
        norm_diff = (diff-diff.min())/(diff.max()-diff.min())
        img = torch.cat((gen,ref,diff), dim=-2)
        p_out = DIR_MATCHES_CONFIRMED+m[0]+'_'+m[1]+'.png'
        tv_save_image(img, p_out)