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
# from torchvision.transforms import ToTensor
# toTensor = ToTensor()

# Custom
from tools import tm_gamma, tm_gamma_inverse, load_image
from utils_cv.io.opencv import save_image
# from utils_color.tonemap import Gamma
# tm = Gamma(2.2)

PATH_panos_generated = "./generated_panorama_run_2"
PATH_panos_outdoor = "./outdoorPanosExr"
PATH_output_matched_exposures = "./run_2_matches_confirmed_grid/matched_exposures"
PATH_output_renders = "./run_2_matches_confirmed_grid/renders"

# shutil.rmtree(PATH_output_renders, ignore_errors=True)
# os.makedirs(PATH_output_renders, exist_ok=True)
# shutil.rmtree(PATH_output_matched_exposures, ignore_errors=True)
# os.makedirs(PATH_output_matched_exposures, exist_ok=True)

NAMES_IMAGES = [
    ("9C4A1280", "[grass field under cloudy sky during daytime]"),
    ("9C4A1322", "[CN tower during daytime]"),
    ("9C4A3692", "[grayscale photo of buildings]"),
    ("9C4A3916", "[green grass at daytime]"),
    ("9C4A4042", "[green trees during daytime]"),
    ("9C4A4700", "[green and brown tree during daytime]"),
    ("9C4A5050", "[green trees under cloudy sky during daytime]"),
    ("9C4A5554", "[house and trees]"),
    ("9C4A5638", "[brown house near body of water]"),
    ("9C4A5855", "[gray mountain covered in fog during daytime]"),
    ("9C4A5939", "[village under cloudy sky during daytime]"),
    ("9C4A6107", "[panorama photo of bridge]"),
    ("9C4A6149", "[Sign Post, Arrow Sign, Direction Sign]"),
    ("9C4A6345", "[landscape photo of waterfalls flowing into river during daytime]"),
    ("9C4A6828", "[aerial view of green grass field]"),
    ("9C4A6870", "[cone mountain with the distance of body of water]"),
    ("9C4A6954", "[four brown wooden boat near dock]"),
    ("9C4A7199", "[green leafed plants during daytime]"),
    ("9C4A7633", "[lake under blue sky during daytime]"),
    ("9C4A9410", "[brown wooden house near brown mountain and green trees during daytime]"),
    ("9C4A9515", "[green pine trees near teal water at daytime]"),
    ("9C4A9711", "[grey concrete arch]"),
]

import blender_runner as br

print('START')

scenes = br.find_scenes('blender')
print('Scenes: ', list(scenes.keys()))
# print(scenes)

from tools import match_exposure, match_shape
def match_exposures_for_render(
    gt,
    gn,
    gn_ldr,
):
    # LDR Generated
    trim=-76
    p_gn_ldr = os.path.join(PATH_panos_generated, 'ldr', gn_ldr)
    img_gn_ldr = load_image(p_gn_ldr)
    img_gn_ldr[trim:,:,:]=0
    img_gn_ldr = match_exposure(img_gn_ldr.copy(),img_gn_ldr)

    # HDR ground truth
    p_gt = os.path.join(PATH_panos_outdoor, gt)
    img_gt_hdr = load_image(p_gt)
    img_gt_hdr = match_shape(img_gn_ldr.copy(), img_gt_hdr)
    img_gn_ldr = img_gn_ldr[:trim,:,:]
    img_gt_hdr = img_gt_hdr[:trim,:,:]

    img_gt_hdr = match_exposure(img_gn_ldr.copy(), img_gt_hdr)
    save_image(os.path.join(PATH_output_matched_exposures, gt), img_gt_hdr.astype(np.float32))

    # Generated
    p_gn_hdr = os.path.join(PATH_panos_generated, 'hdr', gn)
    img_gn_hdr = load_image(p_gn_hdr)
    img_gn_hdr = img_gn_hdr[:trim,:,:]
    img_gn_hdr = match_exposure(img_gt_hdr.copy(), img_gn_hdr)
    save_image(os.path.join(PATH_output_matched_exposures, gn), img_gn_hdr.astype(np.float32))


for (gt, gn) in NAMES_IMAGES:
    file_gt = f'{gt} Panorama_hdr.exr'
    file_gn = f'hdr_SRiTMO_boosted_{gn}.exr'
    file_gn_ldr = f'ldr_{gn}.png'

    # Match Exposures
    # match_exposures_for_render(
    #     gt=file_gt,
    #     gn=file_gn,
    #     gn_ldr=file_gn_ldr,
    # )

    # br.run_blender(
    #     render='image',
    #     scene=scenes['Scene_Default'],
    #     src=os.path.join(PATH_output_matched_exposures, file_gt),
    #     dst=os.path.join(PATH_output_renders, file_gt),
    #     img_fmt='exr',
    #     overwrite=False,
    #     render_every_nTh_image=1,
    # )
    # Generated
    br.run_blender(
        render='image',
        scene=scenes['Scene_Default'],
        src=os.path.join(PATH_output_matched_exposures, file_gn),
        dst=os.path.join(PATH_output_renders, file_gn),
        img_fmt='exr',
        overwrite=False,
        render_every_nTh_image=1,
    )
