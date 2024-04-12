import os
import shutil
from glob import glob
from pathlib import Path
from utils_cv.io.opencv import load_image, save_image

import cv2
import torch
import numpy as np

def clean(path):
    path = path.replace('$','')
    path = path.replace('\r','')
    return path

def purge(paths):
    for p in paths:
        p_ = clean(p)
        if p == p_:
            continue
        os.rename(p, p_)

# purge(glob('generated_panorama/ldr/*.png'))
# purge(glob('generated_panorama/holistic/*.png'))
# purge(glob('generated_panorama/hdr/*.exr'))
# exit()

shutil.rmtree("test", ignore_errors=True)
os.makedirs("test", exist_ok=True)

def wft(hdr_output:torch.Tensor):
    gamma = 1
    boost = 0
    balance = 0.7
    # threshold = 0.83
    threshold = 0.99

    luma = torch.mean(hdr_output, dim=1, keepdim=True)
    mask = torch.clip(luma / luma.max() - threshold, 0, 1)
    hdr_output += hdr_output * mask * boost
    hdr_output = torch.exp((hdr_output - hdr_output.mean()) * gamma - balance)
    return hdr_output


''' Reinhard tonemapping operator (Numpy) '''
def functional_np_reinhard(
        hdr,
        gamma=1.5,
        intensity = 0.,
        light_adapt = 0.,
        color_adapt = 0.
    ):
    scale = hdr.max() - hdr.min()
    if hdr.max() - hdr.min() > 1e-6:
        hdr = (hdr - hdr.min()) / (hdr.max() - hdr.min())
    # the input is opencv RGB format MxNx3
    gray = 0.299*hdr[..., 0] + 0.587*hdr[..., 1] + 0.114*hdr[..., 2]
    log_ = np.log(np.clip(gray, 1e-4, np.inf))
    log_mean = log_.mean()
    log_min = log_.min()
    log_max = log_.max()
    key = (log_max - log_mean) / (log_max - log_min)
    map_key = 0.3 + 0.7*(key**1.4)
    gray_mean = gray.mean()

    intensity = np.exp(-intensity)
    hdr = hdr*(1/(gray_mean**map_key + hdr))
    hdr = hdr**(1/gamma)
    return hdr, scale*(gray_mean**map_key)


PATHS_LDR = glob('generated_panorama/ldr/ldr_*.png')
for path in PATHS_LDR[::200]:

    path_LDR = Path(path)
    newNameLDR = path_LDR.name.replace('ldr_', '').replace('.png', '_LDR.png')

    nameHDR = path_LDR.name.replace('ldr_', 'hdr_').replace('.png', '.exr')
    path_HDR = Path('generated_panorama') / 'hdr' / nameHDR
    newNameHDR = path_LDR.name.replace('ldr_', '').replace('.png', '_HDR.exr')
    newNameWTF = newNameHDR.replace('_HDR.exr', '_WTF.exr')
    # print(path_LDR)
    # print(newNameLDR)
    # print(path_HDR)
    # print(newNameHDR)
    # print(newNameWTF)

    shutil.copyfile(path_LDR, f'test/{newNameLDR}')
    shutil.copyfile(path_HDR, f'test/{newNameHDR}')

    img_HDR_np = load_image(f'test/{newNameHDR}')
    # cv2.cvtColor(img_HDR_np.astype(np.float32), cv2.COLOR_RGB2BGR, img_HDR_np)
    img_LDR_np, _ = functional_np_reinhard(img_HDR_np)
    print(path, img_LDR_np.shape, img_LDR_np.dtype, img_LDR_np.min(), img_LDR_np.max())
    img_LDR_ts = torch.from_numpy(img_LDR_np.copy().astype(np.float32)).permute(2, 0, 1)
    # img_LDR_ts = torch.from_numpy(img_LDR_np.copy().astype(np.float32))

    img_LDR_ts = img_LDR_ts.unsqueeze(0)
    img_LDR_ts = (img_LDR_ts*16)-8
    img_wtf_ts = wft(img_LDR_ts)

    img_wtf_ts = torch.nan_to_num(img_wtf_ts.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)

    img_wtf_np = img_wtf_ts.permute(1, 2, 0).numpy().astype(np.float32)
    img_wtf_np = np.nan_to_num(img_wtf_np, nan=0.0, posinf=0.0, neginf=0.0)
    # cv2.cvtColor(img_wtf_np, cv2.COLOR_BGR2RGB, img_wtf_np)

    save_image(f'test/{newNameWTF}', img_wtf_ts, float16=False)
