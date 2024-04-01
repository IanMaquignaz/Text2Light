
# Standard library
import os
import cv2
import numpy as np

# Machine Vision/Learning
import torch
from torchvision.transforms import functional as F

# Custom
from envmap import EnvironmentMap

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def load_image(path):
    # load the image
    try:
        img = cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise e

    if not path.endswith('*.exr') and img.max() > 1.0:
        img = img / 255.0
    # else:
    # if path.endswith('*.exr'):
    #     img=tm_gamma(img)
    return img


def tm_gamma(img, g=2.2, clip=[0,1]):
    # Gamma tonemap
    img = np.power(img, 1.0/g)
    if clip:
        img = np.clip(img, clip[0], clip[1])
    return img


def save_image(path, img):
    if not path.endswith(".exr"):
        img = img.astype(np.float32)
        img *=255
        img = np.clip(img, 0, 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def match_shape(ref, a):
    # Fix shape
    e = EnvironmentMap(a, 'latlong')
    if isinstance(ref, np.ndarray):
        e = e.convertTo('latlong', ref.shape[0])
    elif isinstance(ref, torch.Tensor):
        e = e.convertTo('latlong', ref.shape[-2])
    else:
        raise ValueError("ref must be a numpy array or a torch tensor")
    e.data[-16:,:,:]=0
    return e.data


def exposure(a, exp):
    a*= np.exp2(exp)
    return a

def match_exposure(ref, a, clip=False, offset=1.0):
    # Fix exposure
    if isinstance(ref, np.ndarray):
        ref_g = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2GRAY)
        a_g = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2GRAY)
        ratio = ref_g/a_g
        ratio = np.nan_to_num(ratio, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
        ratio = np.nanmean(ratio)
    elif isinstance(ref, torch.Tensor):
        ref_g = F.rgb_to_grayscale(ref)
        a_g = F.rgb_to_grayscale(a)
        ratio = ref_g/a_g
        ratio = torch.nan_to_num(ratio, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
        ratio = ratio.mean()
    else:
        raise ValueError("ref must be a numpy array or a torch tensor")
    # print('ratio=', ratio)
    a = a*(ratio*offset)
    if clip:
        a = np.clip(a,0,1)
    return a


def L1(a,b):
    return np.abs(a-b).mean()

def SIL(a,b):
    left = np.power(a-b,2).mean()
    right = np.power((a-b).mean(), 2)
    return left - right


import matplotlib.pyplot as plt
def plot_histogram(
    target,
    pred,
    bins=1000,
    range=(0.,20000.),
    plot_shape=(1000,600),
):
    # To Grayscale
    target_ = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2GRAY)
    pred_ = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_RGB2GRAY)

    # Get Histogram
    hist_target, bins_edges = np.histogram(target_, bins=bins, range=range, density=False, weights=None)
    hist_pred, bins_edges = np.histogram(pred_, bins=bins_edges, range=range, density=False, weights=None)
    # hist_error = np.abs(hist_pred - hist_target)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(plot_shape[0]*px, plot_shape[1]*px))

    # plt.hist(bins_edges[:-1], bins_edges, weights=(hist_error),
    #         label='Error', color='red', alpha=0.5, edgecolor='red', histtype='stepfilled')
    plt.hist(bins_edges[:-1], bins_edges, weights=(hist_target),
            label='Target', color='blue', alpha=0.6, edgecolor='blue', histtype='stepfilled')
    # plt.hist(bins_edges[:-1], bins_edges, weights=(hist_target),
    #         label='Targets', color='blue', alpha=0.2, edgecolor='blue', histtype='stepfilled')
    plt.hist(bins_edges[:-1], bins_edges, weights=(hist_pred),
            label='Prediction', color='red', alpha=1., edgecolor='red', histtype='step')

    plt.legend(loc='upper right')
    ax.set_xlabel(f'Intensity (bins={bins})')
    ax.set_ylabel('Count')
    ax.set_title('Intensity')

    # ax.set_yscale('log')
    # if range[1] > 8:
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # RGBA[0,255] to RGB [0,1]
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)/255.
    plt.close('all')
    return image