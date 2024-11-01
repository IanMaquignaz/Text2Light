
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


def tm_logN(img, base=2, selective=False, clip=[0,1]):
    # log_b (a) = log_k (a) / log_k (b)
    if selective:
        t = img
        t[t > base] = np.log2(t[t > base]+base**base) / np.log2(base)
    else:
        t = np.log2(img+1) / np.log2(base)
    t = np.nan_to_num(t, nan=0, posinf=0, neginf=0)
    if clip:
        t = np.clip(t, clip[0], clip[1])
    return t


def tm_logN_torch(img, base=2, selective=False, clip=[0,1]):
    # log_b (a) = log_k (a) / log_k (b)
    if selective:
        t = img
        t[t > base] = torch.log2(t[t > base]+base**base) / torch.full(t[t > base].shape, np.log2(base)).to(img.device)
    else:
        t = torch.log2(img+1) / torch.full(img.shape, np.log2(base)).to(img.device)
    t = torch.nan_to_num(t, nan=0, posinf=0, neginf=0)
    if clip:
        t = torch.clip(t, clip[0], clip[1])
    return t


def save_image(path, img):
    if not path.endswith(".exr"):
        img = img.astype(np.float32)
        img *=255
        img = np.clip(img, 0, 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def match_shape(ref, a, format='latlong'):
    # Fix shape
    e = EnvironmentMap(a, format)
    if isinstance(ref, np.ndarray):
        e = e.convertTo(format, ref.shape[0])
    elif isinstance(ref, torch.Tensor):
        e = e.convertTo(format, ref.shape[-2])
    elif isinstance(ref, int):
        e = e.convertTo(format, ref)
    else:
        raise ValueError("ref must be a numpy array or a torch tensor")
    return e.data


def convert_envmap(img, format_src='latlong', format_dst='latlong', size_dst=None):
    # Fix shape
    e = EnvironmentMap(img, format_src)
    # if size_dst is None:
    #     size_dst = e.data.shape[0]
    e = e.convertTo(format_dst, size_dst)
    return e.data

def exposure(a, exp):
    if isinstance(exp, torch.Tensor):
        a*= torch.exp2(torch.Tensor(exp))
    else:
        a*= np.exp2(exp)
    return a


def match_grayscale_exposure(ref, a):
    assert ref.shape == a.shape, f"Shapes do not match: {ref.shape} != {a.shape}"
    assert ref.ndim == 3, f"Only 3D BxMN tensors are supported"

    f_shape = ref.shape[1] * ref.shape[2]
    ref = ref - ref.min()
    a = a - a.min()
    ref_mean = np.reshape(ref, (-1, f_shape)).mean(axis=-1)
    a_mean = np.reshape(a, (-1, f_shape)).mean(axis=-1)

    ratio = np.nan_to_num(
        ref_mean/a_mean,
        nan=1., posinf=1., neginf=1.
    )
    ratio = np.repeat(ratio, f_shape).reshape(a.shape)

    # Apply Ratio
    a = (a * ratio)
    return np.nan_to_num(a, nan=0, posinf=0, neginf=0)


def match_exposure(ref, a):
    assert ref.shape == a.shape, f"Shapes do not match: {ref.shape} != {a.shape}"
    assert ref.ndim == 3 or ref.ndim==2, f"Only 2D and 3D tensors are supported"

    a = a - a.min()
    ref = ref - ref.min()

    if ref.ndim == 3:
        a_mean = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1].mean()
        ref_mean = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2XYZ)[:,:,1].mean()

        # Get final Ratio
        ratio = np.nan_to_num(
            ref_mean / a_mean,
            nan=float('nan'),
            posinf=float('nan'),
            neginf=float('nan')
        )
        ratio = np.nan_to_num(ratio, nan=1., posinf=1., neginf=1.)
    elif ref.ndim == 2:
        a_mean = a.mean()
        ref_mean = ref.mean()

        # Get final Ratio
        ratio = np.nan_to_num(
            ref_mean / a_mean,
            nan=float('nan'),
            posinf=float('nan'),
            neginf=float('nan')
        )
        ratio = np.nan_to_num(ratio, nan=1., posinf=1., neginf=1.)

    # Apply Ratio
    a = (a * ratio)
    return np.nan_to_num(a, nan=0., posinf=0., neginf=0.)


def match_exposure_torch(ref, a, global_mean=True):
    assert ref.shape == a.shape, f"Shapes do not match: {ref.shape} != {a.shape}"
    a = a - a.min()
    gray_a = F.rgb_to_grayscale(a)
    ref = ref - ref.min()
    gray_ref = F.rgb_to_grayscale(ref)

    if ref.ndim == 4:
        if global_mean:
            _gray_ref = gray_ref.flatten(start_dim=1)
            _gray_a = gray_a.flatten(start_dim=1)
            shape_flat = _gray_a.shape[1]

            ratio = (torch.nanmean(_gray_ref, axis=1, keepdim=True) / torch.nanmean(_gray_a, axis=1, keepdim=True))
            ratio = torch.nan_to_num(ratio, nan=1., posinf=1., neginf=1.)

            # Get final Ratio
            ratio = ratio.unsqueeze(1).expand(-1, 3, shape_flat)
            ratio = ratio.reshape(a.shape)

        else: # Local mean
            ratio = (gray_ref / gray_a).flatten(start_dim=1)
            shape_flat = ratio.shape[1]
            ratio = torch.nan_to_num(
                ratio,
                nan=float('nan'), posinf=float('nan'), neginf=float('nan')
            )

            # Get final Ratio
            ratio = torch.nanmean(ratio, axis=1, keepdim=True)
            ratio = ratio.unsqueeze(1).expand(-1, 3, shape_flat)
            ratio = ratio.reshape(a.shape)

    elif ref.ndim == 3:
        raise NotImplementedError("3D tensor not implemented")
        ratio = (ref / a).flatten()
        ratio = torch.nan_to_num(ratio, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))

        # Get final Ratio
        ratio = torch.nanmean(ratio)
    else:
        raise ValueError("Only 3D and 4D tensors are supported")

    # Apply Ratio
    a = (a * ratio)
    return torch.nan_to_num(a, nan=0, posinf=0, neginf=0)

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