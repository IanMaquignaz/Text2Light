
# Standard library
import os
import cv2
import numpy as np

# Custom
from envmap import EnvironmentMap

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def load_image(path):
    # load the image
    img = cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    e = e.convertTo('latlong', ref.shape[0])
    e.data[-16:,:,:]=0
    return e.data


def exposure(a, exp):
    a*= np.exp2(exp)
    return a

def match_exposure(ref, a):
    # Fix exposure
    ref_g = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2GRAY)
    a_g = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2GRAY)
    ratio = ref_g/a_g
    ratio = np.nan_to_num(ratio, posinf=1., neginf=0.)
    ratio = np.nanmean(ratio)
    # print('ratio=', ratio)
    a = a*ratio
    return np.clip(a,0,1)


def L1(a,b):
    return np.abs(a-b).mean()

def SIL(a,b):
    left = np.power(a-b,2).mean()
    right = np.power((a-b).mean(), 2)
    return left - right