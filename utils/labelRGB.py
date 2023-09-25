from PIL import Image
import numpy as np
import os
from opt import *

args = get_args()
if args.dataset_name == 'LoveDA':
    map = dict(
        IGNORE=(0, 0, 0),
        Background=(255, 255, 255),
        Building=(255, 0, 0),
        Road=(255, 255, 0),
        Water=(0, 0, 255),
        Barren=(159, 129, 183),
        Forest=(0, 255, 0),
        Agricultural=(255, 195, 128),
    )
elif args.dataset_name == 'whu':
     map = dict(
        Background=(0, 0, 0),
        Building=(255, 255, 255)
    )
else:
    raise Exception('no such dataset')


def labeltoRGB(input:np.array, vis_path):
    cm = np.array(list(map.values())).astype(np.uint8)
    color_img = cm[input]
    color_img = Image.fromarray(np.uint8(color_img))
    color_img.save(vis_path)


