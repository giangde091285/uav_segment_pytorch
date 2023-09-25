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


# def label2RGB(mask_path, vis_path):

#     new_mask = np.array(Image.open(mask_path)).astype(np.uint8)

#     cm = np.array(list(map.values())).astype(np.uint8)

#     color_img = cm[new_mask]

#     color_img = Image.fromarray(np.uint8(color_img))

#     color_img.save(vis_path)


def labeltoRGB(input:np.array, vis_path):


    cm = np.array(list(map.values())).astype(np.uint8)

    color_img = cm[input]

    color_img = Image.fromarray(np.uint8(color_img))

    color_img.save(vis_path)


# if __name__ == '__main__':

#     mask_root = r'E:\studio\研究生期间\研0\1.Data\whu_res\output-23-9-22'
#     mask_list = os.listdir(mask_root)
#     vis_root = r'E:\studio\研究生期间\研0\1.Data\whu_res\vis\pre'

#     for i in range(len(mask_list)):
#         mask_path = os.path.join(mask_root, mask_list[i])
#         mask_name = 'vis' + mask_list[i]
#         vis_path = os.path.join(vis_root, mask_name)
#         label2RGB(mask_path, vis_path)