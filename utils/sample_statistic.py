import numpy as np
from torch.utils.data import DataLoader

from utils.dataset_load import * 
from utils.dataset_spilt import * 

from model.seg_model import *
from utils.precision_eval import *
from opt import *


args = get_args()
dataloader = DataLoader(CusDataset("train"), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

class_pixel_num = np.zeros(args.classes, dtype=np.int64)

for batch in tqdm(dataloader, unit='batch', total=len(dataloader)):
    masks = batch['mask']  
    masks = np.asarray(masks, dtype=np.int64)
    for i in range(0, args.classes, 1):
        class_pixel_num[i] += np.sum(masks == i)

print(class_pixel_num)

ratio = class_pixel_num/np.sum(class_pixel_num)

r = 1/ratio

print(ratio)
print(r)