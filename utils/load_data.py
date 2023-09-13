import torch
from torch.utils import data
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms


class CusDataset(data.Dataset):
    """
    structure : dataset
                  -- train
                    -- img
                    -- mask
                  -- val
                    -- img
                    -- mask
                  -- test
                    -- img
                    -- mask
    """
    def __init__(self, set_type='train'):
        
        # get the filename list of each set
        assert set_type in ['train', 'val', 'test'], ' undefined dataset! '
        self.root_path = "./data_process/dataset"
        self.set_type = set_type
        self.img_path = os.path.join(self.root_path, self.set_type, 'img')
        self.img_list = os.listdir(self.img_path)
        self.mask_path = os.path.join(self.root_path, self.set_type, 'mask')
        self.mask_list = os.listdir(self.mask_path)
        
    def __getitem__(self, index):
        
        img_name = self.img_list[index]
        mask_name = self.mask_list[index]

        # read image and mask (PIL)
        img = Image.open(os.path.join(self.img_path, img_name))
        mask = Image.open(os.path.join(self.mask_path, mask_name))

        ### transform ###
        # for both
        both_trans = transforms.Compose([ transforms.RandomHorizontalFlip(p=0.2)])

        # for PIL image
        img_trans = transforms.Compose([ # transforms.RandomApply(transforms.GaussianBlur(kernel_size=3), p=0.1), # PIL -> blur
                                         transforms.ToTensor(),  # PIL [0,255] (H W C) -> tensor [0,1] (C H W)
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # tensor -> zero-norm
                                         ])
        
        img_pr = both_trans(img.copy())
        mask_pr = both_trans(mask.copy())

        img_tensor = img_trans(img_pr) # float32
        mask_tensor = torch.as_tensor(np.array(mask_pr),dtype=torch.long) # long(int64)
      
        return {'img' : img_tensor, 'mask' : mask_tensor}

    def __len__(self):
        return len(self.img_list)

