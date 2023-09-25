import torch
from torch.utils import data
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms


"""
    dataset load

    * WHU building dataset
    * LoveDA dataset

"""

class WHU(data.Dataset):
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
        #both_trans = transforms.Compose([ 
         #   transforms.RandomHorizontalFlip(p=0.2)])

        # for PIL image
        img_trans = transforms.Compose([ # transforms.RandomApply(transforms.GaussianBlur(kernel_size=3), p=0.1), # PIL -> blur
                                         transforms.ToTensor(),  # PIL [0,255] (H W C) -> tensor [0,1] (C H W)
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # tensor -> zero-norm
                                         ])
        
        mask_array = np.asarray(mask,dtype=np.uint8)
        mask_single = np.zeros(shape=(mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        for i in range(mask_array.shape[0]):
            for j in range(mask_array.shape[1]):
                if mask_array[i,j,0] == 255:
                   mask_single[i,j] = 1

        mask_single = Image.fromarray(mask_single)
        img_tensor = img_trans(img.copy()) # float32 (C H W)
        mask_tensor = torch.as_tensor(np.array(mask_single),dtype=torch.long) # long(int64) (H W)
        return {'img' : img_tensor, 'mask' : mask_tensor}

    def __len__(self):
        return len(self.img_list)


class LoveDA(data.Dataset):
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
        #both_trans = transforms.Compose([ 
         #   transforms.RandomHorizontalFlip(p=0.2)])

        # for PIL image
        img_trans = transforms.Compose([ # transforms.RandomApply(transforms.GaussianBlur(kernel_size=3), p=0.1), # PIL -> blur
                                         transforms.ToTensor(),  # PIL [0,255] (H W C) -> tensor [0,1] (C H W)
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # tensor -> zero-norm
                                         ])
        
        img_tensor = img_trans(img.copy()) # float32 (C H W)
        mask_tensor = torch.as_tensor(np.array(mask),dtype=torch.long) # long(int64) (H W)

        # for LoveDA : (ignore nodata)
        # 1.修改真实mask的值 [0,7] -> [-1,6]
        mask_tensor = mask_tensor - 1
        # 2.将img中对应mask的值为-1的部分，设置为0
        zz_mask = torch.where(mask_tensor == -1, 0, 1).bool() 
        img_tensor = torch.masked_fill(input=img_tensor, mask=~zz_mask, value=0)   
      
        return {'img' : img_tensor, 'mask' : mask_tensor}

    def __len__(self):
        return len(self.img_list)
