import os
from shutil import copy2
from PIL import Image
import random
import glob
from tqdm import tqdm
import torchvision as tv

"""
    dataset_split

    * SplitDataset (split into train / val / test set)
    * CropDataset  (only for train / val set)
    * DataEnhance  (only for train set)

"""
class CropSplitTool():
    def __init__(self) -> None:
        # read images and masks from original path
        self.ori_img_path = glob.glob(r"./data_process/ori_data/img/*")
        self.ori_mask_path = glob.glob(r"./data_process/ori_data/mask/*")
        self.ori_img_name = os.listdir(r"./data_process/ori_data/img/")
        self.ori_mask_name = os.listdir(r"./data_process/ori_data/mask/")

        # original w and h of images
        self.ori_w = (Image.open(self.ori_img_path[0])).size[0]
        self.ori_h = (Image.open(self.ori_img_path[0])).size[1]
        self.ori_len = len(self.ori_img_path)

    def create_folder(self, main_folder, sub_folder):
        # create folder and its sub-folders in specific path.
        if os.path.isdir(main_folder):
            pass
        else:
            os.mkdir(main_folder)

        for item in sub_folder:
            sub_path = os.path.join(main_folder, item)
            if os.path.isdir(sub_path):
                pass
            else:
                os.mkdir(sub_path)

    def SplitDataset(self, train_scale=0.8, val_scale=0.1):
        # divide data into train/val/test set.

        # create dataset structure
        sub_folder = ['train', 'val', 'test']
        sub_folder2 = ['img', 'mask']
        main_folder = "./data_process/dataset"
        self.create_folder(main_folder, sub_folder)
        for i in sub_folder:
            self.create_folder(os.path.join(main_folder, i), sub_folder2)

        root = r"./data_process/ori_data/"
        img_list = self.ori_img_name
        mask_list = self.ori_mask_name

        # stop flag
        train_stop = int(len(img_list) * train_scale)
        val_stop = int(train_stop + len(img_list) * val_scale)

        # throw the data out of order
        new_list = list(range(len(img_list)))
        random.shuffle(new_list)

        train_num = 0
        val_num = 0
        test_num = 0
        cal = 0

        print("START -- Split Dataset \n"
              " train : val = {0} ： {1}"
            .format(train_scale, val_scale))

        with tqdm(total=len(new_list)) as pbar:
            for i in new_list:
                if cal <= train_stop:
                    copy2(os.path.join(root, "img/") + img_list[i],
                        os.path.join(main_folder, "train/img/") + img_list[i])
                    copy2(os.path.join(root, "mask/") + mask_list[i],
                        os.path.join(main_folder, "train/mask/") + mask_list[i])
                    train_num += 1
                elif cal <= val_stop:
                    copy2(os.path.join(root, "img/") + img_list[i],
                        os.path.join(main_folder, "val/img/") + img_list[i])
                    copy2(os.path.join(root, "mask/") + mask_list[i],
                        os.path.join(main_folder, "val/mask/") + mask_list[i])
                    val_num += 1
                else:
                    copy2(os.path.join(root, "img/") + img_list[i],
                        os.path.join(main_folder, "test/img/")+img_list[i])
                    copy2(os.path.join(root, "mask/") + mask_list[i],
                        os.path.join(main_folder, "test/mask/") + mask_list[i])
                    test_num += 1
                cal += 1
                pbar.update(1)

        print("DONE -- Split Dataset \n"
            "train：{0}\n"
            "val：{1}\n"
            "test：{2}".format(train_num, val_num, test_num))
        
    def RandomCropMany(self, 
                       root: str, 
                       set: str = 'train', 
                       tar_size: int = 256, 
                       tar_num: int = 4):
   
        img_dir = os.path.join(root, set, 'img/')
        mask_dir = os.path.join(root, set, 'mask/')

        ori_img_path = glob.glob(os.path.join(root, set, 'img/*'))
        ori_mask_path = glob.glob(os.path.join(root, set, 'mask/*'))

        ori_img_name = os.listdir(img_dir)
        ori_mask_name = os.listdir(mask_dir)

        ori_len = len(ori_img_name)

        i = 0
        print(
            " target set = {0}"
            "\n target size={1}*{1}"
            "\n original size={2}*{3}"
            "\n target num={4}"
            "\n original num={5}"
            .format(set, tar_size, self.ori_w, self.ori_h, tar_num*ori_len, ori_len))

        # randomly sample
        with tqdm(total=ori_len) as pbar:
            while i < ori_len:
                img_i = Image.open(ori_img_path[i])
                mask_i = Image.open(ori_mask_path[i])
                j = 1
                while j <= tar_num:
                    # the coordinate of the left-top point is from 0 to ( ori_img_w - tar_size )
                    crop_box_lt_x = random.randint(0, (self.ori_w-tar_size))
                    crop_box_lt_y = random.randint(0, (self.ori_h-tar_size))
                    crop_box = (crop_box_lt_x, crop_box_lt_y, (crop_box_lt_x+tar_size), (crop_box_lt_y+tar_size))

                    new_img = (img_i.copy()).crop(crop_box)
                    new_mask = (mask_i.copy()).crop(crop_box)

                    new_img.save(img_dir + ori_img_name[i].split(".")[0] + "_" + str(j) + "."
                                + ori_img_name[i].split(".")[1])
                    new_mask.save(mask_dir + ori_mask_name[i].split(".")[0] + "_" + str(j) + "."
                                + ori_mask_name[i].split(".")[1])
                    j += 1
                i += 1
                pbar.update(1)
        
        # delete original imgs
        print("delete original imgs")
        for files in ori_img_name:
            os.remove(os.path.join(img_dir,files))
        for files in ori_mask_name:
            os.remove(os.path.join(mask_dir,files))

    def CropDataset(self, tar_size: int = None, tar_num: int = None):
        
        dir = r'./data_process/dataset/'
        print(" START -- Crop train/val Images ")
        self.RandomCropMany(dir,'train',tar_size, tar_num)
        self.RandomCropMany(dir,'val',tar_size, tar_num)
        print(" DONE -- Crop train/val Images ")
    

    # only for train set
    def DataEnhance(self, factor:float):

        # original path of train set
        img_dir = r"./data_process/dataset/train/img/"
        mask_dir = r"./data_process/dataset/train/mask/"

        img_list = os.listdir(img_dir)
        mask_list = os.listdir(mask_dir)
        
        img_path = glob.glob(img_dir+"*")
        mask_path = glob.glob(mask_dir+"*")

        # according to @factor，randomly choose imgs that apply data enhance，record their index in img_list
        idx_num = int(len(img_list) * factor)
        idx_list = random.sample(range(0,(len(img_list)-1)), idx_num)

        # the path of imgs and masks that apply data enhance
        choice_img_list = []
        choice_mask_list = []
        choice_img_path = []
        choice_mask_path = []
        for i in range(len(img_list)):
            if i in idx_list:
                choice_img_list.append(img_list[i])
                choice_mask_list.append(mask_list[i])
                choice_img_path.append(img_path[i])
                choice_mask_path.append(mask_path[i])

        # enhance methods
        transpose_img = tv.transforms.Compose([tv.transforms.GaussianBlur(kernel_size=3)])
        transpose_both = tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(p=1)])
        
        print("******** enhance train set *********"
              "\n factor: {0}".format(factor))
        with tqdm(total=idx_num) as pbar:
            for i in range(idx_num):
                # read original PIL Images
                img = Image.open(choice_img_path[i])
                mask = Image.open(choice_mask_path[i])

                img = transpose_img(img)

                img = transpose_both(img)
                mask = transpose_both(mask)

                # save
                img.save(img_dir + choice_img_list[i].split(".")[0] + "_trans." + choice_img_list[i].split(".")[1])
                mask.save(mask_dir + choice_mask_list[i].split(".")[0] + "_trans." + choice_mask_list[i].split(".")[1])

                pbar.update(1)





            
