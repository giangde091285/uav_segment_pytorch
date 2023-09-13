import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.load_data import * 
from utils.spilt_set import * 

from model.seg_model import *
from utils.acc_func import *

from opt import *
import os

def predict(model, img_path, mask_path, output_dir):
    
    model.eval()
    # read img
    img = Image.open(img_path)
    gt_mask = Image.open(mask_path)
   
    # pre process
    img_tensor = transforms.ToTensor()(img.copy()).unsqueeze(0)  # PIL [0,255] (H W C) -> tensor [0,1] (B=1 C H W)
    gt_mask_tensor = torch.as_tensor(np.array(gt_mask.copy()),dtype=torch.long) #(H W)

    # get unique values of gt
    unique_value = torch.unique(gt_mask_tensor)
  
    with torch.no_grad():
        # predict
        pre_mask_tensor = model(img_tensor).cpu() #（B=1 C H W）

        # (B=1 C H W) -> (C H W) -> (H W)
        pre_mask_tensor = pre_mask_tensor.squeeze(0)
        pre_mask_tensor = torch.argmax(pre_mask_tensor, dim=0)

        pre_mask_array = np.asarray(pre_mask_tensor, dtype=np.uint8)
        gt_mask_array = np.asarray(gt_mask_tensor, dtype=np.uint8)
   
        # cal acc
        iou_Mean, iou = mIOU(gt= gt_mask_array, 
                             pre= pre_mask_array,
                             class_num=8,
                             unique_value=unique_value) 
    
        # save pred_mask
        res=Image.fromarray(pre_mask_array)
        res.save(os.path.join(output_dir, "pre.png"))

    return iou_Mean, iou
    
def predict_testset(args):
    model.eval() 
    test_dataloader = DataLoader(CusDataset("val"), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    miou = 0
    miou_list = np.zeros(args.classes, dtype=np.float64)
    class_unique = np.zeros(args.classes, dtype=np.int64)
    for batch in tqdm(test_dataloader, unit='batch', total=len(test_dataloader)):
        images, masks = batch['img'], batch['mask']
        # copy to gpu
        images = images.to(device=device, dtype=torch.float32,
                            memory_format = torch.channels_last)  
        masks = masks.to(device = device, dtype = torch.long)
        unique_value = torch.unique(masks)
                
        with torch.no_grad():        
            # predict
            pre_mask_tensor = model(images)
            pre_mask_tensor = pre_mask_tensor.squeeze(0)
            pre_mask_tensor = torch.argmax(pre_mask_tensor, dim=0)

            pre_mask_array = np.asarray(pre_mask_tensor, dtype=np.uint8)
            gt_mask_array = np.asarray(masks, dtype=np.uint8)
   
            # cal acc
            iou_Mean, iou = mIOU(gt= gt_mask_array, 
                                pre= pre_mask_array,
                                class_num=8,
                                unique_value=unique_value) 
            
            for i in range(0, len(iou)):
                miou_list[i] += iou[i]
                if i in unique_value:
                    class_unique[i] += 1

            miou += iou_Mean

    miou /= len(test_dataloader)
    miou_list = np.divide(miou_list, (class_unique+1e-8))

    return miou, miou_list, class_unique

         

if __name__ == '__main__':

    ##### get args #####
    args = get_args()

    torch.cuda.empty_cache()

    # img_path = './data_process/dataset/test/img/1366_5.png'
    # mask_path = './data_process/dataset/test/mask/1366_5.png'

    # load model
    model = U_net(in_channels=3, classes=8)
    # model = SegModel(in_channels=3, classes=8)
    device = torch.device('cpu')
    model.to(device=device)
    
    # load ckpt
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # # # predict
    # # iou_mean, iou = predict(model, 
    # #                         img_path, 
    # #                         mask_path, 
    # #                         output_dir=args.output_dir)

    # print(iou_mean)
    # print(iou)
    
    miou, miou_list, cu = predict_testset(args)
    print(miou)
    print(miou_list)
    print(cu)

