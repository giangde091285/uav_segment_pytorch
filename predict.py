import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.dataset_load import * 
from utils.dataset_spilt import * 

from model.seg_model import *
from utils.precision_eval import *

from opt import *
import os

def PredictSet(model, device, args):
    model.eval() 
    test_dataloader = DataLoader(CusDataset("test"), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    iou_list = np.zeros(shape=(args.classes),dtype=np.float16)
    acc_list = np.zeros(shape=(args.classes),dtype=np.float16)
    i = 0
    af = AccFunc()
    for batch in tqdm(test_dataloader, unit='batch', total=len(test_dataloader)):
        images, masks = batch['img'], batch['mask']
        # copy to gpu
        images = images.to(device=device, dtype=torch.float32, memory_format = torch.channels_last)  
        masks = masks.to(device = device, dtype = torch.long)
        with torch.no_grad():        
            # predict
            pre_mask_tensor = model(images)
            mask_true = F.one_hot(masks, args.classes).permute(0, 3, 1, 2).float()  # (1,C,H,W)
            mask_pred = F.one_hot(pre_mask_tensor.argmax(dim=1), args.classes).permute(0, 3, 1, 2).float()  #(1,C,H,W)
            # iou_mean= multiclass_iou(mask_pred, mask_true, reduce_batch_first=False)

            mask_true = torch.argmax(mask_true, dim=1)
            mask_pred = torch.argmax(mask_pred, dim=1)
            iou_list += af.IOU(mask_true, mask_pred, args.classes)
            acc_list += af.OA(mask_true, mask_pred, args.classes)
            
            if args.ifsave :
                # (1,c,h,w) -> (1,h,w) -> (h,w)
                pre_mask_array = np.asarray(pre_mask_tensor.cpu().argmax(dim=1).squeeze(dim=0), dtype=np.uint8)
                pre_mask_pil = Image.fromarray(pre_mask_array)
                
                if os.path.isdir(args.output_dir):
                    pass
                else:
                    os.mkdir(args.output_dir)
                mask_list = os.listdir(r"data_process/dataset/test/mask")
                pre_mask_pil.save(os.path.join(args.output_dir, ("pre_" + mask_list[i])))

        i += 1

    iou_list /= len(test_dataloader)
    acc_list /= len(test_dataloader)
    class_miou = np.sum(iou_list)/args.classes
    class_macc = np.sum(acc_list)/args.classes
    return [iou_list, class_miou, acc_list, class_macc]

         

if __name__ == '__main__':

    ##### get args #####
    args = get_args()
   
    ##### set device #####
    if args.device != 'cpu' and torch.cuda.is_available():
           device = torch.device(args.device)
           print(" train use gpu : {}".format(device))
    else:
        device = torch.device('cpu')
        print(" train use cpu ")
    
    ##### model #####
    if args.model == 'U_net':
        model = U_net(args.in_channels, args.classes)
    elif args.model == 'ResUNet':
        model = ResUNet(args.in_channels, args.classes, device)
    else:
       raise Exception('no such model!')
    
    ##### load ckpt #####
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device=device)

    ##### predict #####
    [iou_list, class_miou, acc_list, class_macc] = PredictSet(model, device, args)
    print("iou={0}\n"
          +"miou={1}\n"
          +"acc={2}\n"
          +"macc={3}".format(iou_list, class_miou, acc_list, class_macc))


