import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils.dataset_load import * 
from utils.dataset_split import * 
from model.seg_model import *
from utils.metric import *
from utils.labelRGB import *
from opt import *
import os


"""
    predict

    * test dataset

"""

def PredictSet(model, device, args):
    model.eval() 
    if args.dataset_name == 'whu':
        test_dataloader = DataLoader(WHU("test"), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    elif args.dataset_name == 'LoveDA':
        test_dataloader = DataLoader(LoveDA("test"), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    else:
        raise Exception('no such dataset!')
    
    iou_list = np.zeros(shape=(args.classes),dtype=np.float16)
    acc_list = np.zeros(shape=(args.classes),dtype=np.float16)
    mt = Metric(if_ignore=True, ignore_idx=-1)
    i = 0
    for batch in tqdm(test_dataloader, unit='batch', total=len(test_dataloader)):
        images, masks = batch['img'], batch['mask']
        # copy to gpu
        images = images.to(device=device, dtype=torch.float32, memory_format = torch.channels_last)  
        masks = masks.to(device = device, dtype = torch.long) # (N H W)
        with torch.no_grad():        
            pre_mask_tensor = model(images) # (N C H W)

            a = mt.IOU(pre_mask_tensor, masks, args.classes)
            b = mt.Acc(pre_mask_tensor, masks, args.classes)
            iou_list += np.asarray(a,dtype=np.float16)
            acc_list += np.asarray(b,dtype=np.float16)
            
            if args.ifsave :
                # (1,c,h,w) -> (1,h,w) -> (h,w)
                pre_mask_tensor = pre_mask_tensor.argmax(dim=1).squeeze(dim=0)

                if args.dataset_name == "LoveDA":
                    masks = masks.squeeze(0) #(h,w)
                    mask = torch.where(masks==-1, 1, 0).bool()
                    pre_mask_tensor = torch.masked_fill(pre_mask_tensor, mask, -1) + 1  #[-1,6]->[0,7]
                
                pre_mask_array = np.asarray(pre_mask_tensor.cpu(), dtype=np.uint8)
                pre_mask_pil = Image.fromarray(pre_mask_array)

                if os.path.isdir(args.output_dir):
                    pass
                else:
                    os.mkdir(args.output_dir)
                mask_list = os.listdir(r"data_process/dataset/test/mask")

                if args.if_labelRGB:
                    type = 'pre_vis'
                    out_path = os.path.join(args.output_dir, (type + mask_list[i]))
                    labeltoRGB(pre_mask_array, out_path)
                else:
                    type = 'pre_gray'
                    out_path = os.path.join(args.output_dir, (type + mask_list[i]))
                    pre_mask_pil.save(out_path)

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
           print(" use gpu : {}".format(device))
    else:
        device = torch.device('cpu')
        print(" use cpu ")
    
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
    # [iou_list, class_miou] = PredictSet(model, device, args)
    # print("iou={0}\n"
    #       "miou={1}\n".format(iou_list, class_miou))
    print("iou={0}\n"
          "miou={1}\n"
          "acc={2}\n"
          "macc={3}".format(iou_list, class_miou, acc_list, class_macc))


