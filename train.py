import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from utils.load_data import * 
from utils.spilt_set import * 
from utils.optimizer import *
from utils.acc_func import *
from model.seg_model import *
import wandb
from tqdm import tqdm
from opt import *
from torchsummary import summary


def train_model(model, device, epoch, batch_size, lr, args):
    
    ##### dataloader #####
    train_num = len(CusDataset("train"))
    train_dataloader = DataLoader(CusDataset("train"), 
                                batch_size, shuffle=True, drop_last=False, pin_memory=True)
    val_dataloader = DataLoader(CusDataset("val"), 
                                batch_size, shuffle=False, drop_last=False, pin_memory=True)
    
    ##### wandb(log) #####
    # init : Start a new run to track and log to W&B.
    exper = wandb.init(project='UNet_train', resume='allow', anonymous='must') 
    # config ：track hyperparameters and metadata.
    exper.config.update(
        dict(model = model, 
             device = device, 
             epoch = epoch, 
             batch_size = batch_size, 
             lr = lr)
    )     
    
    ##### optimizer #####
    optimizer = OptimizerTool(name='Adam', model=model, args=args)
    ##### learning rate #####
    scheduler = lrSchedulerTool(name='Step', args=args, opt=optimizer)
    ##### loss function #####
    loss_func = nn.CrossEntropyLoss(reduce='mean') # nn.CrossEntropyLoss include softmax！
    ##### amp #####
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    ##### train and val loops #####
    global_step = 0  # total step
    accumulation_num = 1
    for ep in range(1, epoch+1):
        model.train()  # train mode
        epoch_loss = 0  # epoch loss
        with tqdm(total = len(train_dataloader), desc=f'Epoch {ep}/{epoch}', unit='batch') as pbar:  
            i = 1
            for batch in train_dataloader:
                # read this batch : img(N C H W)  mask(N H W)
                images, masks = batch['img'], batch['mask']
                # copy to gpu
                images = images.to(device=device, dtype=torch.float32, # pytorch default float datatype is float32 !!
                                   memory_format = torch.channels_last)  # in gpu memory , set torch format to (N H W C). https://zhuanlan.zhihu.com/p/494620090
                masks = masks.to(device = device, dtype = torch.long) # pytorch default int datatype is int64(long) !!
                
                ##### train #####
                # use amp
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True): 
                    # predict
                    pre = model(images) # out: ( N C H W)
                    # current batch loss
                    # loss = loss_func(pre, masks) # softmax + CEloss
                    loss = Dice_loss(masks, pre, args.classes)
                    loss = loss.to(device=device, dtype=torch.float32)

                scaler.scale(loss).backward() # backward loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # clip grad
                if (i % accumulation_num == 0) : 
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.step() # update params: w and b
                    optimizer.zero_grad(set_to_none=True)  # Sets the gradients of all optimized torch.Tensors to zero.
                
                i += 1
                global_step += 1
                epoch_loss += loss.item() #  use .item() to save memory : https://www.codedi.net/article/11329/

                pbar.update(1) # done a batch's train 
                pbar.set_postfix(**{ "train_loss_step":loss.item() }) # kargs(dict)

                exper.log({
                    "epoch":ep,
                    "global_step": global_step,
                    "train_loss_step" : loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                })

        ##### val #####
        model.eval() # eval mode
        v_loss_epoch = 0
        with torch.no_grad():
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True): 
                for batch in tqdm( val_dataloader, unit='batch', total=len(val_dataloader)):
                    v_img , v_mask = batch["img"], batch["mask"]

                    v_img = v_img.to(device=device, dtype=torch.float32, memory_format = torch.channels_last) 
                    v_mask = v_mask.to(device = device, dtype = torch.long)

                    v_res = model(v_img)

                    v_loss = loss_func(v_res,v_mask) 
                    v_loss_epoch += v_loss.item()
                    
        model.train() # train mode
        v_loss_epoch = v_loss_epoch / len(val_dataloader)
        scheduler.step() # update superparams

        exper.log({
                    "val_loss_epoch": v_loss_epoch,
                    "train_loss_epoch": epoch_loss/len(train_dataloader)
                })
                
        ##### save checkpoint(after a epoch) #####
        checkpoint_path = './checkpoint'
        if os.path.isdir(checkpoint_path) == False:
            os.mkdir(checkpoint_path)
        else:
            pass
        state_dict = model.state_dict()
        torch.save(state_dict, str( os.path.join(checkpoint_path , 'checkpoint_epoch_{}.pth'.format(ep))))


if __name__ == '__main__':
    
    ##### get args #####
    args = get_args()

    ##### set device #####
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(" train use gpu : {}".format(device))
    else:
        device = torch.device('cpu')

    ##### spilt_set ##### 
    if args.if_spilt :
        spilttool = CropSpiltTool()
        spilttool.spilt_set(train_scale=args.train_scale, 
                            val_scale=args.val_scale, 
                            if_crop=args.if_crop, 
                            tar_size=args.target_size, 
                            split_total_num=args.target_num)

    ##### model #####
    model = ResUNet(args.in_channels, args.classes)
    # summary(model=model, input_size=256, batch_size=args.batch_size)
    # model = SegModel(args.in_channels, args.classes)
    model = model.to(device=device,  # to gpu
                     memory_format=torch.channels_last)  # set model weight format -> torch.cuda.FloatTensor
        
    ##### train #####
    train_model( model=model, 
                 device=device,
                 epoch=args.epochs,
                 batch_size=args.batch_size,
                 lr=args.lr,
                 args=args)

