import numpy as np
import torch
from torch import *
from torch import nn
import torch.nn.functional as F

"""
    loss function

    * dice loss 
    * focal loss 

"""

class DiceLoss(nn.Module):
    def __init__(self, class_num:int, if_ignore=True) -> None:
        super(DiceLoss, self).__init__()
        self.class_num = class_num 
        self.if_ignore = if_ignore

    def forward(self, input:torch.Tensor, target:torch.Tensor, ignore_idx:int=-1):
        # input:(N C H W) -> (N H W C) -> (NHW=n C) 
        input = input.permute(0, 2, 3, 1).flatten(0,2)
        input = F.softmax(input, 1)  # softmax
        # target:(N H W) -> (NHW) 
        target = target.flatten(0,2)

        # 忽略nodata处
        if self.if_ignore:
            # 获取target值不为-1的索引
            mask = ~ target.eq(ignore_idx)  # 逐元素对比，如果相同返回true，否则 false
            # 取值不为-1的元素，组成新target
            target = target[mask]   # (n')
            # 取target值不为-1的元素，组成新input
            input = input[mask,:]   # (n' C)
        
         # target: (NHW) -> (NHW C)
        target = F.one_hot(target, self.class_num)

        # intersect = （input * target).sum()
        intersect = (input * target).sum(dim=0)  #(c)
        a = input.sum(dim=0) #(c)
        b = target.sum(dim=0) #(c)

        dice_loss = 1 - 2 * intersect/(a + b) #(c)
        return dice_loss.mean() 
    

class FocalLoss(nn.Module):
    def __init__(self, alpha: list, gamma=2, class_num: int=None, if_ignore=True) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = torch.Tensor(alpha).unsqueeze(dim=1)
        self.gamma = gamma
        self.class_num = class_num
        self.if_ignore = if_ignore

    def forward(self, input: torch.Tensor, target: torch.Tensor, avg: bool = True, ignore_idx:int=-1):
        # input: (N C H W) -> (N H W C) -> (N*H*W=n C)
        input = input.permute(0, 2, 3, 1)
        input = input.flatten(0, 2)
        input = F.softmax(input, 1)  # (n C)

        # target: (N H W) -> (N*H*W) 
        target = target.flatten(0, 2) # (n)
       
        # 忽略nodata处
        if self.if_ignore:
            # 获取target值不为-1的索引
            mask = ~ target.eq(ignore_idx)  # 逐元素对比，如果相同返回true，否则 false
            # 取值不为-1的元素，组成新target
            target = target[mask]   # (n')
            # 取target值不为-1的元素，组成新input
            input = input[mask,:]   # (n' C)
        
        # target: (N*H*W) -> (N*H*W=n C)
        target = F.one_hot(target, self.class_num) #(n C)

        # 把input中，target=0处对应位置的值换为1, 便于计算log(p)
        P_1 = torch.masked_fill(input, ~target.bool(), 1)  # (n C)
        # 把input中，target=0处对应位置的值换为0, 便于计算p
        P_0 = torch.masked_fill(input, ~target.bool(), 0)  # (n C)

        # (1) - alpha * log(p)
        logP = torch.log(P_1)  # (n C)
        alpha = torch.as_tensor(self.alpha, device=torch.device('cuda:0'))
        weight_logP = - torch.mm(logP, alpha).squeeze()  # (n)

        # (2)  (1-p)**gamma
        # 沿列方向求和，求出每一个像元属于所有类的loss之和
        p = torch.sum(P_0, dim=1)  # (n)
        fac = torch.pow((1 - p), self.gamma)  # (n)

        loss = torch.mul(weight_logP, fac)  # (n)

        if avg:
            loss = loss.mean()
        return loss
 

        



