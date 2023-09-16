import numpy as np
import torch
from torch import *
from torch import nn

"""
  train/val loss:
    input  -- pre : float tensor (N C H W)   C : softmax([0, 1])
           -- gt  : float tensor (N C H W)   C : one-hot(0 or 1)
    output -- mean_value : tensor (1)
           -- list  : tensor (C)

  test acc:
    input  -- pre : int tensor (N H W)   element value: class index
           -- gt  : int tensor (N H W)   element value: class index
    output -- mean_value : tensor (1)
           -- list  : tensor (C)
""" 

class AccFunc(nn.Module):
    def __init__(self) -> None:
        super(AccFunc, self).__init__()

    def cfm(self, gt, pre, class_num):
        # convert (N,H,W) to (pixel)
        gt = torch.flatten(gt, start_dim=0, end_dim=-1)
        pre = torch.flatten(pre, start_dim=0, end_dim=-1)
        # get category
        categroy =  class_num * gt + pre
        categroy = np.asarray(categroy.cpu(), dtype=np.uint8) 
        # get confusion mat  -- row : gt , col : pre
        confusion_mat = np.zeros((class_num*class_num), dtype=np.uint8)

        for i in np.linspace(0, confusion_mat.size, endpoint=False, num=confusion_mat.size, dtype=np.int64):
            confusion_mat[i] = np.sum(categroy == i)

        confusion_mat = confusion_mat.reshape(class_num, class_num)
        return confusion_mat

    def cal_TP(self, confusion_mat : np.array, class_num):
        # get TP（gt ∩ predict)
        TP = np.zeros(class_num)
        i = 0
        while i<class_num:
            TP[i] = confusion_mat[i,i]
            i+=1
        return TP
    
    def cal_FP(self, confusion_mat : np.array, class_num):
        # get FP
        FP = np.zeros(class_num)
        for pre_i in np.linspace(0, class_num, num=class_num, endpoint=False, dtype=np.int64):
            item = 0
            for gt_i in np.linspace(0, class_num, num=class_num, endpoint=False, dtype=np.int64):
                if (pre_i != gt_i):
                    item = item + confusion_mat[gt_i, pre_i]
            FP[pre_i] = item
        return FP

    def cal_FN(self, confusion_mat : np.array, class_num):
        # get FN 
        FN = np.zeros(class_num)
        for gt_i in np.linspace(0, class_num, endpoint=False, num=class_num, dtype=np.int64):
            item = 0
            for pre_i in np.linspace(0, class_num, endpoint=False, num=class_num, dtype=np.int64):
                if (pre_i != gt_i):
                    item = item + confusion_mat[gt_i, pre_i]
            FN[gt_i] = item
        return FN
    
    def IOU( self, gt:torch.Tensor, pre:torch.Tensor, class_num: int):
        # IOU =（ gt ∩ predict) / (gt ∪ predict) = TP / (TP + FN + FP) 
        confusion_mat = self.cfm(gt, pre, class_num)
        TP = self.cal_TP(confusion_mat, class_num)
        FP = self.cal_FP(confusion_mat, class_num)
        FN = self.cal_FN(confusion_mat, class_num)
        eps= 1e-5
        IOU_list = (TP)/(TP+FN+FP+eps)
        return IOU_list
    
    def OA(self, gt:torch.Tensor, pre:torch.Tensor, class_num: int):
        # OA = TP / (TP + FN + FP + FN) 
        confusion_mat = self.cfm(gt, pre, class_num)
        TP = self.cal_TP(confusion_mat, class_num)

        gt = torch.flatten(gt, start_dim=0, end_dim=-1)
        total = gt.size(0)
        
        eps= 1e-5
        acc_list = (TP)/(total+eps)
        return acc_list


class LossFunc(nn.Module):
    def __init__(self) -> None:
        super(LossFunc, self).__init__()

    # Dice Loss 
    # output: class-mean 
    def Dice(self, input: Tensor, target: Tensor, eps: float = 1e-6):

        sum_dim = (-1, -2, -3) # (N C H W) 

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + eps) / (sets_sum + eps)
        return dice.mean()

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all classes
        return self.Dice(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

    def dice_loss(self, input: Tensor, target: Tensor, multiclass: bool = False):
        fn = self.multiclass_dice_coeff if multiclass else self.Dice
        return 1 - fn(input, target, reduce_batch_first=True)
    




    def iou(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        iou = dice/(2-dice)
        return iou.mean()

    def multiclass_iou(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        return self.iou(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)