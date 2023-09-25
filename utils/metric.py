import numpy as np
import torch
from torch import *
from torch import nn

"""
    metric 
   
    * IOU
    * Dice Score
    * Acc
    * Confusion Mat

"""

class Metric(nn.Module):
    def __init__(self, if_ignore: bool = True, ignore_idx:int = None):
        super(Metric, self).__init__()
        self.ignore_idx = ignore_idx
        self.if_ignore = if_ignore

    def cal_intersect_and_union(self, input: torch.Tensor, target: torch.Tensor, class_num: int):
        
        # input (N C H W)-> (N H W C)-> (NHW C)-> (NHW)
        input = input.permute(0, 2, 3, 1)
        input = torch.flatten(input, 0, 2)
        input = torch.argmax(input, 1)
        input = torch.as_tensor(input, dtype=torch.float32)
        # target (N H W) -> (NHW)
        target = target.flatten(0, 2)

        # 忽略nodata处
        if self.if_ignore:
            # 获取target值不为-1的索引
            mask = ~ target.eq(self.ignore_idx)  # 逐元素对比，如果相同返回true，否则 false
            # 取值不为-1的元素，组成新target
            target = target[mask]   # (n')
            # 取target值不为-1的元素，组成新input
            input = input[mask]   # (n')

        # A ∩ B = TP
        intersect = input[input == target]
        intersect = torch.histc(intersect, class_num, min=0, max=class_num-1).cpu()  # torch.histc()只接受float类型的tensor
        # A = TP + FP
        tp_plus_fp = torch.histc(input, class_num, min=0, max=class_num - 1).cpu()
        # B = TP + FN
        tp_plus_fn = torch.histc(target, class_num, min=0, max=class_num - 1).cpu()
        # A ∪ B = TP + FN + FP
        union = tp_plus_fp + tp_plus_fn - intersect
      
        return intersect, union, tp_plus_fp, tp_plus_fn

    def IOU(self, input: torch.Tensor, target: torch.Tensor, class_num: int):
        # iou = (A ∩ B)/(A ∪ B) = TP/(TP+FN+FP)
        intersect, union, _, _= self.cal_intersect_and_union(input, target, class_num)
        eps = 1e-4
        iou = (intersect+eps)/(union+eps)
        return iou # (c)
    
    def DiceScore(self, input: torch.Tensor, target: torch.Tensor, class_num: int):
        # Dice = 2*(A ∩ B)/(A + B) = 2*TP/(2*TP+FN+FP)
        intersect, _, tp_plus_fp, tp_plus_fn = self.cal_intersect_and_union(input, target, class_num)
        dice = 2 * intersect/(tp_plus_fp + tp_plus_fn)
        return dice # (c)
    
    def Acc(self, input: torch.Tensor, target: torch.Tensor, class_num: int):
        # Acc = (TP+TN) / total_num
        tp, tp_fp_fn , _, _ = self.cal_intersect_and_union(input, target, class_num)
        total_num = input.size(0) * input.size(2) * input.size(3)
        
        acc = (total_num - tp_fp_fn + tp) / total_num
        return acc # (c)


class ConfusionMat(nn.Module):
    def __init__(self) -> None:
        super(ConfusionMat, self).__init__()

    def cfm(self, input, target, class_num):
        # target (N H W) -> (NHW)
        target = target.flatten(0,2)
        # input (N C H W) -> (N H W C) -> (NHW C) -> (NHW)
        input = input.permute(0, 2, 3, 1).flatten(0,2)
        input = torch.argmax(input, dim=1)

        # get category
        categroy = class_num * target + input
        categroy = np.asarray(categroy.cpu(), dtype=np.uint8) 
        # get confusion mat  -- row : gt , col : pre
        confusion_mat = np.zeros((class_num*class_num), dtype=np.uint8)
        for i in np.linspace(0, confusion_mat.size, endpoint=False, num=confusion_mat.size, dtype=np.int64):
            confusion_mat[i] = np.sum(categroy == i)
        confusion_mat = confusion_mat.reshape(class_num, class_num)
        return confusion_mat









