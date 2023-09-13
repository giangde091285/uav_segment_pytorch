import numpy as np
import torch

def cal_confusion_mat(gt:np.array, pre:np.array, class_num: int):
    # reference: https://www.jianshu.com/p/647c015323bb

    # convert to 1D
    gt = gt.flatten()
    pre = pre.flatten()
    # get category
    categroy =  class_num * gt + pre 
    # get confusion mat
    # row : gt , col : pre
    confusion_mat = np.zeros(class_num*class_num, dtype=int)

    for i in range(0, confusion_mat.size, 1):
        confusion_mat[i] = len(np.where(categroy == i))

    confusion_mat = confusion_mat.reshape(class_num, class_num)
    return confusion_mat


def cal_TP(confusion_mat : np.array, class_num):
    # get TP（gt ∩ predict)
    TP = np.zeros(class_num)
    for i in range(0, class_num, 1):
        TP[i] = confusion_mat[i,i]
    return TP


def cal_FP(confusion_mat : np.array, class_num):
    # get FP
    FP = np.zeros(class_num)
    for pre_i in range(0, class_num, 1):
        item = 0
        for gt_i in range(0, class_num, 1):
            if (pre_i != gt_i):
                item = item + confusion_mat[gt_i, pre_i]
        FP[pre_i] = item
    return FP

def cal_FN(confusion_mat : np.array, class_num):
    # get FN 
    FN = np.zeros(class_num)
    for gt_i in range(0, class_num, 1):
        item = 0
        for pre_i in range(0, class_num, 1):
            if (pre_i != gt_i):
                item = item + confusion_mat[gt_i, pre_i]
        FN[gt_i] = item
    return FN

def mIOU( gt:np.array, pre:np.array, class_num: int, unique_value):
    # IOU =（ gt ∩ predict) / (gt ∪ predict) = TP / (TP + FN + FP) 
    confusion_mat = cal_confusion_mat(gt, pre, class_num)
    TP = cal_TP(confusion_mat, class_num)
    FP = cal_FP(confusion_mat, class_num)
    FN = cal_FN(confusion_mat, class_num)
    eps= 1e-3
    IOU = TP/(TP+FN+FP+eps)
    uni_class = len(unique_value)
    mIOU = np.sum(IOU)/uni_class
    return mIOU,IOU


def Dice_loss(gt:torch.Tensor, pre:torch.Tensor, class_num:int):

    # gt(N H W) pre(N C H W) , C is probability
    pre = torch.argmax(pre, dim=1).cpu()
    gt = gt.cpu() # (N H W) , value is category

    gt_ar = np.asarray(gt, dtype=np.uint8)
    pre_ar = np.asarray(pre, dtype=np.uint8) 
    
    cfm = cal_confusion_mat(gt,pre,class_num)
    TP = cal_TP(cfm, class_num)
    FP = cal_FP(cfm, class_num)
    FN = cal_FN(cfm, class_num)
    eps= 1e-4

    Dice_loss = 1 - 2*TP/(2*TP+FN+FP+eps)

    mean_loss = np.sum(Dice_loss)/class_num
    
    mean_loss = torch.tensor([mean_loss],requires_grad=True)
    return mean_loss
