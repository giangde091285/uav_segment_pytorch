# segment_pytorch
### 内容
* 利用pytorch框架实现最基础的语义分割
* 在遥感影像数据集上测试
### 代码结构
```
--data_process
  |-- ori_data           # 原始image、mask
      |-- img
      |-- mask
  |-- dataset            # 运行dataset_spilt.py后生成
      |-- train
          |-- img
          |-- mask  
      |-- val
          |-- img
          |-- mask
      |-- test
          |-- img
          |-- mask
--model                   # 网络结构
  |-- block.py
  |-- seg_model.py
--utils
  |-- dataset_load.py     # 数据预处理、数据加载
  |-- dataset_spilt.py    # 划分数据集
  |-- loss.py             # 损失函数
  |-- metric.py           # 精度评定
  |-- optimizer.py        # 优化器和学习率调整
--opt.py
--train.py
--predict.py

```
* 网络结构：Unet,  ResNet50(encoder)+Unet
* 损失函数：Cross-Entropy loss,  Dice loss,  [Focal loss](https://github.com/RefineM/FocalLoss_multiclass)
* 精度评定指标：IOU,  Dice-Score,  Acc
  
### 配置
1. 安装Anaconda
2. 安装CUDA
3. 创建虚拟环境并切换
   ```
     conda create -n [name] python==3.8
     conda activate [name]
   ```
4. 安装gpu版torch(cuda 11.3)
   ```
     pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
   ```
5. 安装其他所需的包
   ```
     pip install wandb
     pip install pillow
     pip install tqdm
     pip install numpy
   ```
6. 将图片、标签放在ori_data文件夹
7. 修改opt.py中的参数，自定义划分和裁剪数据集，自定义超参数
8. 运行train.py得到权重.pth文件
9. 使用训练得到的权重文件，运行predict.py进行预测
   
### 测试
1. ***WHU Building Dataset (Satellite dataset I)***  
   [下载](http://gpcv.whu.edu.cn/data/building_dataset.html)
* 数据集信息：  
   |输入图片尺寸|类别数(含背景)|训练集|验证集|测试集|
   |:--:|:--:|:--:|:--:|:--:|   
   |512*512|2|202张|66张|67张|
  
* 训练参数：
   |实验编号|网络结构|batchsize|epoch|learning rate|optimizer|lr-scheduler|loss|使用预训练模型| 
   |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
   |whu-1|UNet|8|100|1e-3|AdamW|Cosine|CE-loss|×|
  
* 训练曲线：
   ![图片1](https://github.com/RefineM/segment_pytorch/assets/112970219/a4fd2895-af13-4d19-b9f2-e6bb958815fd)

* 可视化：
  ![图片1(1)](https://github.com/RefineM/segment_pytorch/assets/112970219/3622e3c3-eadd-4dae-905a-86d96ac3734e)

* 测试结果:  
   |实验编号|IOU(%)|Dice(%)|Acc(%)|  
   |:--:|:--:|:--:|:--:|  
   |whu-1|96.09|97.99|98.58|  
  
2. ***LoveDA Dataset***  
   [下载](http://junjuewang.top/)

### 参考
* https://github.com/milesial/Pytorch-UNet
* https://github.com/open-mmlab/mmsegmentation
