# segment_pytorch
## 内容📝
* 利用pytorch框架实现最基础的语义分割
* 在遥感影像数据集上测试
## 代码结构🐳
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
* 网络结构：
   * Unet
   * ResNet50 + Unet （TO DO）
* 损失函数：
   * Cross-Entropy loss
   * Dice loss
   * [Focal loss](https://github.com/RefineM/FocalLoss_multiclass)
* 精度评定指标：
   * IOU
   * Dice-Score
   * Acc
  
## 开始👋
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
6. 将数据集的图片、标签放在ori_data文件夹
7. 修改opt.py中的参数，自定义划分数据集并裁剪图像，自定义各种超参数
8. 运行train.py，每一个epoch的权重文件（.pth）保存在checkpoint文件夹之下
9. 使用训练得到的权重文件，运行predict.py进行预测
   
## 测试🔨
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
* 数据集信息：
  
  （1）将该数据集的train/urban文件夹下的img和mask作为原始数据，进行数据集划分和裁剪  
  （2）设置opt.py中的参数，以0.7：0.1：0.2的比例划分训练、验证、测试集，将训练和验证集的每张原始影像由（1024，1024）随机裁剪为4张（256，256）的小图
    
   |输入图片尺寸|类别数(含背景)|训练集|验证集|
   |:--:|:--:|:--:|:--:|
   |256*256|7|3240张|460张|

* 训练参数：
  
   |实验编号|网络结构|batchsize|epoch|learning rate|optimizer|lr-scheduler|loss|使用预训练模型| 
   |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
   |love-1|UNet|16|50|1e-3|AdamW|Cosine|DiceLoss|×|

* 训练曲线：
 
* 可视化：

* 测试结果:  
   |实验编号|IOU(%)|Dice(%)|Acc(%)|  
   |:--:|:--:|:--:|:--:|  
   |love-1|96.09|97.99|98.58|  

## 参考📚
* u-net网络结构：
  https://github.com/milesial/Pytorch-UNet
* LoveDA数据集加载：
  https://github.com/Junjue-Wang/LoveDA
* 损失函数和精度评定：
  https://github.com/open-mmlab/mmsegmentation
