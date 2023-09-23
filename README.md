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
* 网络结构：Unet, ResNet50(encoder)+Unet
* 损失函数：Cross-Entropy loss, Dice loss, Focal loss
* 精度评定指标：IOU, Dice, Acc
  
### 配置
### 测试
1. WHU Building Dataset (Satellite dataset I)  
   http://gpcv.whu.edu.cn/data/building_dataset.html
* 数据集信息：  
   |参数|具体设置|  
   |:--:|:--:|  
   |输入图片尺寸|（512, 512）|
   |训练集数量|202张|
   |验证集数量|66张|
   |测试集数量|67张|
   |类别数(含背景)|2|  

* 训练参数：
   |实验编号|网络结构|batchsize|epoch|learning rate|optimizer|lr-scheduler|loss|  
   |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
   |1|UNet|8|100|1e-3|AdamW|Cosine|CE-loss|
* 测试结果:
  
   |实验编号|IOU|Acc|
   |:--:|:--:|
   |1|UNet|8|
   
