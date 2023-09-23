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
1. WHU
