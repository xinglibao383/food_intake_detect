# 实验记录
[M1] 2024_01_08_13_38_55 是最初的SOTA版本

[M1] 2024_01_16_15_16_05 用于 2.2 阈值选择实验

下面八个模型用于 2.1 数据融合实验

[M1] ["2024_01_20_21_07_09", "2024_01_20_21_32_41", "2024_01_20_21_40_21", "2024_01_20_22_37_36"]

[M2] ["2024_01_20_23_12_32", "2024_01_20_23_27_28", "2024_01_20_23_37_51", "2024_01_20_23_48_20"]

[M1] 2024_01_25_12_29_59 使用Autoencoder进行训练 用于1.1 backbone网络对比实验

# 实验一：模型性能实验
## 1.1 backbone网络对比实验
**Unet + Swin-Transformer**
pre_model=2024_01_16_15_16_05, post_model=2024_01_12_16_51_48
mask_percentage=15%, threshold=18.887, M1 Acc=**97.94**, M1+M2 Acc=**88.44**

**Autoencoder + Swin-Transformer**
pre_model=2024_01_25_12_29_59, post_model=2024_01_12_16_51_48
mask_percentage=15%, threshold=62.422, M1 Acc=**83.39**, M1+M2 Acc=**76.85**

**Unet + Resnet**
pre_model=2024_01_16_15_16_05, post_model=2024_01_25_12_55_29
mask_percentage=15%, threshold=18.887, M1 Acc=**97.91**, M1+M2 Acc=**80.31**

## 1.2 M2混淆矩阵实验
![confusion_matrix](E:/typora_images/confusion_matrix.png)

## 1.3 模型推理性能实验
> 实验方案：分别测试模型推理进食/非进食/进食+非进食样本的时间（毫秒）inference_time_performance_experiment.txt

![d27b30bb6904fa15d98d41a6caf9b79](E:/typora_images/d27b30bb6904fa15d98d41a6caf9b79-1706178193635-4.png)

# 实验二：消融实验
## 2.1 数据融合实验
> 手表数据采集频率50hz，眼镜数据采集频率10hz
> 为保证实验公平性：M1统一训练60epoch，学习率为0.0001，掩码率为75%，patience为5；M2统一训练100epoch，学习率为0.0001，patience为10

**仅使用眼镜数据（10hz）**

mode=glasses, pre_model=2024_01_20_21_32_41, post_model=2024_01_20_23_27_28
mask_percentage=15%, threshold=20.835, M1 Acc=**91.71**, M1+M2 Acc=**72.41**

**仅使用手表数据（50hz）**

mode=watch, pre_model=2024_01_20_21_07_09, post_model=2024_01_20_23_12_32
mask_percentage=15%, threshold=35.376, M1 Acc=**94.54**, M1+M2 Acc=**83.21**

**眼镜数据上采样到50hz**

mode=up_sample, pre_model=2024_01_20_21_40_21, post_model=2024_01_20_23_37_51
mask_percentage=15%, threshold=18.42, M1 Acc=**96.33**, M1+M2 Acc=**86.11**

**手表数据下采样到10hz**

mode=down_sample, pre_model=2024_01_20_22_37_36, post_model=2024_01_20_23_48_20
mask_percentage=15%, threshold=71.807, M1 Acc=**95.17**, M1+M2 Acc=**85.37**

## 2.2 阈值/掩码率选择实验
> 实验方案：阈值/掩码率排列组合测试

threshold_and_mask_percentage_experiment.txt

![mask+threshold1](E:/typora_images/mask+threshold1-1706178119658-1.png)

![mask+threshold2](E:/typora_images/mask+threshold2.jpg)

## 2.3 M1有效性实验

> 实验方案：如果当前类别推理出的所有类别的概率都没有超过threshold，则认定为非进食样本

post_model=2024_01_12_16_51_48
threshold=0.1, acc=0.4588173178458289
threshold=0.2, acc=0.4588173178458289
threshold=0.3, acc=0.4588173178458289
threshold=0.4, acc=0.4606652587117212
threshold=0.5, acc=0.4654171066525871
threshold=0.6, acc=0.47756071805702216
threshold=0.7, acc=0.4928722280887012
threshold=0.8, acc=0.5060718057022175
threshold=0.9, acc=0.5242872228088701

# 实验三：跨人实验

> 实验方案：留一法

```
1 0.17793462109955424
2 0.10791847644503842
3 0.14191073919107391
4 0.3247706422018349
5 0.16215213358070502
6 0.1488562649368385
7 0.30970837947582136
8 0.24473358116480792
9 0.1836890243902439
10 0.18003487358326067
```