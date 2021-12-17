# CHIP2021医学对话临床发现阴阳性判别任务冠军方案开源

## 简介

比赛名称：CHIP2021评测一: 医学对话临床发现阴阳性判别任务

测评任务：针对互联网在线问诊记录中的临床发现进行阴阳性的分类判别

测评链接：http://www.cips-chip.org.cn/2021/eval1

方案细节分析：https://mp.weixin.qq.com/s/URBsSyamGoSzpQP4aPDb3Q

## 环境

### 运行设备

```
CPU 型号 AMD EPYC 7742 64-Core Processor
CPU 核数16 核
磁盘空间290 GB
显卡型号RTX A6000
显存大小48 GB
内存大小58 GB
```

### 运行python环境
```
Python 3.8.10
pip install ark-nlp==0.0.2
pip install pandas
pip install scikit-learn
```

## 使用说明

### 训练说明

训练模型请到**train**文件夹下运行

`bash ./train.sh`

PS：

1. 训练过程采用过滤后的训练文件，已包含在'data/source_dataset'，过滤思路可以参考系统描述文档，生成方法`gen_fliter_data.py`

2. 预训练模型可通过`pretrain.py`获取


### 预测说明

预测请到**predict**文件夹下，相关命令如下

- 集成预测，可获得排行榜成绩结果

  `bash ./predict.sh ensemble`

- 单模型预测，可预测选项有`medbert`、`mcbert`、`macbert2-f-f`、`macbert2-f`、`dialog_chinese-macbert`，执行命令如下：

  `bash ./predict.sh medbert`

  `bash ./predict.sh mcbert`

  `bash ./predict.sh macbert2-f-f`

  `bash ./predict.sh macbert2-f`

  `bash ./predict.sh dialog_chinese-macbert`
  
  
### 备注

由于最近其他事情比较多，暂时还没有修整和注释，后期有时间会重新进行改进并适配到ark-nlp最新版本
