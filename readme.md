# sparse-engine

## 1. 简介

&emsp; &emsp; 该项目基于[TinySPOS](https://github.com/Roxbili/TinySPOS)，在通过NAS方法寻找可部署在Tiny设备上的稀疏神经网络后，自动生成可在Tiny设备上部署的代码

## 2. 使用说明
### 2.1 配置环境
&emsp; &emsp; 若要运行环境配置，请使用命令：
```
pip install -r requirements.txt
```
&emsp; &emsp; **在进行模型部署前，请在config.yml中检查各个变量是否设置正确**

### 2.2 模型部署
&emsp; &emsp; 将[TinySPOS](https://github.com/Roxbili/TinySPOS)与sparse-engine文件夹放在相同目录下，按照TinySPOS中的要求配置好环境，并使用命令：

```
bash scripts/quantization_inference_mcu.sh 
```

&emsp; &emsp; 如果运行成功，在TinySPOS/tensor目录下可以看到模型的tensor以numpy格式存储，以及包含各个层输入输出大小的statistic.yml文件。
```
TinySPOS
├ tensor
|	├ qclassifier.M.npy
|	├ qclassifier.qi.scale.npy
|	├ ...
|	└ qlast_conv.M.npy
├ ...
├ statistic.yml
```
&emsp; &emsp; 在sparse-engine文件夹下使用命令：

```
python deploy/model_deploy.py
```

&emsp; &emsp; 如果希望看到向量信息，可以在deploy/model_deploy.py文件中使用print_tensor()函数，在终端可以看到每个block中的向量信息，左侧为向量名，右侧为向量大小或者向量值，如下所示：

```
block: qclassifier
key: fc_module.weight         (10, 512)
key: M                        0.0014607312623411417
key: qi.scale                 0.01739347167313099
key: qi.zero_point            -128.0
key: qo.scale                 0.0209762342274189
key: qo.zero_point            -6.0
key: qw.scale                 0.001761617255397141
key: qw.zero_point            0.0
```

&emsp; &emsp;部署过程中会显示每个block的部署完成情况，以及最后存储至静态内存区域的向量总大小：

```
start
Block:downsample_mv2block_0 deploy completed
----------------------------------------------------------------------
Block:mv2block0_0 deploy completed
----------------------------------------------------------------------
Block:mv2block0_1 deploy completed
----------------------------------------------------------------------
Block:transformer0_0 deploy completed
----------------------------------------------------------------------
Block:transformer0_1 deploy completed
----------------------------------------------------------------------
Block:last_conv deploy completed
----------------------------------------------------------------------
Block:qglobal_pooling deploy completed
----------------------------------------------------------------------
Block:classifier deploy completed
Model deploy completed
All const tensor size: 611.79 KB
```

&emsp; &emsp; 若出现warning信息，您可以查看../TiniySPOS/statistic.yml文件内容是否正确，或模型中某一层的内存占用是否超出了最大值。

&emsp; &emsp; 在部署完成后，您可以在config.yml中查看输出文件路径，并查看输出文件。

### 2.3 稀疏推理

&emsp; &emsp; 使用Keil打开inference\MDK-ARM\uartRev1.uvprojx工程文件，Build工程后可Download至STM32F746NG开发板。调整串口波特率为115200，即可在对应串口查看稀疏推理输出结果。

&emsp; &emsp; 正常情况下，输出结果包括推理结果、top1和top5是否正确，以及推理时间，如下所示：

```
Inference result:
-64 -80 -59 -61 -65 -56 -63 -70 -70 105
Index sort:
5 9 2 3 6 1 4 7 8 0
class:9
Top 1 check: True
Top 5 check: True
Inference time(ms): 44489
```

## 3. 其他事项
### 3.1 稀疏层
&emsp; &emsp; 使用稀疏的层有：mv2block、transformer、last_conv

&emsp; &emsp; 未使用稀疏的层有：downsample、classifier、globalpool

### 3.2稀疏编码
&emsp; &emsp; 参考：https://github.com/Roxbili/TinySPOS/blob/main/doc/sparse_encode.md

&emsp; &emsp; 待办事项：压缩格式2，block稀疏化的压缩格式实现