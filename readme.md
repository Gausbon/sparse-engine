# sparse-engine

## 1. 简介

&emsp; &emsp; 该项目基于[TinySPOS](https://github.com/Roxbili/TinySPOS)，在通过NAS方法寻找可部署在Tiny设备上的稀疏神经网络后，自动生成可在Tiny设备上部署的代码

&emsp; &emsp; 针对Tiny设备使用的CMSIS库文件，请参考[CMSIS_5_FOR_TINYSPOS](https://github.com/Gausbon/CMSIS_5_FOR_TINYSPOS)

## 2. 使用说明
### 2.1 配置环境
&emsp; &emsp; 若要运行环境配置，请使用命令：
```
pip install -r requirements.txt
```
&emsp; &emsp; 若需要修改模型部署相关的参数，请在config.yml中进行改动

### 2.2 模型部署
&emsp; &emsp; 将[TinySPOS](https://github.com/Roxbili/TinySPOS)与sparse-engine文件夹放在相同目录下，按照TinySPOS中的要求配置好环境

&emsp; &emsp; 在TinySPOS目录下按要求配置环境，并使用命令
```
bash scripts/quantization_inference_mcu.sh 
```

&emsp; &emsp; 如果运行成功，在TinySPOS/tensor目录下可以看到模型的tensor以numpy格式存储
```
TinySPOS
├ tensor
|	├ qclassifier.M.npy
|	├ qclassifier.qi.scale.npy
|	├ ...
|	└ qlast_conv.M.npy
```
> 目前保存tensor的代码尚未merge进TinySPOS中

&emsp; &emsp; 在sparse-engine文件夹下使用命令 

```
python deploy/model_deploy.py
```

&emsp; &emsp; 如果运行成功，在终端可以看到每个block中的向量信息，左侧为向量名，右侧为向量大小或者向量值

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

&emsp; &emsp; 同时在最后还会显示每一个block部署成功的信息

```
Block:downsample_mv2block_0 deploy completed
--------------------------------------------
Block:mv2block1_0 deploy completed
--------------------------------------------
Block:mv2block0_1 deploy completed
--------------------------------------------
Block:transformer1_0 deploy completed
--------------------------------------------
Block:transformer0_1 deploy completed
--------------------------------------------
Block:last_conv deploy completed
--------------------------------------------
Block:global_pooling deploy completed
--------------------------------------------
Block:classifier deploy completed
Model deploy completed
Remaining size list count: 0 (0 is correct)
```

&emsp; &emsp; 您可以在config.yml中查看输出文件路径，并查看输出文件

### 2.3 稀疏推理

&emsp; &emsp; 待办事项：完成并验证稀疏推理过程

## 3. 其他事项
### 3.1 稀疏层
&emsp; &emsp; 使用稀疏的层有：mv2block、transformer、last_conv

&emsp; &emsp; 未使用稀疏的层有：downsample、classifier、globalpool

### 3.2稀疏编码
&emsp; &emsp; 参考：https://github.com/Roxbili/TinySPOS/blob/main/doc/sparse_encode.md

&emsp; &emsp; 待办事项：压缩格式2，block稀疏化的压缩格式实现