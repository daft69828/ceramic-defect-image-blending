## 简介
这里包含三个子目录：
1. BASNet：针对缺陷图像样本，生成对应的mask图像
2. defect_blending：使用泊松融合方法，将缺陷图像融合到指定瓷砖背景中
3. mmgeneration：一个GAN模型工具箱
## 具体说明
### 1. BASNet
**步骤 1**
依照[官方教程](https://github.com/xuebinqin/BASNet)安装需要的库

**步骤 2**
使用`python basnet-test.py`进行推理，需修改文件中`image_dir`(缺陷图像的来源)、`prediction_dir`(mask的保存路径)和`model_dir`（权重文件保存位置）。
训练好的权重文件[点此下载](https://pan.quark.cn/s/be4838a52dbc)

**步骤 3**
生成结果中包含很多低质量mask图像，需要进行人工筛选。`pick.py`脚本同时展示缺陷图像及其mask，对比后按键“1”选择保存，“3”选择丢弃，“q”退出脚本
### 2. defect_blending
**前提**
缺陷图像大小为128x128，瓷砖背景图像为640x640，其他尺寸的可以用性没有验证

**步骤 1**
- 将缺陷图像（sources），放入`./sources_n_masks/${对应类别(6种)}/img`目录下
- 将缺陷掩码（masks），放入`./sources_n_masks/${对应类别(6种)}/msk`目录下
- 将瓷砖背景图像（targets），放入`./targets/${对应类别(3种)}/msk`目录下（已提供大量背景）

**步骤 2**
执行脚本：
```shell
  python main.py --category ${m} --quantity ${n}
```
m 为缺陷类型，可输入数字0~5，分别表示：边异常、角异常、白色点瑕疵、浅色块瑕疵、深色点块瑕疵、光圈瑕疵
n 为需要的缺陷数量，注意此处不是指当次融合的数量，而是指保存结果的路径中的图像总数

**步骤 3**
执行完毕后，在`./results/${对应类别}`找到融合结果，其中`images`保存融合得到的缺陷图像，`lables`保存自动生成的标注文件

### 3. mmgeneration
**步骤 1**
依照[官方教程](https://github.com/open-mmlab/mmgeneration)安装需要的库

**步骤 2**
```shell
sh tools/dist_train.sh ${配置文件路径} ${GPU数量(单卡为1)} --work-dir ${结果保存路径}
```
其中，配置文件保存在`./configs/${对应模型}`中，dcgan、lsgan、wgan、ggan可以参考我的配置文件（保存在对应模型的`my`路径下）