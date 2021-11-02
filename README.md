# 飞桨常规赛：遥感影像地块分割
10月第2名方案

## 概述
主要借鉴了[榜首](https://aistudio.baidu.com/aistudio/projectdetail/2284066)的模型结构，但由于实现细节的差异，分数低了不少。<br>和榜首方案不同的是，个人比较偏好API编程的模式。

## 环境准备
安装paddleseg和解压数据集。
```
!pip install paddleseg==2.3.0
!cd data/data77571/ && unzip -q train_and_label.zip
!cd data/data77571/ && unzip -q img_test.zip
```

## 代码说明
### 目录结构
代码都放在work文件夹下，结果也都保存在work/result文件夹下，模型保存在models文件夹下。

|文件|内容|
|--|--|
|configs.py|全局参数设置|
|my_dataset.py|构造数据集|
|my_model.py|构造模型|
|predict.py|预测并保存结果|
|train.py|训练模型|
|utils.py|一些工具函数|

模块文件下都通过
```
if __name__=='__main__':
    ...
```
的形式编写了一些测试语句，方便调试。

### 模型
模型采用了HRNet_W48+OCRNet，loss采用LovaszSoftmax和CrossEntropy（详见my_model.py）。
```
backbone = paddleseg.models.backbones.HRNet_W48(pretrained='https://bj.bcebos.com/paddleseg/dygraph/hrnet_w48_ssld.tar.gz', has_se=False)
model = paddleseg.models.OCRNet(num_classes=4,backbone=backbone,backbone_indices=[-1],ocr_mid_channels=512,ocr_key_channels=256, pretrained='https://bj.bcebos.com/paddleseg/dygraph/ccf/fcn_hrnetw48_rs_256x256_160k/model.pdparams')
```
```
ce_coef = 1.0
lovasz_coef = 0.3
main_loss = lovasz_coef*self.lovasz(yp[0], yt)+ce_coef*self.ce(yp[0], yt)
soft_loss = lovasz_coef*self.lovasz(yp[1], yt)+ce_coef*self.ce(yp[1], yt)
return 1.0*main_loss+0.4*soft_loss
```

### 数据集
数据集采用了paddle.io.Dataset包装，方便采用paddle.io.DataLoader实现组batch和并行预处理，能够提高训练效率。

在数据集中调用paddle.vision.transforms实现数据增强，采用的增强策略有颜色抖动、随机旋转、随机翻转、随机crop等。

另外，发现训练集中有大约5000余张样本对应的标签完全由255组成，他们对训练不会有任何收益，故构造数据集时将相应的样本剔除。

代码详见my_dataset.py

### 训练
优化器为Momentum，学习率策略为PolynomialDecay和LinearWarmup。其他训练参数为：
```
BATCH_SIZE = 32
LR = 1e-3
WARMUP_EPOCH = 10 # warmup轮数
TRAIN_EPOCHS = 40 # 训练轮数
EVAL_EPOCH = 2 # 每两轮验证一次
```
训练主函数见train.py

### 预测
训练时有一个很奇怪的现象，验证集的miou曲线是一个先降后升的'U'型。采用验证集最优模型的话，提交结果不如最后一轮，所以预测时加载的模型是训练50轮的参数。

预测时使用with paddle.no_grad():或者@paddle.no_grad()装饰器可以在推理时不保存中间结果，节省巨量显存。当然AI Studio的V100显卡太强了，有没有都无所谓。

推理时把数据集用DataLoader包装一下可以更好的并行。

预测后在左侧直接右键result文件夹，选择'打包下载'即可直接下载zip压缩包，可以直接在比赛页面提交。奇怪，前两天用的时候好像还是英文'Download as zip archive'，现在就变成中文了。

具体代码见predict.py。

可通过如下命令完成训练和预测
```
# 训练
!cd work && python train.py

# 预测
!cd work && python predict.py
```
