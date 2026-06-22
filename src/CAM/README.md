# Local Image CAM Visualizer with CORnet-S Support

本项目提供一个**独立运行**的本地图片 CAM 可视化脚本：

```text
local_image_CAM_cornet_official.py
```

该脚本可以直接读取本地图片或图片文件夹，加载模型，生成 CAM 热力图叠加结果，并保存到指定输出目录。

当前脚本支持：

```text
CORnet-S
ResNet18 / ResNet34 / ResNet50
AlexNet
VGG11 / VGG19
```

支持的 CAM 方法包括：

```text
GradCAM
GradCAM++
XGradCAM
ScoreCAM
AblationCAM
EigenCAM
FullGrad
```

---

## 1. 文件结构建议

建议项目目录如下：

```text
D:\image_CAM\
│
├─ local_image_CAM_cornet_official.py
│
├─ input\
│   ├─ fig1.png
│   ├─ fig2.png
│   └─ ...
│
├─ weights\
│   └─ cornet_s-1d3f7974.pth
│
└─ cam_outputs\
```

其中：

```text
input\      放待可视化的图片
weights\    放 CORnet-S 官方预训练权重
cam_outputs\ 脚本自动生成的输出结果目录
```

---

## 2. 安装依赖

建议先进入你的 conda 环境，例如：

```powershell
conda activate carla_cam
```

安装基础依赖：

```powershell
python -m pip install numpy pillow torch torchvision grad-cam
```

如果你已经安装过 `torch` 和 `torchvision`，可以只补充缺失包：

```powershell
python -m pip install numpy pillow grad-cam
```

安装后可以检查 PyTorch 是否正常：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出：

```text
True
```

说明当前环境可以使用 CUDA。  
如果输出：

```text
False
```

则脚本会默认使用 CPU 运行，速度会慢一些，但仍然可以生成 CAM。

---
## 3.下载输入数据
你可以下载我们使用的[输入数据](
https://pan.baidu.com/s/1qZNsVYPIGsvRT4wOe1mT0Q?pwd=iiyh)  
然后放置在input文件夹下
## 4. 无需安装CORnet
为了方便使用，当前脚本内部已经定义了 CORnet-S 网络结构，包括：

```text
V1
V2
V4
IT
decoder
```

因此不需要执行：

```powershell
pip install git+https://github.com/dicarlolab/CORnet
```

只要准备好官方 CORnet-S 权重文件即可。

---

## 5. 下载 CORnet-S 官方权重

CORnet-S 官方 ImageNet 预训练权重文件名为：

```text
cornet_s-1d3f7974.pth
```

官方权重下载地址：

```text
https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth
```

### 方式 A：使用浏览器下载

在浏览器中打开：

```text
https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth
```

下载完成后，将文件放到：

```text
D:\image_CAM\weights\cornet_s-1d3f7974.pth
```

### 方式 B：PowerShell 下载

先创建权重目录：

```powershell
mkdir D:\image_CAM\weights
```

使用 `Invoke-WebRequest` 下载：

```powershell
Invoke-WebRequest `
  -Uri "https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth" `
  -OutFile "D:\image_CAM\weights\cornet_s-1d3f7974.pth"
```

如果失败，可以改用：

```powershell
curl.exe -L `
  "https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth" `
  -o "D:\image_CAM\weights\cornet_s-1d3f7974.pth"
```


## 6. 使用 CORnet-S 生成 CAM

### 6.1 单层可视化：IT 层

最推荐先使用 `IT.output`：

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_cornet_IT" `
  --model cornet_s `
  --weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth" `
  --target-layer IT.output `
  --method GradCAM `
  --recursive
```

### 6.2 可选 CORnet-S target layer

CORnet-S 支持以下 target layer：

```text
V1.output
V2.output
V4.output
IT.output
```

一般解释：

```text
V1.output: 低层边缘、颜色、局部纹理
V2.output: 中低层局部形状
V4.output: 中高层物体结构
IT.output: 高层类别相关响应
```

## 7. 一次性跑四个 CORnet-S 层

在 PowerShell 中运行：

```powershell
$layers = @("V1.output", "V2.output", "V4.output", "IT.output")

foreach ($layer in $layers) {
    $outLayer = $layer.Replace(".", "_")

    python local_image_CAM_cornet_official.py `
      --input "D:\image_CAM\input" `
      --output "D:\image_CAM\cam_outputs_cornet_$outLayer" `
      --model cornet_s `
      --weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth" `
      --target-layer $layer `
      --method GradCAM `
      --recursive
}
```

运行完成后会得到：

```text
D:\image_CAM\cam_outputs_cornet_V1_output
D:\image_CAM\cam_outputs_cornet_V2_output
D:\image_CAM\cam_outputs_cornet_V4_output
D:\image_CAM\cam_outputs_cornet_IT_output
```

如果希望同时保存灰度 heatmap 和处理后的输入图像，可以加：

```powershell
--save-heatmap `
--save-processed-image
```

完整命令：

```powershell
$layers = @("V1.output", "V2.output", "V4.output", "IT.output")

foreach ($layer in $layers) {
    $outLayer = $layer.Replace(".", "_")

    python local_image_CAM_cornet_official.py `
      --input "D:\image_CAM\input" `
      --output "D:\image_CAM\cam_outputs_cornet_$outLayer" `
      --model cornet_s `
      --weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth" `
      --target-layer $layer `
      --method GradCAM `
      --recursive `
      --save-heatmap `
      --save-processed-image
}
```

---

## 8. 使用其他模型生成 CAM

除了 CORnet-S，该脚本也支持 torchvision 的 ImageNet 预训练模型。

### ResNet34

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_resnet34" `
  --model resnet34 `
  --method GradCAM `
  --recursive
```

### ResNet50

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_resnet50" `
  --model resnet50 `
  --method GradCAM `
  --recursive
```

### AlexNet

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_alexnet" `
  --model alexnet `
  --method GradCAM `
  --recursive
```

### VGG19

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_vgg19" `
  --model vgg19 `
  --method GradCAM `
  --recursive
```

对于 ResNet、AlexNet、VGG，如果不指定 `--weights`，脚本会默认使用 torchvision 的 ImageNet 预训练权重。

---

## 9. 更换 CAM 方法

只需要修改 `--method` 参数。

例如 CORnet-S + GradCAM++：

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_cornet_gradcampp" `
  --model cornet_s `
  --weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth" `
  --target-layer IT.output `
  --method "GradCAM++" `
  --recursive
```

CORnet-S + XGradCAM：

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_cornet_xgradcam" `
  --model cornet_s `
  --weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth" `
  --target-layer IT.output `
  --method XGradCAM `
  --recursive
```

建议优先使用：

```text
GradCAM
GradCAM++
XGradCAM
EigenCAM
```

`ScoreCAM` 和 `AblationCAM` 通常较慢，尤其在 CPU 环境下不建议一开始批量运行。

---

## 10. 指定可视化类别

默认情况下，脚本使用模型预测的 Top-1 类别作为 CAM target。

如果想固定可视化某个 ImageNet 类别 index，可以使用：

```powershell
--target-class 281
```

例如：

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\cam_outputs_class281" `
  --model cornet_s `
  --weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth" `
  --target-layer IT.output `
  --target-class 281 `
  --method GradCAM `
  --recursive
```

---

## 11. 查看模型可用层名

如果不确定 target layer 名称，可以运行：

```powershell
python local_image_CAM_cornet_official.py `
  --input "D:\image_CAM\input" `
  --output "D:\image_CAM\tmp" `
  --model cornet_s `
  --weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth" `
  --list-layers
```

## 12. 输出结果

脚本会在 `--output` 指定目录中保存 CAM 叠加图，例如：

```text
fig1_cornet_s_GradCAM_IT_output_class281_overlay.png
```

同时会保存预测结果表：

```text
cam_predictions_cornet_s_GradCAM_IT_output.csv
```

如果添加：

```powershell
--save-heatmap
```

会额外保存灰度 CAM 热力图：

```text
fig1_cornet_s_GradCAM_IT_output_class281_heatmap.png
```

如果添加：

```powershell
--save-processed-image
```

会额外保存经过 resize 和 center crop 后的输入图片：

```text
fig1_processed_224.png
```

---

## 13. 常见问题

### 13.1 FileNotFoundError: Checkpoint does not exist

说明 `--weights` 后面的路径不存在。

检查文件是否存在：

```powershell
dir "D:\image_CAM\weights"
```

确保路径写成真实存在的文件，例如：

```powershell
--weights "D:\image_CAM\weights\cornet_s-1d3f7974.pth"
```

### 13.2 当前 pytorch-grad-cam 版本不支持 image_weight

如果看到：

```text
[WARN] 当前 pytorch-grad-cam 版本不支持 image_weight，已使用默认叠加强度。
```

这不是严重错误，只是当前 `grad-cam` 版本不支持透明度参数。脚本会自动回退到默认叠加强度，CAM 仍然会正常生成。

### 13.3 运行速度很慢

如果日志显示：

```text
[INFO] Device: cpu
```

说明当前使用 CPU。CPU 上运行 `ScoreCAM` 和 `AblationCAM` 会非常慢。建议优先使用：

```text
GradCAM
GradCAM++
XGradCAM
EigenCAM
```

### 13.4 torchvision 模型自动下载失败

ResNet、AlexNet、VGG 默认使用 torchvision ImageNet 预训练权重。如果首次运行时本地没有缓存，torchvision 可能需要联网下载。网络不稳定时，可以先使用 CORnet-S 本地权重，或提前手动准备 torchvision 权重缓存。


