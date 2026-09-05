# BeautyProof EfficientNet-B0 Retouch Detector V2

## 1. 模型概览

V2 是 BeautyProof 项目的二分类人像修图检测模型，用于判断输入图片是否存在数字美颜/修图。

- 架构：EfficientNet-B0
- 任务：`no_retouch`（0）/ `digital_retouch`（1）
- 输入：RGB 人像图片
- 输出：数字修图概率
- 固定判定阈值：0.5
- Checkpoint：`best_model.pt`
- Checkpoint SHA256：`52F38353CEB4F20325B8AF84C0A0973FD48FEB57323B4429465D5C10FCFDC94D`
- 文件大小：16,328,815 bytes

V2 是当前 V2/V3/V4 三个版本中最适合 MVP 使用的版本：主测试 Accuracy/F1 均超过 94%，旧 hard-negative FPR 为 3.75%。

## 2. 能做什么

模型当前可以：

1. 对单张人像给出是否存在数字修图的二分类结果。
2. 输出 `retouch_probability`，表示模型对数字修图类别的置信概率。
3. 对 JPEG 压缩、缩放、曝光、轻度模糊等非美颜变化具备一定抗干扰能力。

模型当前不能可靠完成：

- 判断具体修改了脸的哪个部位；
- 区分磨皮、美白、瘦脸等修图类型；
- 判断人物是否化妆；
- 证明某个美妆产品具有特定功效；
- 将概率解释为修图强度或司法级真实性证据。

## 3. 模型结构

模型基于 `torchvision.models.efficientnet_b0`：

```python
from torch import nn
from torchvision.models import efficientnet_b0

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
```

最后一层输出一个 logit，经 sigmoid 转换为修图概率。

## 4. Checkpoint 格式

`best_model.pt` 是 PyTorch checkpoint 字典，而不是 TorchScript 文件。主要字段为：

```text
model_state   EfficientNet-B0 state_dict
epoch         保存时的训练轮次
metric        验证选择指标
config        训练配置（若存在）
```

只应加载可信来源的 PyTorch checkpoint。本仓库文件可使用上面的 SHA256 校验。

## 5. 输入预处理

V2 的推理预处理必须保持如下顺序：

1. 使用 Pillow 打开图片并转换为 RGB；
2. 将短边缩放到 256；
3. 中心裁剪为 224×224；
4. 转为 `[0, 1]` tensor；
5. 使用 ImageNet 均值和标准差归一化。

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

V2 不使用 V3/V4 的 MediaPipe 人脸裁剪和背景抑制。将 V3 预处理套到 V2 上会导致分布不一致。

## 6. Python 推理示例

依赖：

```bash
pip install torch torchvision pillow
```

```python
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

checkpoint_path = Path("models/retouch_detector_v2/best_model.pt")
image_path = Path("example.jpg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state"])
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

with Image.open(image_path) as image:
    batch = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    probability = torch.sigmoid(model(batch).flatten()[0]).item()

result = {
    "digital_retouch": probability >= 0.5,
    "retouch_probability": probability,
    "confidence": max(probability, 1.0 - probability),
    "threshold": 0.5,
}
print(result)
```

## 7. 冻结评测结果

### 主测试集

| 指标 | 数值 |
|---|---:|
| Accuracy | 94.00% |
| Precision | 92.31% |
| Recall | 96.00% |
| F1 | 94.12% |
| AUROC | 98.72% |
| TN / FP / FN / TP | 69 / 6 / 3 / 72 |

### 旧 Hard-negative 外部集

| 指标 | 数值 |
|---|---:|
| 样本数 | 80 |
| True negatives | 77 |
| False positives | 3 |
| FPR | 3.75% |

### 合成修图强度召回

| 强度 | Recall |
|---|---:|
| Low | 25.33% |
| Medium | 73.33% |
| High | 90.67% |

所有结果均使用固定阈值 0.5。单类别 hard-negative 集不报告 AUROC。

## 8. Shortcut Risk 与限制

V2 的整体 Shortcut Risk 被审计为 **HIGH**。主要原因是训练来源中的分辨率、格式和数据源与标签存在关联，模型可能部分依赖这些非语义特征，而不是只关注皮肤和真实修图痕迹。

使用时应注意：

- 对社交媒体截图、极低分辨率图片、生成式 AI 人脸和未知相机管线可能失效；
- Low-strength recall 只有 25.33%，轻微修图容易漏检；
- 输出是模型判断，不等价于“图片造假”的事实证明；
- 不应作为医疗、法律、招聘或身份认证的唯一决策依据；
- 在新数据域部署前必须建立独立外部测试集重新评测。

## 9. 随附文件

| 文件 | 说明 |
|---|---|
| `best_model.pt` | 推荐部署 checkpoint |
| `config.yaml` | V2 训练配置 |
| `label_map.json` | 类别映射 |
| `metrics.json` | 主冻结测试指标 |
| `external_metrics.json` | 旧 hard-negative 外部指标 |
| `strength_metrics.csv` | 低/中/高修图敏感度 |
| `SHA256SUMS.txt` | 文件完整性校验 |

## 10. 版本建议

在现有 V2、V3、V4 中，V2 的主任务与旧 hard-negative 综合表现最好，因此保留为当前推荐版本。V4 虽显著改善新 hard-negative 与低强度召回，但主测试 Accuracy/F1 下降到 86.67%，尚不能替代 V2。

