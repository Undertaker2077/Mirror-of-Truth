# Mirror of Truth

Mirror of Truth 是一个面向人像数字修图审计的机器学习项目。当前统一 API 可以判断图片是否疑似经过数字修图，进一步识别磨皮美白、瘦脸和五官立体效果，并返回疑似修改区域。

项目采用三模型协同架构：BeautyProof V2 负责最终二分类决策，Three Type V1 负责修图类型与强度，YOLO11n Seg 负责区域定位。调用方只需要接入一个 Python 接口、命令行工具或 FastAPI 服务。

> 当前结果属于概率性辅助判断。三类型模型和区域模型主要在受控合成数据上验证，不能用于证明化妆品功效，也不能确定图片是否完全由 AI 生成。

## 当前能力

| 能力 | API 字段 | 状态 |
|---|---|---|
| 判断是否疑似数字修图 | `retouched`、`retouch_probability` | 正式提供 |
| 磨皮美白 | `skin_enhancement` | 正式提供 |
| 瘦脸 | `face_slimming` | 正式提供 |
| 五官立体 | `facial_contouring` | 正式提供 |
| 疑似修改区域定位 | `modified_regions`、`region_status` | 正式提供 |
| 放大眼睛 | `eye_enlargement` | 未发布，控制误报后召回不足 |
| 是否化妆 | 无 | 不支持 |
| AI 生成图片检测 | 无 | 未并入统一生产 API |
| 美妆产品功效证明 | 无 | 不支持 |

## 系统架构

```text
JPEG / PNG 人像
       │ multipart/form-data 或本地路径
       ▼
BeautyProof Unified API
       ├── BeautyProof V2
       │      └── 是否修图 + 修图概率
       ├── Retouch Three Type V1
       │      ├── 磨皮美白
       │      ├── 瘦脸
       │      └── 五官立体
       └── YOLO11n Seg Regions V1
              └── 疑似修改区域多边形
       ▼
统一 JSON 响应
```

统一编排规则：

1. V2 是 `retouched` 的权威决策模型。
2. 只有 V2 判定为修图时，才执行类型分类和区域定位。
3. 类型模型采用多标签输出，一张图片可以同时命中多种效果。
4. 每种类型使用独立验证阈值。
5. 没有定位到区域不能推翻 V2 的阳性判断。
6. V2 为阴性时，API 抑制下游类型和区域结论。

## 模型与指标

### BeautyProof V2

EfficientNet-B0 二分类模型，区分 `no_retouch` 和 `digital_retouch`。输入预处理为缩放到 256、中心裁剪到 224，并使用 ImageNet 归一化。

| 指标 | 冻结测试结果 |
|---|---:|
| Accuracy | 94.00% |
| Precision | 92.31% |
| Recall | 96.00% |
| F1 | 94.12% |
| AUROC | 98.72% |

主测试集包含 150 张图片。详情参见 [`models/retouch_detector_v2/metrics.json`](models/retouch_detector_v2/metrics.json)。

### Retouch Three Type V1

EfficientNet-B0 多任务模型。训练清单包含 421 个单人身份和 4,631 张图片，并按身份隔离为 3,311 张训练、660 张验证和 660 张测试图片。

| 类型 | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| 磨皮美白 | 95.24% | 100.00% | 97.56% | 99.98% |
| 瘦脸 | 100.00% | 95.56% | 97.73% | 99.81% |
| 五官立体 | 94.44% | 94.44% | 94.44% | 99.17% |

三类 Macro F1 为 96.58%。这些结果来自身份隔离的合成测试集，只说明模型能识别当前生成配方，不代表任意真实美颜软件上的准确率。

### YOLO11n Seg Regions V1

区域模型使用 3,500 张、500 个身份的混合数据训练，划分为 2,450 张训练、525 张验证和 525 张测试图片。

| 指标 | 最佳验证结果 |
|---|---:|
| Mask Precision | 68.77% |
| Mask Recall | 74.58% |
| Mask mAP50 | 68.90% |
| Mask mAP50-95 | 62.60% |

区域输出是疑似定位，不单独证明图片应用了哪一种修图操作。

## 环境要求

- Python 3.11 或更高版本
- 推荐使用支持 CUDA 的 NVIDIA GPU；无 GPU 时自动回退到 CPU
- Windows、Linux 或 macOS
- REST 服务需要 FastAPI、Uvicorn 和 `python-multipart`

## 安装

```bash
git clone https://github.com/Undertaker2077/Mirror-of-Truth.git
cd Mirror-of-Truth
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux 或 macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

首次推理会延迟加载模型。使用 CUDA 时需要安装与本机驱动兼容的 PyTorch 版本；仓库不会自动修改系统 CUDA 环境。

## 使用方法

### Python API

```python
from beautyproof_api import UnifiedBeautyProofAPI

api = UnifiedBeautyProofAPI()
result = api.analyze("face.jpg")

print(result["retouched"])
print(result["retouch_probability"])
print(result["retouch_types"])
print(result["modified_regions"])
```

推荐保留逐类别阈值：

```python
api = UnifiedBeautyProofAPI(
    retouch_threshold=0.5,
    type_threshold=None,
)
```

`type_threshold=None` 表示采用模型配置中的逐类别阈值。

### 命令行

```bash
python -m beautyproof_api.cli face.jpg
python -m beautyproof_api.cli face.jpg --output result.json
```

### REST API

```bash
uvicorn beautyproof_api.server:app --host 0.0.0.0 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/analyze -F "image=@face.jpg"
```

启动后可访问交互式接口文档：`http://127.0.0.1:8000/docs`。

## 响应示例

```json
{
  "schema_version": "1.0",
  "retouched": true,
  "retouch_probability": 0.923114,
  "retouch_types": [
    {
      "name": "skin_enhancement",
      "probability": 0.981733,
      "strength": 0.71
    },
    {
      "name": "face_slimming",
      "probability": 0.874201,
      "strength": 0.58
    }
  ],
  "retouch_strength": 0.71,
  "modified_regions": [
    {
      "name": "left_cheek",
      "confidence": 0.741203,
      "polygon": [[0.21, 0.43], [0.46, 0.51], [0.38, 0.72]]
    }
  ],
  "region_status": "localized",
  "models": {
    "binary_detector": "BeautyProof-V2",
    "type_classifier": "BeautyProof-Retouch-Three-Type-V1",
    "region_segmenter": "YOLO11n-Seg-30e"
  },
  "limitations": [
    "The three retouch types and region localization are validated primarily on synthetic data.",
    "A missing region does not override the V2 binary decision.",
    "This API does not assess makeup, eye enlargement, or product efficacy."
  ]
}
```

`polygon` 坐标已归一化到 `[0, 1]`。`region_status` 有三种取值：

- `localized`：至少返回一个受支持区域。
- `not_localized`：V2 判定为修图，但区域模型没有返回达到阈值的区域。
- `not_applicable`：V2 判定为未修图，下游输出被抑制。

## HTTP 错误码

| 状态码 | 含义 |
|---|---|
| `400` | 上传内容为空 |
| `413` | 图片超过 10 MiB |
| `422` | 图片无法读取或格式无效 |
| `503` | 模型权重缺失、不兼容或 SHA-256 校验失败 |

`GET /health` 只表示服务可访问，不代表三个模型都已完成推理自检。

## 模型文件

| 组件 | 路径 | 用途 |
|---|---|---|
| BeautyProof V2 | `models/retouch_detector_v2/best_model.pt` | 是否修图 |
| Three Type V1 | `models/retouch_three_type_v1/best_model.pt` | 三种类型与强度 |
| YOLO Regions V1 | `models/yolo_retouch_regions_v1/best.pt` | 修改区域定位 |

生产加载器会在反序列化前核验权重 SHA-256。各模型目录中的 `SHA256SUMS.txt`、`config.yaml` 和 `metrics.json` 分别记录校验值、推理配置与评估结果。

`models/retouch_multitask_cnn_v1/` 是旧版本类型模型，仅为版本追溯保留，统一 API 已不再加载该权重。

## 项目结构

```text
Mirror-of-Truth/
├── beautyproof_api/                  # 统一 Python、CLI 和 FastAPI 接口
│   ├── production.py                 # 权重加载与生产预测器
│   ├── unified.py                    # 决策编排与统一响应
│   ├── server.py                     # REST 服务
│   └── cli.py                        # 命令行入口
├── models/
│   ├── retouch_detector_v2/          # V2 二分类模型
│   ├── retouch_three_type_v1/        # 当前三类型模型
│   ├── yolo_retouch_regions_v1/      # 区域分割模型
│   └── retouch_multitask_cnn_v1/     # 历史类型模型
├── docs/                             # API、模型卡和工程接入文档
├── tests/                            # 单元测试与真实权重冒烟测试
├── demo/                             # Vue 3 + FastAPI 演示应用
└── requirements.txt
```

## 前后端演示

`demo/` 包含 Vue 3 前端和 FastAPI 后端。根目录统一 API 是模型能力的权威实现；Demo 还包含实验性 AI 图像检测和 Before/After 展示代码，不应与正式统一 API 的能力混淆。

Windows PowerShell：

```powershell
cd demo
npm install
npm run build
$env:BEAUTYPROOF_USE_UNIFIED = "1"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

Linux 或 macOS：

```bash
cd demo
npm install
npm run build
BEAUTYPROOF_USE_UNIFIED=1 python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。更多环境变量和实验路由参见 [`demo/README.md`](demo/README.md)。

## 测试与验收

```bash
python -m pytest tests -q
python -m pytest tests/test_unified_api.py -q
python -m pytest tests/test_production_smoke.py -q
python -m pytest demo/tests -q
```

## 已知限制

- 三类型分类和区域定位主要使用合成编辑数据，需要真实美颜软件配对数据进行外部验证。
- 单张图片无法直接知道人物未经处理时的自然脸型，因此瘦脸判断是概率性的。
- 自然侧光、轮廓光和妆容可能与五官立体效果相似。
- V2 的 Grad-CAM 曾出现关注头发、眼睛或中心面部结构的 shortcut 风险；Grad-CAM 只用于审计，不作为区域分割真值。
- YOLO 未返回区域不代表图片一定没有修图。
- API 不判断是否化妆，不检测放大眼睛，不证明产品功效。
- 完全 AI 生成图片检测仅存在于 Demo 的实验代码中，未并入根目录统一 API。

## 文档索引

- [统一 API 完整参考](docs/UNIFIED_BEAUTYPROOF_API.md)
- [三类型模型说明](docs/RETOUCH_THREE_TYPE_V1.md)
- [统一模型卡](docs/BEAUTYPROOF_UNIFIED_MODEL_CARD.md)
- [V2 模型详细文档](docs/BeautyProof_V2_Model_Documentation.md)
- [工程接入指南](docs/ENGINEERING_INTEGRATION.md)
- [API 接口说明](docs/API_INTERFACE.md)
- [Demo 使用说明](demo/README.md)

## 使用建议

产品界面应采用“疑似修图”“疑似磨皮美白”等概率性表述，同时展示置信度和限制说明。不要将输出描述为对图片真实性、人物自然外貌或产品效果的确定事实。
