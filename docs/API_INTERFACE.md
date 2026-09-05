# BeautyProof V2 图像鉴伪接口文档

## 1. 接口目的

本接口接收一张本地图片，调用 BeautyProof V2 二分类模型计算图片存在数字修图的可疑程度，生成 Grad-CAM 热力图，并返回供上游 A 模块直接消费的 `visual_evidence` JSON。

当前接口是 **Python 函数接口**，不是 HTTP API。

## 2. 环境要求

- Python 3.10 或更高版本
- PyTorch 2.4–2.x
- torchvision 0.19–0.x
- Pillow、NumPy、OpenCV

安装依赖：

```bash
pip install -r requirements.txt
```

默认模型权重查找顺序：

1. `./best_model.pt`
2. `./models/retouch_detector_v2/best_model.pt`

## 3. 主接口

```python
from model_inference import predict

result = predict(
    image_path="example.jpg",
    checkpoint_path=None,
    output_dir="outputs",
)
```

完整签名：

```python
def predict(
    image_path: str | pathlib.Path,
    *,
    checkpoint_path: str | pathlib.Path | None = None,
    output_dir: str | pathlib.Path = "outputs",
) -> dict[str, object]:
    ...
```

### 3.1 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `image_path` | `str \| Path` | 是 | 无 | 本地图片文件路径。图片必须能被 Pillow 解码；读取后统一转换为 RGB |
| `checkpoint_path` | `str \| Path \| None` | 否 | `None` | 指定 V2 checkpoint。为 `None` 时按默认顺序查找 |
| `output_dir` | `str \| Path` | 否 | `"outputs"` | 热力图输出目录；不存在时自动创建 |

建议输入 JPEG、PNG 或 WebP 人像图片。超大图片会在热力图输出阶段恢复到原图尺寸，因此调用方应自行设置文件大小和像素尺寸限制。

### 3.2 返回类型

返回一个可直接被 `json.dumps()` 序列化的 Python 字典。顶层固定包含 `visual_evidence`。

```json
{
  "visual_evidence": {
    "integrity_score": 0.73,
    "reliability": "Medium",
    "metrics": {
      "skin_smoothing": {
        "level": "Medium",
        "score": 0.73,
        "basis": "binary_score_proxy",
        "limitation": "Proxy derived from the V2 binary retouch score; not an independent detector for this manipulation type."
      },
      "texture_loss": {
        "level": "Medium",
        "score": 0.73,
        "basis": "binary_score_proxy",
        "limitation": "Proxy derived from the V2 binary retouch score; not an independent detector for this manipulation type."
      },
      "whitening": {
        "level": "Medium",
        "score": 0.73,
        "basis": "binary_score_proxy",
        "limitation": "Proxy derived from the V2 binary retouch score; not an independent detector for this manipulation type."
      }
    },
    "manipulation_map": "outputs/example_heatmap.png",
    "reliability_details": {
      "level": "Medium",
      "score": 0.71,
      "classification_confidence": 0.8,
      "map_concentration": 0.5,
      "method": "0.7 * classification_confidence + 0.3 * Grad-CAM concentration"
    },
    "model_version": "BeautyProof-V2",
    "threshold": 0.5,
    "limitations": [
      "Business metrics are proxies from one binary score.",
      "Grad-CAM is model attention, not pixel-level forensic ground truth.",
      "Do not use as the sole basis for legal, medical, hiring, or identity decisions."
    ]
  }
}
```

## 4. 输出字段定义

| 字段 | 类型 | 取值/范围 | 说明 |
|---|---|---|---|
| `visual_evidence` | `object` | 固定存在 | A 模块消费的证据对象 |
| `integrity_score` | `number` | `[0, 1]` | 数字修图可疑度，越高越可疑；等于 V2 sigmoid 概率 |
| `reliability` | `string` | `Low/Medium/High` | 本次结果的工程可靠性等级 |
| `metrics` | `object` | 固定包含三个键 | 业务指标集合 |
| `metrics.*.level` | `string` | `Low/Medium/High` | 对应业务指标等级 |
| `metrics.*.score` | `number` | `[0, 1]` | 当前为同一 V2 二分类分数 |
| `metrics.*.basis` | `string` | `binary_score_proxy` | 明确该指标为代理映射 |
| `metrics.*.limitation` | `string` | 非空 | 防止调用方把代理指标解释为独立检测结论 |
| `manipulation_map` | `string \| null` | 文件路径 | 叠加到原图尺寸的彩色 Grad-CAM PNG 路径 |
| `reliability_details` | `object` | 固定结构 | 可靠性计算细节 |
| `model_version` | `string` | `BeautyProof-V2` | 模型及接口版本标识 |
| `threshold` | `number` | `0.5` | 二分类判定阈值 |
| `limitations` | `array[string]` | 非空 | 必须向下游保留的解释限制 |

## 5. 等级映射规则

### 5.1 业务指标等级

| score 范围 | level |
|---|---|
| `score < 0.50` | `Low` |
| `0.50 <= score < 0.90` | `Medium` |
| `score >= 0.90` | `High` |

V2 只有一个二分类输出头，并不能独立区分磨皮、纹理损失和美白。三个指标均使用同一个 score 作为代理值，调用方必须读取并保留 `basis` 与 `limitation`。

### 5.2 可靠性等级

```text
classification_confidence = max(score, 1 - score)
reliability_score = 0.7 * classification_confidence
                  + 0.3 * map_concentration
```

| reliability_score 范围 | reliability |
|---|---|
| `< 0.65` | `Low` |
| `0.65–0.799999` | `Medium` |
| `>= 0.80` | `High` |

`map_concentration` 衡量 Grad-CAM 是否集中在少数区域。该方法参考 reliability map 的工程思路，但不是 TruFor 的实现或复现。

## 6. 底层接口

不需要生成 PNG、只需要数值矩阵的调用方可使用：

```python
from model_inference import predict_raw

raw = predict_raw("example.jpg", checkpoint_path="best_model.pt")
score = raw["score"]
manipulation_map = raw["manipulation_map"]  # NumPy float32 二维数组，范围 [0, 1]
```

`predict_raw()` 的返回值包含 NumPy 数组，不能未经转换直接执行 `json.dumps()`。跨服务传输时应优先使用主接口 `predict()` 返回的 PNG 路径。

## 7. A 模块消费示例

```python
from model_inference import predict


def collect_visual_evidence(image_path: str) -> dict:
    response = predict(image_path, output_dir="runtime/heatmaps")
    evidence = response["visual_evidence"]

    return {
        "source": "beautyproof_v2",
        "suspicious": evidence["integrity_score"] >= evidence["threshold"],
        "integrity_score": evidence["integrity_score"],
        "reliability": evidence["reliability"],
        "metrics": evidence["metrics"],
        "heatmap_path": evidence["manipulation_map"],
        "limitations": evidence["limitations"],
    }
```

调用方不应仅因 `reliability == "Low"` 删除结果；应展示为低可靠性证据，或交由人工复核。

## 8. 异常约定

当前 Python 接口直接抛出异常，不返回伪造的成功 JSON。

| 异常 | 典型原因 | 调用方处理建议 |
|---|---|---|
| `FileNotFoundError` | 输入图片或 checkpoint 不存在 | 返回参数错误，并记录解析后的文件路径 |
| `PIL.UnidentifiedImageError` | 文件不是有效图片或内容损坏 | 返回不支持/损坏图片错误 |
| `PermissionError` | 图片、权重或输出目录无权限 | 检查运行账户和挂载权限 |
| `RuntimeError` | 权重与网络结构不匹配、CUDA 或 OpenCV 执行失败 | 记录异常并降级为人工处理；不要输出默认分数 |
| `ValueError` | 分数或 manipulation map 不符合契约 | 视为程序错误并告警 |

推荐由服务层统一转换为以下失败结构；该结构不属于 `predict()` 的直接返回值：

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "The input file is not a decodable image.",
    "retryable": false
  }
}
```

## 9. 热力图文件管理

- 文件名为 `{输入文件名去扩展名}_heatmap.png`。
- 相同 `output_dir` 和相同文件名会覆盖旧热力图；并发服务应为每次请求使用唯一目录或请求 ID。
- 热力图包含原图内容，必须继承原图的隐私、访问控制和保留期限。
- Grad-CAM 表示模型关注区域，不是经过像素级篡改标签训练的分割结果。

## 10. 兼容性规则

当前契约版本：`BeautyProof-V2`。

- 下游必须忽略无法识别的新增字段，以支持向后兼容扩展。
- `visual_evidence`、`integrity_score`、`reliability`、`metrics` 和 `manipulation_map` 不得在 V2 中删除或改名。
- 字段类型、分数方向和 Low/Medium/High 大小写不得在 V2 中改变。
- 如需真正独立的三类操作分数、像素级篡改图或不同可靠性算法，应发布新的模型/接口版本。

## 11. 最小验收流程

```bash
pytest -q
```

然后选择一张已获授权的测试图片执行：

```python
from model_inference import predict

payload = predict("example.jpg")
assert 0.0 <= payload["visual_evidence"]["integrity_score"] <= 1.0
assert payload["visual_evidence"]["reliability"] in {"Low", "Medium", "High"}
assert payload["visual_evidence"]["manipulation_map"].endswith(".png")
```

