# BeautyProof V2 工程集成说明

## 快速调用

仓库根目录的 `best_model.pt` 是默认 checkpoint。安装依赖后调用：

```python
from model_inference import predict

result = predict("example.jpg", output_dir="outputs")
print(result["visual_evidence"])
```

返回值为固定的 `visual_evidence` JSON；`integrity_score` 越高表示模型认为数字修图越可疑。`manipulation_map` 指向叠加到原图尺寸的彩色 Grad-CAM PNG。

## E1–E7 对应关系

| 验收项 | 实现 |
|---|---|
| E1 | `model_inference.predict_raw()` 返回 `score` 与二维 `manipulation_map`；`predict()` 返回可序列化结果 |
| E2 | `metric_mapper.map_business_metrics()` 将 score 映射为 Low/Medium/High |
| E3 | `heatmap_generator.generate_heatmap_overlay()` 输出红色高响应区域 PNG |
| E4 | `reports/hard_negative_evaluation.json` 固化总体和分组 FPR |
| E5 | `integrity_score` 等于 V2 的数字修图概率，范围 0–1 |
| E6 | 可靠性综合分类置信度（70%）与 Grad-CAM 集中度（30%） |
| E7 | `visual_evidence_schema.VisualEvidence` 提供稳定 JSON 合约 |

## 阈值依据

- Low：`score < 0.50`
- Medium：`0.50 <= score < 0.90`
- High：`score >= 0.90`

0.50 是 V2 冻结测试采用的分类阈值。0.90 用于避免将边界阳性描述为高强度；现有合成强度集的 score 中位数约为 Low 0.26、Medium 0.80、High 0.98。阈值是面向当前版本的工程分层，应在新增真实标注数据后重新校准。

## 重要限制

V2 只有一个二分类输出头。因此 `skin_smoothing`、`texture_loss` 和 `whitening` 当前都是同一 binary score 的代理展示，JSON 中通过 `basis: binary_score_proxy` 明示，不能解释为三个独立模型结论。

Grad-CAM 显示影响模型决策的区域，并非经过像素级篡改标注监督的真实性分割图。`reliability` 借鉴 reliability-map 的思路加入空间集中度，但不是 TruFor 的复现。

困难负样本总体 FPR 为 3.75%，达到 `<20%` 验收标准；柔光子集 FPR 为 0%。现有评测没有独立的“天然平滑皮肤”标签，因此不得宣称该子集已单独通过。

## 运行测试

```bash
pytest -q
```

测试覆盖阈值边界、JSON 合约、热力图尺寸与 Unicode 路径、非法分数和基础模型推理格式。

