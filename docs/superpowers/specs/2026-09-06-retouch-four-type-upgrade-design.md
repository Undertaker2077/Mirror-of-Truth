# BeautyProof 四类美颜检测升级设计

## 1. 目标

在现有 BeautyProof 修图检测能力上新增一个可独立训练和接入统一 API 的多任务模型，识别以下四类效果：

1. `skin_enhancement`：磨皮与美白合并后的业务类别；
2. `face_slimming`：基于脸部轮廓内收的瘦脸；
3. `eye_enlargement`：基于双眼局部几何形变的眼睛放大；
4. `facial_contouring`：通过鼻梁、眉骨、面中提亮和鼻翼、眼窝、脸侧压暗形成的五官立体效果。

模型必须支持多标签输出，因为一张图片可能同时包含多种效果。现有 V2 仍是是否修图的权威检测器；新模型负责具体效果类型和强度，统一 API 负责组合结果。

## 2. 非目标

- 不检测人物是否化妆；
- 不识别具体化妆品；
- 不证明产品功效；
- 不把自然大眼、自然窄脸或自然侧光直接视作美颜；
- 不宣称仅由合成数据训练的模型能覆盖所有真实美颜 App。

## 3. 数据来源和划分

以 `makeup_pair.zip/original` 中通过质量检查的素颜人脸为基础。每个身份生成：

| 类型 | 每个身份数量 |
|---|---:|
| clean 原图 | 1 |
| skin_enhancement | 2 |
| face_slimming | 2 |
| eye_enlargement | 2 |
| facial_contouring | 2 |
| 随机多效果组合 | 2 |

按 500 个有效身份估算，共生成约 5,500 张图片。数据按 `person_id` 固定划分为 train/validation/test = 70%/15%/15%。同一身份的原图、所有派生图、mask 和配方只能属于同一个 split。

若有效身份数不是 500，保持每人 11 张的构成，报告实际身份数和样本数，不复制身份补足数量。

## 4. 预处理与质量门控

每张原图依次执行：

1. 检测主脸，仅接受一张清晰主脸；
2. 获取人脸关键点和 face parsing 区域；
3. 检查人脸尺寸、姿态、遮挡和关键点置信度；
4. 生成效果图、修改 mask、参数配方和标签；
5. 执行自动质量检查；
6. 生成分层人工复核包。

关键点或分割置信度不满足要求时跳过对应几何效果，不强行生成。失败记录写入审计表。

## 5. 效果生成

### 5.1 磨皮美白

只处理皮肤区域，保护眼睛、眉毛、嘴唇、鼻孔、牙齿和头发。磨皮采用保边滤波，美白在 Lab 或线性 RGB 空间完成。业务标签合并为 `skin_enhancement`，但 metadata 单独保存 `smoothing_strength` 和 `whitening_strength`。

强度范围：mild、medium、strong。磨皮和美白参数独立随机，避免模型把固定组合当作类别特征。

### 5.2 瘦脸

使用轮廓关键点和局部 TPS 或 piecewise-affine 网格形变，将左右脸颊和下颌向内移动。额头、眼睛、鼻子和背景保持稳定，强度限制为脸宽约 2%–10%。

自动拒绝出现明显背景弯曲、左右不对称、五官漂移或脸部边缘断裂的结果。保存原始与变形后关键点。

### 5.3 放大眼睛

围绕左右眼中心分别进行局部径向形变，范围覆盖眼裂和眼球附近区域，边缘平滑过渡。强度限制为约 2%–10%。不得用裁剪后直接缩放粘贴的方式生成。

### 5.4 五官立体

由关键点生成连续羽化的 dodge-and-burn mask：提亮鼻梁、眉骨、额头中央、眼下三角区和下巴中央；压暗鼻翼两侧、眼窝、颧骨下方和面部外轮廓。局部亮度变化控制在约 3%–15%，模板位置、宽度和左右差异均随机化。

## 6. 标签与目录

建议目录：

```text
retouch_four_type_v1/
├── images/{train,val,test}/
├── masks/{train,val,test}/
├── landmarks/
├── recipes/
├── metadata.csv
├── audit.csv
└── dataset.yaml
```

`metadata.csv` 至少包含：

```text
sample_id,person_id,image_path,source_image_path,
skin_enhancement,face_slimming,eye_enlargement,facial_contouring,
skin_enhancement_strength,face_slimming_strength,
eye_enlargement_strength,facial_contouring_strength,
modified_region_mask,landmarks_path,recipe_path,split
```

所有随机操作必须使用可记录的 seed；相同源图、recipe 和 seed 应生成相同结果。

## 7. 防止 shortcut 的约束

- 每类效果使用多个参数范围和模板变体；
- clean 图也进行与合成图一致的编码、resize 和保存流程；
- 加入 JPEG、缩放、曝光、去噪、自然侧光等 hard negatives；
- 加入自然大眼、自然窄脸和强轮廓光 clean 样本；
- 不允许按文件名、生成批次或身份泄漏标签；
- 测试集包含未参与参数选择的 recipe 和组合；
- 通过 Grad-CAM 和类别遮挡实验检查模型是否关注目标脸部区域。

## 8. 模型和训练

采用 EfficientNet-B0 多任务模型：

```text
backbone
├── retouched 二分类头
├── 四类 sigmoid 多标签分类头
├── 四类强度回归头
└── 可选修改区域辅助头
```

第一阶段先训练二分类、多标签和强度头；区域定位继续复用现有 YOLO/Grad-CAM。只有分类稳定后再决定是否增加分割头。

建议损失：

```text
0.35 * retouched BCE
+ 0.40 * multilabel focal BCE
+ 0.10 * strength SmoothL1
+ 0.15 * region Dice/BCE（启用区域头时）
```

训练参数：224×224、batch size 32（显存不足改为 16）、AMP、AdamW、初始学习率 3e-4、weight decay 1e-4、cosine scheduler、30–40 epochs、early stopping patience 7。先冻结 backbone 3 epochs，再解冻最后两个 stage。

每个类别根据验证集 PR 曲线独立选择阈值，不使用统一 0.5 阈值作为最终默认值。

## 9. 验收标准

| 项目 | 最低标准 |
|---|---:|
| 是否修图 ROC-AUC | ≥ 0.85 |
| 四类 macro F1 | ≥ 0.75 |
| skin_enhancement recall | ≥ 0.80 |
| face_slimming recall | ≥ 0.70 |
| eye_enlargement recall | ≥ 0.75 |
| facial_contouring recall | ≥ 0.70 |
| mild 效果 recall | ≥ 0.60 |
| clean 误报率 | ≤ 15% |
| 身份泄漏 | 0 |
| 人工复核合成通过率 | ≥ 90% |

测试必须分别报告单效果、组合效果、不同强度、不同身份和 hard-negative 指标。若某一类别未达到最低标准，统一 API 必须将其标为实验性或不对外输出。

## 10. 真实数据校准

在对外声称可检测真实美颜效果前，每类至少准备 50–100 张真实软件处理的配对图片，并覆盖至少两种不同软件或滤镜。真实数据优先作为独立测试集；需要微调时，必须保留一个从未参与训练的真实外部测试集。

## 11. 交付物

- 可复现的数据生成命令和配置；
- 生成数据、metadata、mask、recipe 与审计报告；
- 人工复核包；
- 训练脚本、配置、最佳权重和训练日志；
- 分类别评估、混淆分析、阈值和 shortcut 审计；
- 统一 API 的四类输出适配；
- 模型卡和数据集说明文档。
