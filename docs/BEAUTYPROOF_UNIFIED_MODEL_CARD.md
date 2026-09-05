# BeautyProof Unified Model Card

## Intended use

The pipeline screens face photographs for digital retouching, estimates smoothing/whitening, and provides a region hypothesis. It is intended for prototype review and hackathon demonstration, not high-stakes forensic adjudication.

## Components

| Component | Role | Held-out result |
|---|---|---|
| EfficientNet-B0 V2 | Authoritative binary decision | Accuracy 0.940, F1 0.941, AUROC 0.987 |
| Multitask CNN V1 | Smoothing/whitening and strength | Synthetic type Macro-F1 0.835; low-strength recall 0.929 |
| YOLO11n-Seg Regions V1 | Anatomical region hypothesis | Mask P 0.688, R 0.746, mAP50 0.689, mAP50-95 0.626 |

YOLO training used 30 epochs at image size 384 with 2,450 training and 525 validation images. It learned an experimental jawline class, but performance was inadequate; the public API removes jawline and does not claim slimming support.

## Supported outputs

- Binary digital-retouch decision and probability.
- Smoothing and whitening probabilities.
- Coarse strength: none, low, medium, or high.
- Forehead, cheek, nose, chin, and full-face polygons.

## Limitations

- Type and region results were validated primarily on synthetic controlled effects.
- Region confidence is not calibrated as legal or scientific evidence.
- V2 Grad-CAM and YOLO polygons have different meanings; neither proves which pixels were edited.
- Makeup, slimming, identity, demographic attributes, and product efficacy are out of scope.
- Do not use results as the sole basis for legal, medical, employment, insurance, or identity decisions.

## Checkpoint integrity

| Model | SHA-256 |
|---|---|
| V2 | `52F38353CEB4F20325B8AF84C0A0973FD48FEB57323B4429465D5C10FCFDC94D` |
| Type CNN V1 | `19F0D5F220A2DBE8C86E6959321FA87336475C99ABEE154E77579AC7841707BB` |
| YOLO Regions V1 | `DDD344917465425FFD15379DFC00324CFFA4126BDB41ECE4C0BBED7DF071CCDB` |

## Recommended next validation

Manually verify the review set and evaluate on at least two unseen real beauty applications. Report real and synthetic metrics separately.
