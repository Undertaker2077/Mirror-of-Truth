# BeautyProof Retouch Three-Type V1

## Capabilities

This EfficientNet-B0 multi-task checkpoint classifies three independently enabled retouch effects from one portrait:

- `skin_enhancement`: combined smoothing and whitening;
- `face_slimming`: face/jaw narrowing;
- `facial_contouring`: synthetic dodge-and-burn enhancement of facial depth.

The model does not replace BeautyProof V2. V2 remains authoritative for whether an image is retouched; this checkpoint enriches positive V2 decisions with effect types and continuous strengths.

## Dataset

The generator started from 499 detected identities. MediaPipe FaceMesh rejected invalid geometry and YOLO person counting removed 77 multi-person or overlapping-person identities. The accepted training manifest contains 421 identities and 4,631 images, split by identity into 3,311 train, 660 validation, and 660 test images.

Each identity contributes a clean image, two strengths for each synthetic effect, and two combined-effect images. The eye-enlargement training label remains in the internal checkpoint but is intentionally excluded from production output.

## Independent synthetic test results

| Type | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| skin_enhancement | 0.9524 | 1.0000 | 0.9756 | 0.9998 |
| face_slimming | 1.0000 | 0.9556 | 0.9773 | 0.9981 |
| facial_contouring | 0.9444 | 0.9444 | 0.9444 | 0.9917 |

Three-type macro F1 is 0.9658 on the identity-disjoint synthetic test set. These figures measure matching synthetic recipes and are not a claim of equivalent performance on arbitrary commercial beauty applications.

## Thresholds

- `skin_enhancement`: 0.9731336
- `face_slimming`: 0.7373627
- `facial_contouring`: 0.9494020

Thresholds were selected on validation data. The independent test split was not used for threshold selection.

## Limitations

- Real application filters may use transforms not represented by the synthetic recipes.
- Single-image face slimming is probabilistic; natural face shape is a hard negative.
- Natural directional lighting can resemble facial contouring.
- Makeup, eye enlargement, and product efficacy are outside the published contract.
- Region localization comes from the separate YOLO model and does not prove the exact operation that produced the image.
