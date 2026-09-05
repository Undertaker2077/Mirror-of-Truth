# BeautyProof Four-Type Retouch Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, evaluate, and publish a multi-label BeautyProof model for skin enhancement, face slimming, eye enlargement, and facial contouring.

**Architecture:** Extend the existing `retouch_multitask` pipeline with deterministic landmark-aware recipes and identity-safe dataset generation. Train an EfficientNet-B0 multi-task classifier for retouched/type/strength outputs, retain the existing V2 binary detector as the authoritative binary decision, and expose the new type model through the existing unified API.

**Tech Stack:** Python 3.12, PyTorch/torchvision, OpenCV, MediaPipe or the existing landmark backend, NumPy, Pillow, pandas, scikit-learn, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-retouch-four-type-upgrade-design.md`

## Global Constraints

- Public labels are exactly `skin_enhancement`, `face_slimming`, `eye_enlargement`, and `facial_contouring`.
- `skin_enhancement` combines smoothing and whitening for inference while retaining both recipe parameters in metadata.
- Split by `person_id` at 70%/15%/15%; one identity must occur in exactly one split.
- Generate one clean, eight single-effect, and two combination samples per accepted identity.
- Every generated sample must have a deterministic seed, recipe JSON, modification mask, and source reference.
- Reject unsafe geometry instead of generating low-quality slimming or eye samples.
- Do not expose a class in production unless it passes the per-class acceptance threshold.
- Existing V2 remains authoritative for the final binary `retouched` decision.

---

## File Structure

Training workspace (`D:/照妖镜model/BeautyProof`):

- `src/retouch_four_type/schema.py`: public labels, recipe and sample record validation.
- `src/retouch_four_type/landmarks.py`: normalized landmark interface and quality gates.
- `src/retouch_four_type/effects.py`: four deterministic image transforms.
- `src/retouch_four_type/builder.py`: identity-safe dataset generation and manifests.
- `src/retouch_four_type/audit.py`: leakage, balance, mask, artifact, and review-pack checks.
- `src/retouch_four_type/dataset.py`: PyTorch dataset and augmentations.
- `src/retouch_four_type/model.py`: EfficientNet-B0 multi-task network and loss.
- `src/retouch_four_type/train.py`: AMP training, early stopping, thresholds, checkpoints.
- `src/retouch_four_type/evaluate.py`: per-class, strength, combination, and hard-negative reports.
- `configs/retouch_four_type_v1.yaml`: reproducible paths, seeds, recipes, and hyperparameters.
- `scripts/build_retouch_four_type_v1.py`: dataset build CLI.
- `scripts/train_retouch_four_type_v1.py`: training CLI.
- `scripts/evaluate_retouch_four_type_v1.py`: evaluation CLI.
- `tests/test_four_type_*.py`: focused tests for every unit.

Published repository (`D:/照妖镜model/mirror_of_truth_publish`):

- `models/retouch_four_type_v1/best_model.pt`: accepted checkpoint.
- `models/retouch_four_type_v1/config.yaml`: inference configuration and thresholds.
- `models/retouch_four_type_v1/metrics.json`: acceptance evidence.
- `models/retouch_four_type_v1/SHA256SUMS.txt`: artifact integrity.
- `beautyproof_api/production.py`: production model adapter.
- `beautyproof_api/unified.py`: four-label unified response.
- `docs/RETOUCH_FOUR_TYPE_V1.md`: model card, training data, metrics, and limitations.

---

### Task 1: Define Stable Schemas and Identity Splits

**Files:**
- Create: `BeautyProof/src/retouch_four_type/__init__.py`
- Create: `BeautyProof/src/retouch_four_type/schema.py`
- Create: `BeautyProof/tests/test_four_type_schema.py`
- Modify: `BeautyProof/configs/retouch_four_type_v1.yaml`

**Interfaces:**
- Produces: `LABELS: tuple[str, ...]`, `EffectRecipe`, `SampleRecord`, `validate_record(record)`, `assign_identity_splits(person_ids, seed)`.
- Consumes: paths to accepted original images and stable `person_id` values.

- [ ] **Step 1: Write failing schema and split tests**

```python
def test_labels_are_stable():
    assert LABELS == ("skin_enhancement", "face_slimming", "eye_enlargement", "facial_contouring")

def test_identity_split_has_no_overlap():
    split = assign_identity_splits([f"p{i:03d}" for i in range(100)], seed=260906)
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert len(split["train"]) == 70
    assert len(split["val"]) == 15
```

- [ ] **Step 2: Verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_schema.py -q`

Expected: FAIL because `src.retouch_four_type.schema` does not exist.

- [ ] **Step 3: Implement immutable label and record schemas**

Use dataclasses with explicit validation: four binary labels, four strengths in `[0,1]`, non-empty source/recipe/mask paths, and split restricted to `train|val|test`. Implement seeded identity shuffling with exact 70/15/remainder allocation.

- [ ] **Step 4: Add the reproducible configuration**

Set `seed: 260906`, `image_size: 224`, `samples_per_identity: 11`, `split: [0.70, 0.15, 0.15]`, four public labels, and the strength ranges from the spec.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_schema.py -q`

Expected: PASS.

Commit: `git commit -am "feat: define four-type dataset schema"`

---

### Task 2: Build Landmark-Aware Effect Generators

**Files:**
- Create: `BeautyProof/src/retouch_four_type/landmarks.py`
- Create: `BeautyProof/src/retouch_four_type/effects.py`
- Create: `BeautyProof/tests/test_four_type_effects.py`
- Reuse: `BeautyProof/src/before_after/landmarks.py`
- Reuse: `BeautyProof/src/retouch_multitask/regions.py`

**Interfaces:**
- Produces: `FaceGeometry`, `detect_geometry(image) -> FaceGeometry | None`, `apply_effect(image, geometry, recipe) -> EffectResult`.
- `EffectResult` contains `image`, a binary `mask`, changed-pixel ratio, post-transform landmarks, and quality flags.

- [ ] **Step 1: Write failing deterministic and locality tests**

```python
def test_same_recipe_and_seed_are_deterministic(face_fixture):
    a = apply_effect(*face_fixture, EffectRecipe("eye_enlargement", 0.5, seed=7))
    b = apply_effect(*face_fixture, EffectRecipe("eye_enlargement", 0.5, seed=7))
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.mask, b.mask)

def test_skin_enhancement_does_not_modify_eye_mask(face_fixture):
    result = apply_effect(*face_fixture, EffectRecipe("skin_enhancement", 0.5, seed=8))
    assert result.mask[face_fixture[1].eye_mask].mean() < 0.01
```

- [ ] **Step 2: Verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_effects.py -q`

Expected: FAIL on missing interfaces.

- [ ] **Step 3: Implement geometry normalization and quality gates**

Wrap the existing landmark backend behind `detect_geometry`. Reject no-face, multiple-primary-face ambiguity, face width below 128 pixels, yaw above 30 degrees, pitch above 25 degrees, or landmark confidence below 0.8. Produce skin, eyes, facial oval, nose/bridge, brow, cheek, and jaw masks.

- [ ] **Step 4: Implement four effect transforms**

Implement protected-region guided smoothing plus Lab whitening; TPS/piecewise-affine jaw inward warping; radial eye warping; and randomized feathered dodge-and-burn contour masks. Calculate masks from actual pixel/geometry changes, not preset rectangles.

- [ ] **Step 5: Implement artifact rejection**

Reject geometry when background pixels outside the dilated face mask change by more than 0.5%, eye-center drift occurs during slimming, left/right deformation differs by more than 30%, or the modified pixel ratio is outside the configured range.

- [ ] **Step 6: Run tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_effects.py -q`

Expected: PASS.

Commit: `git commit -am "feat: add landmark-aware beauty effects"`

---

### Task 3: Generate the Reproducible Dataset

**Files:**
- Create: `BeautyProof/src/retouch_four_type/builder.py`
- Create: `BeautyProof/scripts/build_retouch_four_type_v1.py`
- Create: `BeautyProof/tests/test_four_type_builder.py`

**Interfaces:**
- Consumes: original image directory, config path, output directory.
- Produces: `build_dataset(config) -> BuildSummary`, `metadata.csv`, `audit.csv`, recipes, masks, landmarks, and split directories.

- [ ] **Step 1: Write failing miniature-build test**

```python
def test_builder_emits_eleven_records_per_valid_identity(tmp_path, three_faces):
    summary = build_dataset(test_config(three_faces, tmp_path))
    rows = pd.read_csv(tmp_path / "metadata.csv")
    assert summary.accepted_identities == 3
    assert len(rows) == 33
    assert rows.groupby("person_id")["split"].nunique().max() == 1
    assert rows.recipe_path.map(Path.exists).all()
    assert rows.modified_region_mask.map(Path.exists).all()
```

- [ ] **Step 2: Verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_builder.py -q`

Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement identity discovery and recipes**

Discover stable identity IDs from the source tree. For every accepted identity emit one clean sample, two strength-randomized samples per class, and two seeded combination samples. Encode and resize clean and edited images through the same function and JPEG quality distribution.

- [ ] **Step 4: Implement atomic output and resume behavior**

Write each sample to a temporary filename and rename only after image, mask, landmarks, and recipe succeed. On restart, verify the recipe hash before skipping a completed record. Log rejection code, source path, effect, and reason to `audit.csv`.

- [ ] **Step 5: Run miniature build tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_builder.py -q`

Expected: PASS and exactly 33 sample records.

Commit: `git commit -am "feat: generate four-type retouch dataset"`

- [ ] **Step 6: Build the full dataset**

Run:

```powershell
.venv\Scripts\python.exe scripts\build_retouch_four_type_v1.py `
  --config configs\retouch_four_type_v1.yaml `
  --output data\retouch_four_type_v1 `
  --workers 6
```

Expected: command exits 0 and reports accepted identities, generated samples, rejected samples, and output bytes. Do not assume exactly 5,500 images if input quality gates reject identities.

---

### Task 4: Audit Synthetic Quality and Shortcut Risk

**Files:**
- Create: `BeautyProof/src/retouch_four_type/audit.py`
- Create: `BeautyProof/tests/test_four_type_audit.py`
- Create: `BeautyProof/outputs/retouch_four_type_v1/review_pack/`

**Interfaces:**
- Produces: `audit_dataset(root) -> AuditReport`, `build_review_pack(root, count_per_class=30)`, and machine-readable `audit_report.json`.

- [ ] **Step 1: Write failing leakage, balance, and corruption tests**

```python
def test_audit_rejects_identity_leakage(valid_manifest):
    leaked = duplicate_identity_into_test(valid_manifest)
    assert "identity_leakage" in audit_manifest(leaked).fatal_errors

def test_audit_rejects_missing_mask(valid_manifest):
    valid_manifest.loc[0, "modified_region_mask"] = "missing.png"
    assert "missing_artifact" in audit_manifest(valid_manifest).fatal_errors
```

- [ ] **Step 2: Verify failure, implement audits, and rerun**

Check path existence, identity leakage, duplicates by perceptual hash, class/strength/split balance, mask area distribution, background-change rate, recipe diversity, and clean/edited encoding parity.

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_audit.py -q`

Expected: PASS.

- [ ] **Step 3: Create a stratified 150-image review pack**

Include 30 contact-sheet examples per class plus 30 combinations, always showing source, edited image, difference map, and mask. Sample mild/medium/strong and all splits; write reviewer decisions to `review.csv`.

- [ ] **Step 4: Enforce the human acceptance gate**

Require at least 90% accepted samples overall and at least 85% per class. Any failing class must be regenerated with corrected recipes before training.

- [ ] **Step 5: Commit audit code and reports**

Commit: `git commit -am "test: audit four-type synthetic dataset"`

---

### Task 5: Implement Multi-Task Dataset, Model, and Loss

**Files:**
- Create: `BeautyProof/src/retouch_four_type/dataset.py`
- Create: `BeautyProof/src/retouch_four_type/model.py`
- Create: `BeautyProof/tests/test_four_type_model.py`

**Interfaces:**
- Produces: `FourTypeDataset`, `FourTypeEfficientNet`, `compute_multitask_loss(outputs, targets, weights)`.
- Model outputs: `retouched_logit [B,1]`, `type_logits [B,4]`, `strength [B,4]`.

- [ ] **Step 1: Write failing tensor contract tests**

```python
def test_model_output_shapes():
    out = FourTypeEfficientNet(pretrained=False)(torch.randn(2, 3, 224, 224))
    assert out["retouched_logit"].shape == (2, 1)
    assert out["type_logits"].shape == (2, 4)
    assert out["strength"].shape == (2, 4)
    assert ((out["strength"] >= 0) & (out["strength"] <= 1)).all()
```

- [ ] **Step 2: Verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_model.py -q`

Expected: FAIL on missing model.

- [ ] **Step 3: Implement dataset and augmentations**

Load four binary labels and strengths from metadata. Apply identical color/compression augmentation policy to clean and edited images; allow horizontal flip with landmark/mask consistency; forbid elastic or face-shape transforms that alter the class semantics.

- [ ] **Step 4: Implement EfficientNet-B0 multi-task heads and masked strength loss**

Use ImageNet normalization. Compute SmoothL1 only where the corresponding type label is 1. Use class-balanced focal BCE derived from training-split prevalence. Return each loss component for logging.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_model.py -q`

Expected: PASS including a CPU forward/backward step with finite gradients.

Commit: `git commit -am "feat: add four-type multi-task model"`

---

### Task 6: Train, Select Thresholds, and Evaluate

**Files:**
- Create: `BeautyProof/src/retouch_four_type/train.py`
- Create: `BeautyProof/src/retouch_four_type/evaluate.py`
- Create: `BeautyProof/scripts/train_retouch_four_type_v1.py`
- Create: `BeautyProof/scripts/evaluate_retouch_four_type_v1.py`
- Create: `BeautyProof/tests/test_four_type_training.py`

**Interfaces:**
- Produces: resumable checkpoint, `thresholds.json`, `metrics.json`, predictions CSV, PR curves, confusion summaries, and Grad-CAM audit images.

- [ ] **Step 1: Write failing one-epoch resume test**

```python
def test_training_checkpoint_resumes(tmp_path, tiny_loaders):
    first = train(tiny_config(tmp_path, epochs=1), tiny_loaders)
    second = train(tiny_config(tmp_path, epochs=2, resume=first.last), tiny_loaders)
    assert second.start_epoch == 2
    assert Path(second.best).exists()
```

- [ ] **Step 2: Implement AMP training and checkpoint state**

Save model, optimizer, scheduler, scaler, epoch, best macro F1, config, label order, input normalization, and seed. Freeze backbone for epochs 1–3, unfreeze the last two stages at epoch 4, and stop after seven validation epochs without macro-F1 improvement.

- [ ] **Step 3: Implement per-class threshold selection**

Choose each threshold on the validation split by maximum F1 subject to precision at least 0.65. Never tune thresholds on test data.

- [ ] **Step 4: Implement complete evaluation**

Report ROC-AUC, AP, precision, recall, F1, and confusion counts per class; macro/micro aggregates; metrics by mild/medium/strong; single versus combination effects; clean false-positive rate; and hard-negative false-positive rates.

- [ ] **Step 5: Run training tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_four_type_training.py -q`

Expected: PASS on CPU and CUDA when available.

Commit: `git commit -am "feat: train and evaluate four-type model"`

- [ ] **Step 6: Train on the local GPU**

Run:

```powershell
.venv\Scripts\python.exe scripts\train_retouch_four_type_v1.py `
  --config configs\retouch_four_type_v1.yaml `
  --data data\retouch_four_type_v1\metadata.csv `
  --device cuda `
  --amp
```

Expected: CUDA device is the RTX 5070 Laptop GPU; training runs 30–40 epochs or stops early, with no NaN losses.

- [ ] **Step 7: Evaluate the best checkpoint**

Run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_retouch_four_type_v1.py `
  --checkpoint models\retouch_four_type_v1\best_model.pt `
  --data data\retouch_four_type_v1\metadata.csv `
  --split test
```

Expected: `metrics.json` explicitly marks each acceptance criterion pass/fail. Classes that fail are disabled in the production configuration.

---

### Task 7: Integrate, Document, Verify, and Publish

**Files:**
- Modify: `mirror_of_truth_publish/beautyproof_api/production.py`
- Modify: `mirror_of_truth_publish/beautyproof_api/unified.py`
- Modify: `mirror_of_truth_publish/tests/test_unified_api.py`
- Modify: `mirror_of_truth_publish/tests/test_production_smoke.py`
- Create: `mirror_of_truth_publish/docs/RETOUCH_FOUR_TYPE_V1.md`
- Create: `mirror_of_truth_publish/models/retouch_four_type_v1/*`

**Interfaces:**
- Produces: the existing `UnifiedBeautyProofAPI.analyze(path)` with four type records containing `type`, `detected`, `confidence`, `strength`, and `enabled`.
- Consumes: accepted checkpoint, label map, per-class thresholds, and SHA256 values.

- [ ] **Step 1: Write failing unified-response tests**

```python
def test_unified_api_returns_exact_four_type_order(fake_models, image_path):
    result = UnifiedBeautyProofAPI(models=fake_models).analyze(image_path)
    assert [x["type"] for x in result["retouch_types"]] == [
        "skin_enhancement", "face_slimming", "eye_enlargement", "facial_contouring"
    ]
```

- [ ] **Step 2: Verify failure and implement production adapter**

Load the checkpoint with `weights_only=True`, validate SHA256 before load, verify the embedded label order, preserve aspect-ratio preprocessing, apply per-class thresholds, and suppress any class disabled by acceptance results.

- [ ] **Step 3: Preserve V2 binary authority**

Do not replace `retouched.detected` with the new model output. If V2 says clean while a type head is positive, expose the type confidence but mark it `detected: false` and add a limitation message explaining the disagreement.

- [ ] **Step 4: Copy only accepted artifacts and document them**

Copy best checkpoint, config, thresholds, metrics, and checksums. Document dataset composition, synthetic generation, identity split, all test metrics, thresholds, disabled classes, known shortcut risks, and the need for real-app calibration.

- [ ] **Step 5: Run full verification**

Run in the training workspace:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Run in the publish repository using the same environment:

```powershell
D:\照妖镜model\BeautyProof\.venv\Scripts\python.exe -m pytest tests -q
D:\照妖镜model\BeautyProof\.venv\Scripts\python.exe -m compileall -q beautyproof_api
git diff --check
```

Expected: all tests pass, real-checkpoint smoke inference returns finite probabilities, compile succeeds, and diff check is clean.

- [ ] **Step 6: Commit and publish**

Commit: `git commit -m "feat: add four-type BeautyProof model"`

Push: `git push origin main`

Verify: local `git rev-parse HEAD` must exactly match `git ls-remote origin refs/heads/main`.

---

## Execution Checkpoints

1. Stop after Task 3 and report actual accepted identities, generated sample count, rejection causes, disk usage, and a five-image-per-class preview.
2. Do not start GPU training until Task 4 meets the 90% overall and 85% per-class human quality gates.
3. Stop after Task 6 and compare every acceptance threshold before changing the public API.
4. Publish only classes that pass acceptance; document failures rather than hiding them.
