<template>
  <div class="app" :style="{ '--center-column-height': centerColumnHeight }">
    <aside>
      <div class="brand">
        Beauty Trust Check
        <span>AI 图像鉴伪与妆效归因</span>
      </div>
      <div class="mode-list" role="tablist" aria-label="识别模式">
        <button
          v-for="mode in modeList"
          :key="mode.key"
          class="mode-button"
          :class="{ active: currentMode === mode.key }"
          type="button"
          @click="setMode(mode.key)"
        >
          <div class="mode-title">{{ mode.title }}</div>
          <div class="mode-caption">{{ mode.caption }}</div>
        </button>
      </div>
    </aside>

    <main ref="centerColumn">
      <div class="topbar">
        <div>
          <h1>{{ config.pageTitle }}</h1>
        </div>
        <div class="status-chip">{{ config.status }}</div>
      </div>

      <div class="workspace">
        <section class="panel">
          <div class="panel-header">
            <h2>{{ config.inputTitle }}</h2>
            <span class="status-chip">{{ inputCount }} / {{ config.slots }}</span>
          </div>
          <div class="panel-body">
            <div class="upload-grid" :class="{ single: config.slots === 1 }">
              <label class="dropzone">
                <span class="file-label">{{ config.labels[0] }}</span>
                <img
                  v-if="store.previews.A"
                  alt="上传图片预览"
                  class="preview"
                  :src="store.previews.A"
                />
                <div v-else class="empty">
                  <strong>选择图片</strong>
                  JPG / PNG / WebP
                </div>
                <input :key="`${currentMode}-A-${store.inputVersion}`" type="file" accept="image/*" @change="handleFile('A', $event)" />
              </label>

              <label v-if="config.slots === 2" class="dropzone">
                <span class="file-label">{{ config.labels[1] }}</span>
                <img
                  v-if="store.previews.B"
                  alt="上传图片预览"
                  class="preview"
                  :src="store.previews.B"
                />
                <div v-else class="empty">
                  <strong>选择 {{ config.labels[1] }}</strong>
                  JPG / PNG / WebP
                </div>
                <input :key="`${currentMode}-B-${store.inputVersion}`" type="file" accept="image/*" @change="handleFile('B', $event)" />
              </label>
            </div>

            <div class="actions">
              <label class="model-select">
                <span>AI模型</span>
                <select v-model="aiBackend">
                  <option value="hf3">AI / Deepfake / Real</option>
                  <option value="ultra">Lynote fallback</option>
                </select>
              </label>
              <button class="secondary" type="button" @click="resetCurrent">清空</button>
              <button class="secondary" type="button" @click="exitDetection">退出检测</button>
              <button class="primary" type="button" :disabled="!canAnalyze || loading" @click="analyze">
                {{ loading ? "检测中" : "开始检测" }}
              </button>
            </div>
          </div>
        </section>
      </div>
    </main>

    <aside class="result">
      <div class="score-box">
        <div class="score-row">
          <div>
            <div class="verdict" :class="verdictClass">{{ verdictText }}</div>
          </div>
          <div class="score" :class="scoreClass">{{ scoreText }}</div>
        </div>
        <div class="meter"><div class="meter-fill" :style="{ width: meterWidth }"></div></div>
      </div>

      <div class="result-section">
        <h3>关键证据</h3>
        <div v-if="!store.result && !store.error" class="placeholder">上传图片后生成检测结果。</div>
        <div v-else class="evidence-list">
          <div v-for="item in evidenceItems" :key="item" class="evidence">{{ item }}</div>
        </div>
      </div>

      <div class="result-section">
        <h3>风险标签</h3>
        <div class="tag-row">
          <span v-for="tag in config.risks" :key="tag" class="tag">{{ tag }}</span>
        </div>
      </div>

      <div class="result-section">
        <h3>热力图</h3>
        <img v-if="heatmapUrl" class="heatmap" alt="BeautyProof 热力图" :src="heatmapUrl" />
        <div v-else class="placeholder">上传图片后显示模型关注区域。</div>
      </div>

      <div v-if="alignedBeforeUrl || alignedAfterUrl" class="result-section">
        <h3>对齐图</h3>
        <div class="aligned-grid">
          <img v-if="alignedBeforeUrl" class="aligned-image" alt="对齐后的 Before" :src="alignedBeforeUrl" />
          <img v-if="alignedAfterUrl" class="aligned-image" alt="对齐后的 After" :src="alignedAfterUrl" />
        </div>
      </div>

      <div class="result-section">
        <h3>输出 JSON</h3>
        <div class="json-box">{{ jsonText }}</div>
      </div>
    </aside>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

const modes = {
  "makeup-single": {
    pageTitle: "妆造单图检测",
    title: "妆造单图",
    caption: "AI 生成、美颜、磨皮、P 图痕迹",
    status: "single image / makeup",
    inputTitle: "输入妆造图片",
    slots: 1,
    labels: ["Image"],
    apiMode: "makeup",
    risks: ["AI生成", "过度美颜", "磨皮遮瑕", "局部P图"],
  },
  "fashion-single": {
    pageTitle: "服装卖家秀检测",
    title: "服装卖家秀",
    caption: "AI 生成、模特异常、布料与背景一致性",
    status: "single image / fashion",
    inputTitle: "输入卖家秀图片",
    slots: 1,
    labels: ["Image"],
    apiMode: "fashion",
    risks: ["AI生成", "人体比例异常", "布料纹理异常", "背景一致性"],
  },
  "before-after": {
    pageTitle: "妆前妆后对比检测",
    title: "妆前妆后",
    caption: "美颜差异、局部美颜、光线与角度干扰",
    status: "before image + after image",
    inputTitle: "输入 Before / After",
    slots: 2,
    labels: ["Before", "After"],
    apiMode: "before_after",
    risks: ["美颜程度差异", "局部美颜", "曝光差异", "角度裁剪差异"],
  },
};

const modeList = Object.entries(modes).map(([key, value]) => ({ key, ...value }));
const currentMode = ref("makeup-single");
const aiBackend = ref("hf3");
const loading = ref(false);
const centerColumn = ref(null);
const centerColumnHeight = ref("100vh");
let centerResizeObserver = null;
let updateCenterHeight = () => {};
const stores = reactive(
  Object.fromEntries(
    Object.keys(modes).map((mode) => [
      mode,
      {
        files: { A: null, B: null },
        previews: { A: null, B: null },
        result: null,
        error: null,
        inputVersion: 0,
      },
    ]),
  ),
);

const config = computed(() => modes[currentMode.value]);
const store = computed(() => stores[currentMode.value]);
const inputCount = computed(() => Number(Boolean(store.value.files.A)) + Number(config.value.slots === 2 && Boolean(store.value.files.B)));
const canAnalyze = computed(() => inputCount.value === config.value.slots);
const activePayload = computed(() => store.value.result);
const activeModel = computed(() => activePayload.value?.model_output || activePayload.value?.after_model_output || {});
const activeVisual = computed(() => activePayload.value?.visual_evidence || {});
const activeComparison = computed(() => activePayload.value?.before_after_evidence || null);
const activeUnified = computed(
  () => activePayload.value?.beautyproof_unified?.result || activePayload.value?.after_beautyproof_unified?.result || null,
);
const confidence = computed(() => Math.round((activePayload.value?.false_advertising_confidence || 0) * 100));
const scoreText = computed(() => (activePayload.value ? `${confidence.value}%` : "--"));
const meterWidth = computed(() => (activePayload.value ? `${confidence.value}%` : "0"));
const verdictText = computed(() => {
  if (store.value.error) return "检测失败";
  if (!activePayload.value) return "待检测";
  if (activePayload.value.verdict === "High risk") return "高风险";
  if (activePayload.value.verdict === "Medium risk") return "中风险";
  return "低风险";
});
const scoreClass = computed(() => ({
  danger: confidence.value >= 70,
  warning: confidence.value >= 42 && confidence.value < 70,
  success: activePayload.value && confidence.value < 42,
}));
const verdictClass = scoreClass;
const heatmapUrl = computed(() => activeVisual.value?.manipulation_map_url || "");
const alignedBeforeUrl = computed(() => activeComparison.value?.aligned_before_url || "");
const alignedAfterUrl = computed(() => activeComparison.value?.aligned_after_url || "");
const evidenceItems = computed(() => {
  if (store.value.error) return [store.value.error];
  if (!activePayload.value) return [];
  const model = activeModel.value;
  const visual = activeVisual.value;
  const unified = activeUnified.value;
  const breakdown = activePayload.value.risk_breakdown?.inputs;
  const aiLabel = model.label === "ai" ? "是" : model.label === "real" ? "否" : "未知";
  const items = [];
  if (activePayload.value.mode === "before_after" && breakdown) {
    items.push(
      `Before AI概率 ${formatPercent(breakdown.before_ai_probability)}，After AI概率 ${formatPercent(breakdown.after_ai_probability)}，AI差距 ${formatPercent(breakdown.ai_probability_gap ?? breakdown.ai_probability_delta)}，${breakdown.ai_change_text || formatChangeText("AI概率", breakdown.after_ai_minus_before_ai)}。`,
    );
    items.push(
      `Before修图概率 ${formatPercent(breakdown.before_retouch_probability)}，After修图概率 ${formatPercent(breakdown.after_retouch_probability)}，美颜差距 ${formatPercent(breakdown.retouch_probability_delta)}，${breakdown.retouch_change_text || formatChangeText("修图概率", breakdown.after_retouch_minus_before_retouch)}。`,
    );
    items.push(
      `综合风险公式：美颜对比40% + After修图22% + After AI 15% + After AI增加5% + AI差异5% + 几何干扰10% + 条件干扰3%。`,
    );
  } else {
    items.push(
      `AI生成检测：${aiLabel}，AI概率 ${formatPercent(model.probability_ai)}，backend=${model.backend || "N/A"}，mock=${Boolean(model.mock)}。`,
      `BeautyProof修图检测：修图概率 ${formatPercent(visual.retouch_probability)}，可靠性 ${visual.reliability || "N/A"}，model=${visual.model_version || "N/A"}。`,
    );
  }
  items.push(...(activePayload.value.evidence || []));
  if (model.model_name && activePayload.value.mode !== "before_after") {
    items.splice(
      1,
      0,
      `AI模型：${model.model_name}，checkpoint=${model.checkpoint || "N/A"}，source=${model.source || "N/A"}。`,
    );
  }
  if (model.raw_label && activePayload.value.mode !== "before_after") {
    items.push(
      `三分类输出：${model.raw_label}，Artificial ${formatPercent(model.probability_artificial)}，Deepfake ${formatPercent(model.probability_deepfake)}，Real ${formatPercent(model.probability_real)}。`,
    );
  }
  if (unified) {
    items.push(
      `Unified结论：retouched=${Boolean(unified.retouched)}，strength=${unified.retouch_strength || "none"}，region_status=${unified.region_status || "N/A"}。`,
    );
    if (unified.retouch_types?.length) {
      items.push(`类型输出：${unified.retouch_types.map((item) => `${item.name} ${formatPercent(item.probability)}`).join("，")}。`);
    }
    if (unified.modified_regions?.length) {
      items.push(`区域输出：${unified.modified_regions.map((item) => `${item.name} ${formatPercent(item.confidence)}`).join("，")}。`);
    }
  }
  if (activePayload.value.before_after_evidence) {
    items.push(
      `前后对比：可靠性 ${activePayload.value.before_after_evidence.comparison_reliability}，曝光差异 ${activePayload.value.before_after_evidence.exposure_diff}。`,
    );
    items.push(
      `人脸对齐：${activePayload.value.before_after_evidence.alignment_status || "Unknown"}，中心偏移 ${activePayload.value.before_after_evidence.alignment_offset || "Unknown"}，角度差 ${activePayload.value.before_after_evidence.face_angle_diff_degrees ?? "Unknown"}°，裁剪 ${activePayload.value.before_after_evidence.crop_similarity || "Unknown"}。`,
    );
  }
  return items;
});
const jsonText = computed(() => JSON.stringify(store.value.error ? { error: store.value.error } : activePayload.value || {}, null, 2));

function setMode(mode) {
  currentMode.value = mode;
}

function handleFile(slot, event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const current = store.value;
  if (current.previews[slot]) URL.revokeObjectURL(current.previews[slot]);
  current.files[slot] = file;
  current.previews[slot] = URL.createObjectURL(file);
  current.result = null;
  current.error = null;
}

function resetCurrent() {
  const current = store.value;
  for (const slot of ["A", "B"]) {
    if (current.previews[slot]) URL.revokeObjectURL(current.previews[slot]);
    current.files[slot] = null;
    current.previews[slot] = null;
  }
  current.result = null;
  current.error = null;
  current.inputVersion += 1;
}

function exitDetection() {
  resetCurrent();
}

async function analyze() {
  const current = store.value;
  loading.value = true;
  current.error = null;
  try {
    let response;
    if (config.value.slots === 1) {
      const form = new FormData();
      form.append("image", current.files.A);
      form.append("mode", config.value.apiMode);
      form.append("backend", aiBackend.value);
      response = await fetch("/api/analyze/single", { method: "POST", body: form });
    } else {
      const form = new FormData();
      form.append("before_image", current.files.A);
      form.append("after_image", current.files.B);
      form.append("backend", aiBackend.value);
      response = await fetch("/api/analyze/before-after", { method: "POST", body: form });
    }
    if (!response.ok) throw new Error(`接口错误 ${response.status}`);
    current.result = await response.json();
  } catch (error) {
    current.error = error instanceof Error ? error.message : String(error);
  } finally {
    loading.value = false;
  }
}

function formatPercent(value) {
  if (typeof value !== "number") return "N/A";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatSignedPercent(value) {
  if (typeof value !== "number") return "N/A";
  const percentage = Math.round(value * 1000) / 10;
  return `${percentage >= 0 ? "+" : ""}${percentage}%`;
}

function formatChangeText(label, value) {
  if (typeof value !== "number") return `${label}变化未知`;
  const percentage = Math.abs(Math.round(value * 1000) / 10);
  if (percentage < 0.5) return `${label}基本一致`;
  return `After 比 Before ${value > 0 ? "高" : "低"} ${percentage}%`;
}

onMounted(() => {
  updateCenterHeight = () => {
    if (!centerColumn.value) return;
    centerColumnHeight.value = `${Math.ceil(centerColumn.value.offsetHeight)}px`;
  };
  updateCenterHeight();
  centerResizeObserver = new ResizeObserver(updateCenterHeight);
  centerResizeObserver.observe(centerColumn.value);
  window.addEventListener("resize", updateCenterHeight);
});

onBeforeUnmount(() => {
  if (centerResizeObserver) centerResizeObserver.disconnect();
  window.removeEventListener("resize", updateCenterHeight);
});
</script>

<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #172033;
  --muted: #667085;
  --line: #d7dce5;
  --ink: #111827;
  --red: #b42318;
  --red-bg: #fee4e2;
  --amber: #b54708;
  --amber-bg: #fef0c7;
  --green: #027a48;
  --green-bg: #d1fadf;
  --blue: #175cd3;
  --blue-bg: #d1e9ff;
  --shadow: 0 14px 40px rgba(16, 24, 40, 0.08);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
  background: var(--bg);
  color: var(--text);
  letter-spacing: 0;
}

.app {
  min-height: max(100vh, var(--center-column-height));
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 390px;
  align-items: stretch;
}

aside {
  background: #161b26;
  color: white;
  padding: 22px 18px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  min-height: max(100vh, var(--center-column-height));
}

.brand {
  font-size: 18px;
  font-weight: 700;
  line-height: 1.25;
}

.brand span {
  display: block;
  color: #a4bcfd;
  font-size: 12px;
  margin-top: 4px;
  font-weight: 400;
}

.mode-list {
  display: grid;
  gap: 8px;
}

.mode-button {
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
  color: white;
  text-align: left;
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  min-height: 72px;
}

.mode-button.active {
  background: #eff4ff;
  color: #172033;
  border-color: #84adff;
}

.mode-title {
  font-weight: 700;
  font-size: 14px;
  margin-bottom: 5px;
}

.mode-caption {
  color: inherit;
  opacity: .72;
  font-size: 12px;
  line-height: 1.35;
}

main {
  padding: 24px;
  min-width: 0;
}

.topbar {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  margin-bottom: 18px;
}

h1 {
  margin: 0;
  font-size: 24px;
  line-height: 1.2;
  color: var(--ink);
}

.status-chip {
  border: 1px solid var(--line);
  background: white;
  color: var(--muted);
  padding: 8px 10px;
  border-radius: 999px;
  font-size: 12px;
  white-space: nowrap;
}

.workspace {
  display: grid;
  gap: 16px;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.panel-header {
  padding: 16px 18px;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.panel-header h2 {
  margin: 0;
  font-size: 16px;
  color: var(--ink);
}

.panel-body {
  padding: 18px;
}

.upload-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.upload-grid.single {
  grid-template-columns: minmax(0, 1fr);
}

.dropzone {
  position: relative;
  min-height: 360px;
  border: 1px dashed #98a2b3;
  background: #fbfcfe;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dropzone input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.empty {
  text-align: center;
  color: var(--muted);
  padding: 24px;
}

.empty strong {
  display: block;
  color: var(--ink);
  font-size: 15px;
  margin-bottom: 6px;
}

.preview {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  background: #eef1f6;
}

.file-label {
  position: absolute;
  left: 10px;
  top: 10px;
  z-index: 2;
  background: rgba(17, 24, 39, .78);
  color: white;
  border-radius: 999px;
  padding: 6px 9px;
  font-size: 12px;
  max-width: calc(100% - 20px);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 14px;
}

.model-select {
  margin-right: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.model-select select {
  min-height: 40px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  color: var(--text);
  padding: 0 10px;
  font-weight: 700;
}

button.primary,
button.secondary {
  border: 1px solid transparent;
  border-radius: 8px;
  min-height: 40px;
  padding: 0 14px;
  font-weight: 700;
  cursor: pointer;
}

button.primary {
  background: #1d2939;
  color: white;
}

button.primary:disabled {
  background: #98a2b3;
  cursor: not-allowed;
}

button.secondary {
  background: white;
  color: #344054;
  border-color: var(--line);
}

.result {
  background: #ffffff;
  border-left: 1px solid var(--line);
  padding: 24px 20px;
  overflow-y: auto;
  height: max(100vh, var(--center-column-height));
  max-height: max(100vh, var(--center-column-height));
  color: var(--ink);
  align-self: start;
}

.score-box {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  background: white;
  box-shadow: var(--shadow);
}

.score-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
}

.score {
  font-size: 42px;
  line-height: 1;
  font-weight: 800;
  color: var(--red);
}

.score.warning { color: var(--amber); }
.score.success { color: var(--green); }

.verdict {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 12px;
  font-weight: 700;
  background: #eef1f6;
  color: #475467;
}

.verdict.danger {
  background: var(--red-bg);
  color: var(--red);
}

.verdict.warning {
  background: var(--amber-bg);
  color: var(--amber);
}

.verdict.success {
  background: var(--green-bg);
  color: var(--green);
}

.meter {
  height: 10px;
  background: #eef1f6;
  border-radius: 999px;
  overflow: hidden;
}

.meter-fill {
  height: 100%;
  width: 0;
  background: linear-gradient(90deg, #12b76a, #f79009, #d92d20);
  transition: width .35s ease;
}

.result-section {
  margin-top: 18px;
}

.result-section h3 {
  font-size: 14px;
  margin: 0 0 10px;
  color: var(--ink);
}

.evidence-list {
  display: grid;
  gap: 8px;
}

.evidence {
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
  color: var(--ink);
  font-size: 13px;
  line-height: 1.45;
}

.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.tag {
  border-radius: 999px;
  padding: 5px 8px;
  font-size: 12px;
  background: var(--blue-bg);
  color: var(--blue);
  font-weight: 700;
}

.json-box {
  white-space: pre-wrap;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
  background: #101828;
  color: #e4e7ec;
  border-radius: 8px;
  padding: 12px;
  max-height: 260px;
  overflow: auto;
}

.placeholder {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
  border: 1px dashed var(--line);
  border-radius: 8px;
  padding: 14px;
  background: #fbfcfe;
}

.heatmap {
  width: 100%;
  max-height: 260px;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
}

.aligned-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.aligned-image {
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
}

@media (max-width: 1080px) {
  .app {
    grid-template-columns: 1fr;
  }

  aside {
    position: static;
  }

  .mode-list {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .result {
    height: auto;
    max-height: none;
    border-left: 0;
    border-top: 1px solid var(--line);
  }
}

@media (max-width: 760px) {
  main {
    padding: 16px;
  }

  .mode-list,
  .upload-grid {
    grid-template-columns: 1fr;
  }

  .dropzone {
    min-height: 260px;
  }

  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
