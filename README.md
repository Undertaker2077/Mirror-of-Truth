# C 模块：文本核验

## 安装

```bash
pip install -r requirements.txt
cp .env.example .env   # Windows 可手动复制并重命名为 .env
# 在 .env 中填入 DEEPSEEK_API_KEY
```

## 调用方式

```python
from main_text_pipeline import run_text_pipeline

result = run_text_pipeline("图片路径")
```

## 返回字段说明

| 字段 | 说明 |
|------|------|
| `text_claims.ocr_text` | OCR 提取的原始文字 |
| `text_claims.local_claims` | 本地关键词匹配的宣称 |
| `text_claims.contradictions` | 矛盾点列表（LLM） |
| `text_claims.exaggerations` | 夸张话术列表（LLM） |
| `efficacy_evidence` | 产品库 RAG 匹配结果 |

## 本地测试

```bash
python main_text_pipeline.py
```
