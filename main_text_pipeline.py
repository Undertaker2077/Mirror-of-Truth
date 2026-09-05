"""
美妆文本核验流水线：OCR → 本地规则 → LLM 增强 → RAG 产品匹配
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# 本地规则兜底关键词
# ---------------------------------------------------------------------------
CLAIM_KEYWORDS = ["遮瑕", "持妆", "控油", "不卡粉"]

# ---------------------------------------------------------------------------
# 内置产品库（用于简单 RAG / 关键词匹配）
# ---------------------------------------------------------------------------
PRODUCT_DB: list[dict[str, Any]] = [
    {
        "name": "雅诗兰黛 DW",
        "aliases": ["雅诗兰黛", "DW", "Double Wear", "持妆粉底液"],
        "claims": ["遮瑕", "持妆", "控油", "全天持妆"],
        "efficacy": "高遮瑕长效持妆粉底，适合油皮与需全天遮盖的场景。",
    },
    {
        "name": "兰蔻持妆",
        "aliases": ["兰蔻", "持妆粉底", "兰蔻持妆轻透"],
        "claims": ["持妆", "轻透", "控油", "不卡粉"],
        "efficacy": "轻透持妆粉底，强调持久不暗沉与控油表现。",
    },
    {
        "name": "植村秀小方瓶",
        "aliases": ["植村秀", "小方瓶", "unlimited"],
        "claims": ["遮瑕", "持妆", "不卡粉", "轻薄"],
        "efficacy": "小方瓶粉底主打轻薄贴合与较长时间的持妆遮瑕。",
    },
    {
        "name": "NARS 超绒瓶",
        "aliases": ["NARS", "超绒瓶", "Light Reflecting"],
        "claims": ["遮瑕", "持妆", "自然光泽", "不卡粉"],
        "efficacy": "超绒瓶强调光泽提亮与自然遮瑕，适合追求妆感的场景。",
    },
    {
        "name": "阿玛尼权力",
        "aliases": ["阿玛尼", "权力粉底", "Power Fabric", "权力"],
        "claims": ["遮瑕", "持妆", "控油", "雾面"],
        "efficacy": "权力粉底主打雾面控油与高遮瑕持久覆盖。",
    },
]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
_ocr_reader = None


def get_ocr_reader():
    """懒加载 EasyOCR Reader（CPU 模式）。"""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
    return _ocr_reader


def extract_text_from_image(image_path: str) -> str:
    """使用 EasyOCR（CPU）从图片提取文字。"""
    if not image_path or not os.path.isfile(image_path):
        return ""
    reader = get_ocr_reader()
    results = reader.readtext(image_path, detail=0, paragraph=True)
    if isinstance(results, list):
        return "\n".join(str(r).strip() for r in results if str(r).strip())
    return str(results).strip()


# ---------------------------------------------------------------------------
# 本地规则：基础宣称提取
# ---------------------------------------------------------------------------
def extract_local_claims(text: str) -> list[str]:
    """根据关键词列表提取基础宣称。"""
    if not text:
        return []
    found: list[str] = []
    for kw in CLAIM_KEYWORDS:
        if kw in text and kw not in found:
            found.append(kw)
    return found


# ---------------------------------------------------------------------------
# LLM 增强（DeepSeek via openai SDK）
# ---------------------------------------------------------------------------
def call_llm(text: str) -> dict[str, Any]:
    """
    调用 DeepSeek API，返回 contradictions / exaggerations。
    Key 为空或调用失败时降级为空列表，不中断程序。
    额外返回 llm_status / llm_error 便于排查配置问题。
    """
    empty: dict[str, Any] = {
        "contradictions": [],
        "exaggerations": [],
        "llm_status": "skipped",
        "llm_error": None,
    }
    api_key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        empty["llm_status"] = "no_key"
        return empty
    if not text.strip():
        return empty

    prompt = f"""你是美妆广告合规审核助手。请分析以下文本中的功效宣称问题。

要求：
1. 找出相互矛盾的宣称（contradictions）
2. 找出夸张、绝对化或不科学的表述（exaggerations）
3. 必须只返回合法 JSON，不要 markdown，不要其它说明文字
4. JSON 格式严格为：
{{"contradictions": ["..."], "exaggerations": ["..."]}}

待分析文本：
{text}
"""

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.openai-next.com/v1")
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": "你只输出 JSON 对象，字段为 contradictions 与 exaggerations，值均为字符串数组。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = (response.choices[0].message.content or "").strip()
        # 容忍模型偶尔包一层 ```json
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        return {
            "contradictions": list(data.get("contradictions") or []),
            "exaggerations": list(data.get("exaggerations") or []),
            "llm_status": "ok",
            "llm_error": None,
        }
    except Exception as exc:
        empty["llm_status"] = "error"
        empty["llm_error"] = str(exc)
        return empty


# ---------------------------------------------------------------------------
# 简易 RAG：产品库匹配
# ---------------------------------------------------------------------------
def rag_match_products(text: str, claims: list[str]) -> list[dict[str, Any]]:
    """根据 OCR 文本与宣称关键词，在内置产品库中做匹配打分。"""
    if not text and not claims:
        return []

    text_lower = text.lower()
    matched: list[dict[str, Any]] = []

    for product in PRODUCT_DB:
        score = 0
        hit_aliases: list[str] = []
        hit_claims: list[str] = []

        for alias in product["aliases"]:
            if alias.lower() in text_lower or alias in text:
                score += 3
                hit_aliases.append(alias)

        for claim in product["claims"]:
            if claim in text or claim in claims:
                score += 1
                hit_claims.append(claim)

        if score > 0:
            matched.append(
                {
                    "product": product["name"],
                    "score": score,
                    "matched_aliases": hit_aliases,
                    "matched_claims": sorted(set(hit_claims)),
                    "efficacy": product["efficacy"],
                }
            )

    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------
def run_text_pipeline(image_path: str) -> dict[str, Any]:
    """
    美妆文本核验主流程。
    返回包含 text_claims 与 efficacy_evidence 等字段的汇总字典。
    """
    ocr_text = extract_text_from_image(image_path)
    local_claims = extract_local_claims(ocr_text)
    llm_result = call_llm(ocr_text)
    rag_results = rag_match_products(ocr_text, local_claims)

    text_claims = {
        "ocr_text": ocr_text,
        "local_claims": local_claims,
        "contradictions": llm_result.get("contradictions", []),
        "exaggerations": llm_result.get("exaggerations", []),
        "llm_status": llm_result.get("llm_status"),
        "llm_error": llm_result.get("llm_error"),
    }

    efficacy_evidence = [
        {
            "product": item["product"],
            "score": item["score"],
            "matched_claims": item["matched_claims"],
            "evidence": item["efficacy"],
        }
        for item in rag_results
    ]

    return {
        "text_claims": text_claims,
        "efficacy_evidence": efficacy_evidence,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    # 优先使用命令行或当前目录下的测试图；没有图片时用模拟文本走通非 OCR 逻辑
    demo_image = os.environ.get("DEMO_IMAGE", "demo_makeup.jpg")

    print("=" * 60)
    print("美妆文本核验流水线 Demo")
    print("=" * 60)

    if os.path.isfile(demo_image):
        print(f"[OCR] 读取图片: {demo_image}")
        result = run_text_pipeline(demo_image)
    else:
        print(f"[提示] 未找到图片 {demo_image}，使用模拟 OCR 文本演示后续流程。")
        mock_text = (
            "全新雅诗兰黛DW粉底液，超强遮瑕一整天持妆不脱妆，"
            "同时做到轻薄不卡粉、控油又水润，绝对零瑕疵完美妆效！"
        )

        local_claims = extract_local_claims(mock_text)
        llm_result = call_llm(mock_text)
        rag_results = rag_match_products(mock_text, local_claims)

        result = {
            "text_claims": {
                "ocr_text": mock_text,
                "local_claims": local_claims,
                "contradictions": llm_result.get("contradictions", []),
                "exaggerations": llm_result.get("exaggerations", []),
                "llm_status": llm_result.get("llm_status"),
                "llm_error": llm_result.get("llm_error"),
            },
            "efficacy_evidence": [
                {
                    "product": item["product"],
                    "score": item["score"],
                    "matched_claims": item["matched_claims"],
                    "evidence": item["efficacy"],
                }
                for item in rag_results
            ],
        }

    pprint.pprint(result, width=100, sort_dicts=False)
    print("=" * 60)

    llm_status = result.get("text_claims", {}).get("llm_status")
    llm_error = result.get("text_claims", {}).get("llm_error")
    if llm_status == "ok":
        print("完成。LLM 检测已启用并成功返回结果。")
    elif llm_status == "no_key":
        print("完成。未检测到 API Key，请在项目根目录创建 .env 文件（注意文件名以点开头），内容为：")
        print("  DEEPSEEK_API_KEY=你的密钥")
    elif llm_status == "error":
        print("完成。已读取 API Key，但 DeepSeek 调用失败，请检查 Key 是否有效、账户是否有余额：")
        print(f"  {llm_error}")
    else:
        print("完成。")