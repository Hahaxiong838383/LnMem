"""VLM 云端标注器：给人脸帧生成 AU 描述 + valence/arousal。

优先级链（环境变量 VLM_PROVIDER 可强制指定）：
  1. gemini-3.1-flash-lite-preview（proxy，最新 3.x，速度最快，主力）
  2. gemini-3.1-pro-preview        （proxy，最高质量，~4.5s，2026-05-09 实测视觉 ✅）
  3. gemini-2.5-flash              （proxy，稳定备选）
  4. MiniMax-M2.7                  （直连，最后兜底，/v1/coding_plan/vlm 端点）

所有 API key 从环境变量读取，不硬编码。
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── 提示词 ─────────────────────────────────────────────────
_PROMPT = """你是FACS表情分析专家。分析这张人脸图片，识别激活的面部动作单元（AU），
直接输出JSON不要任何其他内容（不要markdown代码块）：
{"au_description":"激活的AU及强度描述，30字以内","valence":-0.5,"arousal":0.4,"confidence":0.85}

valence范围[-1,1]（正=积极，负=消极），arousal范围[0,1]（0=平静），confidence为识别置信度。"""

# ── 数据契约 ────────────────────────────────────────────────
@dataclass(frozen=True)
class VLMLabel:
    au_description: str
    valence: float
    arousal: float
    confidence: float
    model_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "au_description": self.au_description,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "confidence": round(self.confidence, 3),
            "model_used": self.model_used,
        }


# ── 主类 ────────────────────────────────────────────────────
class VLMAnnotator:
    """VLM 云端标注器（同步，PoC 用，0.1 次/秒完全够）。

    用法：
        annotator = VLMAnnotator.from_env()
        label = annotator.annotate(bgr_frame)
    """

    _PROXY_URL = "http://23.226.135.149:4000/v1/chat/completions"
    _MINIMAX_URL = "https://api.minimaxi.com/v1/chat/completions"

    # 优先级顺序：(model_id, api_type)
    # 2026-05-09 更新：pro-preview 视觉实测 4.5s，与 flash-lite 速度相当，质量更高
    _MODEL_CHAIN = [
        ("gemini-3.1-flash-lite-preview", "proxy"),   # 速度最快，主力
        ("gemini-3.1-pro-preview",        "proxy"),   # 最高质量，次选
        ("gemini-2.5-flash",              "proxy"),   # 稳定备选
        ("MiniMax-M2.7",                  "minimax"), # 最后兜底
    ]

    def __init__(
        self,
        proxy_key: str,
        minimax_key: str,
        timeout_s: int = 20,
        provider_override: str | None = None,
    ) -> None:
        self._proxy_key = proxy_key
        self._minimax_key = minimax_key
        self._timeout = timeout_s
        self._provider_override = provider_override  # e.g. "gemini-2.5-flash"

    @classmethod
    def from_env(cls) -> "VLMAnnotator":
        """从环境变量构建实例。

        必须设置：
          GEMINI_PROXY_KEY  — proxy Bearer token
          MINIMAX_API_KEY   — MiniMax API key
        可选：
          VLM_TIMEOUT       — 单次请求超时秒数（默认 20）
          VLM_PROVIDER      — 强制指定模型（跳过链式降级）
        """
        import os

        proxy_key = os.environ.get("GEMINI_PROXY_KEY", "")
        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        if not proxy_key and not minimax_key:
            raise EnvironmentError(
                "VLMAnnotator requires GEMINI_PROXY_KEY or MINIMAX_API_KEY in env"
            )
        return cls(
            proxy_key=proxy_key,
            minimax_key=minimax_key,
            timeout_s=int(os.environ.get("VLM_TIMEOUT", "20")),
            provider_override=os.environ.get("VLM_PROVIDER"),
        )

    # ── 公开 API ───────────────────────────────────────────

    def annotate(self, frame: Any) -> VLMLabel | None:
        """给一帧 BGR numpy array 或 JPEG 文件路径打标注。

        返回 VLMLabel，失败返回 None（不抛异常）。
        """
        img_b64 = self._encode_frame(frame)
        if not img_b64:
            return None

        chain = self._build_chain()
        for model_id, api_type in chain:
            try:
                raw = self._call(model_id, api_type, img_b64)
                if raw:
                    label = _parse_vlm_json(raw, model_id)
                    if label:
                        logger.info("VLM annotate OK model=%s val=%.2f aro=%.2f conf=%.2f",
                                    model_id, label.valence, label.arousal, label.confidence)
                        return label
                    logger.warning("VLM parse failed model=%s raw=%s", model_id, raw[:80])
            except Exception:
                logger.exception("VLM call error model=%s", model_id)
            logger.warning("VLM fallback: %s failed, trying next", model_id)

        logger.error("VLM all models failed")
        return None

    # ── 内部方法 ───────────────────────────────────────────

    def _build_chain(self) -> list[tuple[str, str]]:
        if self._provider_override:
            # 强制指定时确认 api_type
            for model_id, api_type in self._MODEL_CHAIN:
                if model_id == self._provider_override:
                    return [(model_id, api_type)]
            # 未知 model → 当 proxy 用
            return [(self._provider_override, "proxy")]
        return list(self._MODEL_CHAIN)

    def _call(self, model_id: str, api_type: str, img_b64: str) -> str | None:
        if api_type == "proxy":
            return self._call_proxy(model_id, img_b64)
        if api_type == "minimax":
            return self._call_minimax(img_b64)
        return None

    def _call_proxy(self, model_id: str, img_b64: str) -> str | None:
        if not self._proxy_key:
            return None
        body = {
            "model": model_id,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": _PROMPT},
            ]}],
        }
        status, resp = _post_json(
            self._PROXY_URL,
            {"Authorization": f"Bearer {self._proxy_key}"},
            body,
            self._timeout,
        )
        if status != 200:
            logger.warning("proxy %s HTTP %s: %s", model_id, status, str(resp)[:120])
            return None
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content or None

    def _call_minimax(self, img_b64: str) -> str | None:
        """MiniMax VLM 视觉接口。

        正确端点：/v1/coding_plan/vlm（非 chat/completions）
        M2.7 是纯文本模型，vision 走专属 VLM 端点，请求为扁平结构。
        响应在 resp["content"] 而非 choices[0].message.content。
        """
        if not self._minimax_key:
            return None
        body = {
            "prompt": _PROMPT,
            "image_url": f"data:image/jpeg;base64,{img_b64}",
        }
        status, resp = _post_json(
            "https://api.minimaxi.com/v1/coding_plan/vlm",
            {"Authorization": f"Bearer {self._minimax_key}"},
            body,
            self._timeout,
        )
        if status != 200:
            logger.warning("minimax vlm HTTP %s: %s", status, str(resp)[:120])
            return None
        base_resp = resp.get("base_resp", {})
        if base_resp.get("status_code", -1) != 0:
            logger.warning("minimax vlm base_resp error: %s", base_resp)
            return None
        content = resp.get("content", "")
        # 过滤 <think>...</think> reasoning trace
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content or None

    def _encode_frame(self, frame: Any) -> str | None:
        """BGR numpy array 或文件路径 → JPEG base64 字符串。"""
        if frame is None:
            return None
        if isinstance(frame, str):
            # 文件路径
            try:
                with open(frame, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except OSError:
                logger.error("VLMAnnotator: frame file not found: %s", frame)
                return None
        # numpy array (BGR)
        try:
            import cv2  # type: ignore
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode()
        except Exception:
            logger.exception("VLMAnnotator: frame encode failed")
            return None


# ── 工具函数 ────────────────────────────────────────────────

def _post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
) -> tuple[int, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={**headers, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        logger.warning("HTTP request failed url=%s err=%s", url, e)
        return 0, {}


def _parse_vlm_json(raw: str, model_id: str) -> VLMLabel | None:
    """从 LLM 回复中提取 JSON，容忍 markdown 代码块包裹。"""
    text = raw.strip()
    # 去 markdown ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # 找第一个完整 {} 块
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    try:
        return VLMLabel(
            au_description=str(d.get("au_description", "")),
            valence=float(d.get("valence", 0.0)),
            arousal=float(d.get("arousal", 0.5)),
            confidence=float(d.get("confidence", 0.7)),
            model_used=model_id,
        )
    except (TypeError, ValueError):
        return None


__all__ = ["VLMAnnotator", "VLMLabel"]
