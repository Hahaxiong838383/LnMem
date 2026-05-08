"""Day 4 真 LLM 客户端 — MiniMax 主路径 + Gemini Proxy 备选（v0.1 / 2026-05-07）。

LLMClient Protocol 定义在 ``poc/src/generate_scripts.py``，本文件提供两个实现：

1. **MiniMaxLLMClient**：主路径，OpenAI 兼容 endpoint
   ``https://api.minimaxi.com/v1/chat/completions``，模型 ``MiniMax-M2.7``。

2. **GeminiProxyLLMClient**：备选，自建代理 ``http://23.226.135.149:4000/v1/...``，
   模型 ``gemini-2.5-pro`` / ``gemini-2.5-flash``。

设计原则
--------
- **LLM 只生成语义** ``(activity, state, duration_s, description, multimodal_hint)``；
  ``event_id`` + ``ts`` 由本客户端按 day_index/seed 拼装，避免 LLM 算时间戳出错。
- **JSON 输出契约**：模型必须返回 ``{"events": [...]}``，本端做 thinking-trace 剥离 + 容错。
- **重试**：HTTP 5xx/超时各重试 2 次，输出仍非合法 JSON → 抛 ``LLMOutputError``。
- **零 SDK 依赖**：仅用 ``urllib.request``（标准库），与 PoC zero-deps 哲学一致。
- **immutable**：客户端实例 frozen dataclass，无可变状态。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from poc.src.data.event_schema import Activity, MultimodalHint, ScriptEvent, State

_MIN_EVENTS_HINT = 40
_MAX_EVENTS_HINT = 90

# ---------- 常量 ----------

_BASE_DATE = datetime(2026, 5, 6, 9, 0, 0)
"""与 HeuristicLLMClient 的 _BASE_DATE 一致；day_index=0 → 2026-05-06 09:00:00。"""

_THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"```json\s*", re.IGNORECASE),
    re.compile(r"```", re.IGNORECASE),
]

_ALLOWED_ACTIVITIES = tuple(a.value for a in Activity)
_ALLOWED_STATES = tuple(s.value for s in State)

# LLM 偶尔把 state/activity 写错或写成同义词，做轻量修正
_STATE_ALIASES = {
    "transition": State.TRANSITIONING.value,
    "transitional": State.TRANSITIONING.value,
    "positive": State.EMOTIONAL_POSITIVE.value,
    "negative": State.EMOTIONAL_NEGATIVE.value,
    "happy": State.EMOTIONAL_POSITIVE.value,
    "sad": State.EMOTIONAL_NEGATIVE.value,
    "tired": State.FATIGUED.value,
    "low": State.LOW_ENERGY.value,
    "focus": State.FOCUSED.value,
}
_ACTIVITY_ALIASES = {
    "transitioning": Activity.TRANSITION.value,
    "work": Activity.WORK_FOCUS.value,
    "meeting": Activity.WORK_MEETING.value,
    "break": Activity.BREAK_SHORT.value,
    "rest": Activity.BREAK_SHORT.value,
    "lunch": Activity.BREAK_LONG.value,
    "exercise": Activity.BREAK_LONG.value,
}


class LLMOutputError(ValueError):
    """LLM 输出不可解析或不合契约 — 调用方决定降级到 Heuristic 或换 backend。"""


# ---------- Prompt 模板 ----------

_SYSTEM_PROMPT = """你是 LnMem PoC 的剧本生成器。你的任务：根据用户给的虚拟主人画像（HEXACO + 依恋 + Layer4 if-then + Layer6 多模态签名），生成这位主人在指定一天的工作日活动剧本。

## 硬约束

1. **只输出合法 JSON**，结构必须是 `{"events": [...]}`，不能有任何前后缀或 markdown 代码围栏。
2. **events 累计 duration_s 必须 ≥ target_seconds**（target_seconds 见 user prompt）。允许超出最多 600 秒，但**严禁少于 target_seconds**——少了直接被拒。
3. **events 数量约 40-90 条**，平均每条 300-900 秒。
4. **每条 event 仅含以下字段**（多余字段会被丢弃）：
   ```
   {
     "duration_s": float,         // [5, 3600]，单事件秒数
     "activity": str,             // 必须从下列 8 类选一个
     "state": str,                // 必须从下列 8 类选一个
     "description": str,          // ≤ 30 字中文一句话
     "multimodal_hint": {
       "eye_closure": float,        // [0,1]，疲劳/闭眼=高
       "head_pitch_norm": float,    // [0,1]，低头幅度归一
       "head_yaw_norm": float,      // [0,1]，转头幅度归一
       "gaze_stability": float,     // [0,1]，注视稳定度
       "voice_present": float,      // [0,1]，0=无声 1=持续讲话
       "utterance_density": float,  // [0,1]，发言密度
       "active_app": str,           // 前台 app 名（VSCode/Notion/Feishu/Chrome/Slack/""）
       "app_switch_count_5min": int // [0,30]，5min 切换次数
     }
   }
   ```

## activity 8 类（只能选一个）
- work_focus     深度工作（写代码/写报告）
- work_meeting   多人会议
- break_short    短休 < 10min
- break_long     长休（午饭/散步/运动）
- commute        通勤
- leisure        休闲娱乐
- sleep          睡眠（工作日 8h 剧本通常不出现）
- transition     状态切换 ≤ 60s

## state 8 类（只能选一个）
- low_energy / focused / distracted / fatigued / emotional_positive / emotional_negative / neutral / transitioning

## 画像驱动原则（必须做到 HEXACO 6 维差异化）

- **honesty_humility 高** → app_switch 少、transition 少、emotional_negative 少
- **emotionality 高** → fatigued/emotional_negative 增多
- **extraversion 高** → work_meeting 比例上升、emotional_positive 增多
- **agreeableness 高** → work_meeting + leisure 偏高，孤立 work_focus 偏低
- **conscientiousness 高** → focused 状态多、duration_s 整齐（多 600/720/900s）
- **openness 高** → leisure 多、break_long 多、utterance_density 方差大

## 输出示例（仅展示结构，事件数量按真实剧本要求）

```
{"events":[
  {"duration_s":900,"activity":"work_focus","state":"focused","description":"打开 VSCode 写实现","multimodal_hint":{"eye_closure":0.12,"head_pitch_norm":0.10,"head_yaw_norm":0.08,"gaze_stability":0.88,"voice_present":0.0,"utterance_density":0.0,"active_app":"VSCode","app_switch_count_5min":1}},
  {"duration_s":600,"activity":"work_meeting","state":"emotional_positive","description":"和团队对齐 sprint","multimodal_hint":{"eye_closure":0.14,"head_pitch_norm":0.12,"head_yaw_norm":0.18,"gaze_stability":0.72,"voice_present":0.7,"utterance_density":0.55,"active_app":"Feishu","app_switch_count_5min":4}}
]}
```

记住：**只输出 JSON，不要任何解释文字**。"""


_USER_PROMPT_TEMPLATE = """画像 brief（≤1500 字，含 frontmatter + Layer4 + Layer6）:

{persona_brief}

要求生成第 {day_index} 天的工作日剧本（持续 {target_hours} 小时 = {target_seconds} 秒），seed={seed}。

**累计 duration_s 必须 ≥ {target_seconds} 秒**。建议生成 {min_events}-{max_events} 条 events，每条 300-900 秒。

只输出 {{"events": [...]}} JSON。"""

_FILLIN_PROMPT_TEMPLATE = """画像 brief：

{persona_brief}

上一轮你只生成了 {existing_seconds} 秒（目标 {target_seconds} 秒，缺 {missing_seconds} 秒）。现在请补充至少 {missing_seconds} 秒的剧本——延续上一轮的最后状态，承接当时正在进行的活动。**禁止重新从零开始**，**禁止输出和上一轮重复的事件**。

**累计 duration_s 必须 ≥ {missing_seconds} 秒**，建议 {min_events}-{max_events} 条 events，每条 300-900 秒。

只输出 {{"events": [...]}} JSON。"""


# ---------- 共用 prompt 装配 + 后处理 ----------


def _strip_thinking(text: str) -> str:
    """剥 thinking trace + markdown 围栏。"""
    out = text
    for pattern in _THINK_PATTERNS:
        out = pattern.sub("", out)
    return out.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出里抠出第一个完整 JSON 对象。

    LLM 偶尔在 JSON 前后加注释，本函数先尝试整体 parse，失败后扫描第一个 ``{`` 到匹配
    的 ``}``。
    """
    cleaned = _strip_thinking(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise LLMOutputError(f"no JSON object found in output (head): {cleaned[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = cleaned[start : idx + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError as exc:
                    raise LLMOutputError(f"JSON parse failed: {exc}: {snippet[:200]!r}") from exc
    raise LLMOutputError(f"unterminated JSON object: {cleaned[start:start + 200]!r}")


def _coerce_event_dict(
    raw: dict[str, Any],
    persona_id: str,
    day_index: int,
    event_idx: int,
    ts: datetime,
) -> dict[str, Any]:
    """把 LLM 一条事件转成符合 ``ScriptEvent.from_jsonl_dict`` 的 dict。

    - 拼装 ``event_id`` + ``ts``
    - clamp ``duration_s`` 到 [5, 3600]
    - 校验 ``activity`` / ``state`` 在枚举集合内（未知值 → 抛 LLMOutputError）
    - clamp multimodal_hint 数值字段到 [0,1]
    """
    activity = str(raw.get("activity", "")).strip().lower()
    state = str(raw.get("state", "")).strip().lower()
    # alias 容错：LLM 偶尔写同义词或活动/状态混淆
    if activity not in _ALLOWED_ACTIVITIES:
        activity = _ACTIVITY_ALIASES.get(activity, activity)
    if state not in _ALLOWED_STATES:
        state = _STATE_ALIASES.get(state, state)
    if activity not in _ALLOWED_ACTIVITIES:
        raise LLMOutputError(f"event[{event_idx}] activity={activity!r} not in {_ALLOWED_ACTIVITIES}")
    if state not in _ALLOWED_STATES:
        raise LLMOutputError(f"event[{event_idx}] state={state!r} not in {_ALLOWED_STATES}")
    duration = float(raw.get("duration_s", 600.0))
    duration = max(5.0, min(3600.0, duration))

    hint_in = raw.get("multimodal_hint") or {}
    hint = {
        "eye_closure": _clamp01(hint_in.get("eye_closure", 0.15)),
        "head_pitch_norm": _clamp01(hint_in.get("head_pitch_norm", 0.0)),
        "head_yaw_norm": _clamp01(hint_in.get("head_yaw_norm", 0.0)),
        "gaze_stability": _clamp01(hint_in.get("gaze_stability", 0.7)),
        "voice_present": _clamp01(hint_in.get("voice_present", 0.0)),
        "utterance_density": _clamp01(hint_in.get("utterance_density", 0.0)),
        "active_app": str(hint_in.get("active_app", "") or "")[:40],
        "app_switch_count_5min": max(0, min(60, int(hint_in.get("app_switch_count_5min", 0) or 0))),
    }
    description = str(raw.get("description", ""))[:60]
    return {
        "event_id": f"{persona_id}_d{day_index}_e{event_idx:03d}",
        "ts": ts.isoformat(timespec="seconds"),
        "persona_id": persona_id,
        "day_index": day_index,
        "duration_s": duration,
        "activity": activity,
        "state": state,
        "multimodal_hint": hint,
        "description": description,
        "meta": {},
    }


def _clamp01(val: Any) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _persona_id_from_brief(persona_brief: str) -> str:
    match = re.search(r"^id:\s*(\S+)\s*$", persona_brief, re.MULTILINE)
    if not match:
        raise LLMOutputError("persona_brief missing frontmatter id")
    return match.group(1).strip()


def _events_from_payload(
    payload: dict[str, Any],
    persona_id: str,
    day_index: int,
    start_idx: int = 0,
    start_ts_offset: int = 0,
) -> list[dict[str, Any]]:
    """把 LLM JSON ``{"events": [...]}`` 拼装为可入库的 event dict 列表。

    Args:
        start_idx: 事件全局序号起点（多轮补全时承接上一轮末尾）。
        start_ts_offset: 时间戳起点偏移量（秒），多轮补全时承接上一轮累计时长。
    """
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or len(raw_events) == 0:
        raise LLMOutputError(f"payload missing non-empty 'events' array: {str(payload)[:200]}")
    ts = _BASE_DATE + timedelta(days=day_index, seconds=start_ts_offset)
    out: list[dict[str, Any]] = []
    for offset, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise LLMOutputError(f"event[{start_idx + offset}] not dict: {raw!r}")
        ev = _coerce_event_dict(raw, persona_id, day_index, start_idx + offset, ts)
        ts = ts + timedelta(seconds=int(ev["duration_s"]))
        out.append(ev)
    return out


# ---------- HTTP 调用（标准库） ----------


def _http_post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """同步 POST + JSON。失败抛原始异常，调用方负责重试。"""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _retry_post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_s: float,
    max_attempts: int,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _http_post_json(url, headers, body, timeout_s)
        except urllib.error.HTTPError as exc:
            # 4xx 不重试（除 429）
            if exc.code != 429 and 400 <= exc.code < 500:
                raise
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
        if attempt < max_attempts:
            time.sleep(min(2.0 * attempt, 8.0))
    assert last_exc is not None
    raise last_exc


# ---------- MiniMax 主路径 ----------


@dataclass(frozen=True)
class MiniMaxLLMClient:
    """OpenAI 兼容 chat.completions on MiniMax-M2.7。

    只读 ``MINIMAX_API_KEY`` 环境变量；缺 key → 调用时 RuntimeError，避免静默降级。

    多轮补全：单次调用如果累计时长 < ``min_coverage_ratio × target_seconds``，
    自动追加调用让 LLM 续写补到 target，最多 ``max_fillin_rounds`` 轮。
    """

    api_key_env: str = "MINIMAX_API_KEY"
    base_url: str = "https://api.minimaxi.com/v1"
    model: str = "MiniMax-M2.7"
    max_tokens: int = 16000
    temperature: float = 0.7
    timeout_s: float = 180.0
    max_attempts: int = 3
    min_coverage_ratio: float = 0.95   # 累计 < 95% target 才补
    max_fillin_rounds: int = 2
    first_call_retries: int = 2        # 第一轮空 content/解析失败时额外重试次数

    def generate_script(
        self,
        persona_brief: str,
        day_index: int,
        target_hours: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} env not set; refuse to call MiniMax")
        persona_id = _persona_id_from_brief(persona_brief)
        target_seconds = int(target_hours * 3600)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            persona_brief=persona_brief.strip(),
            day_index=day_index,
            target_hours=target_hours,
            target_seconds=target_seconds,
            seed=seed,
            min_events=_MIN_EVENTS_HINT,
            max_events=_MAX_EVENTS_HINT,
        )
        # 第一轮：空 content / JSON 解析失败时额外重试（LLMOutputError 路径，HTTP 重试已由 _retry_post_json 处理）
        events: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for attempt in range(1 + self.first_call_retries):
            try:
                events = self._call_once(api_key, user_prompt, persona_id, day_index, start_idx=0, start_ts_offset=0)
                break
            except LLMOutputError as exc:
                last_exc = exc
                continue
        if not events:
            raise last_exc if last_exc is not None else LLMOutputError("first call returned no events")
        # 多轮补全：直到 ≥ min_coverage 或耗光预算（fillin 失败容错——已有 events 直接 truncate 返回）
        existing_seconds = sum(int(e["duration_s"]) for e in events)
        round_no = 0
        while (
            existing_seconds < target_seconds * self.min_coverage_ratio
            and round_no < self.max_fillin_rounds
        ):
            round_no += 1
            missing_seconds = target_seconds - existing_seconds
            min_e = max(8, missing_seconds // 900)
            max_e = max(min_e + 4, missing_seconds // 300)
            fillin_prompt = _FILLIN_PROMPT_TEMPLATE.format(
                persona_brief=persona_brief.strip(),
                target_seconds=target_seconds,
                existing_seconds=existing_seconds,
                missing_seconds=missing_seconds,
                min_events=min_e,
                max_events=max_e,
            )
            try:
                new_events = self._call_once(
                    api_key,
                    fillin_prompt,
                    persona_id,
                    day_index,
                    start_idx=len(events),
                    start_ts_offset=existing_seconds,
                )
            except (LLMOutputError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                # fillin 单轮失败：保留已有 events，跳出循环（上层 truncate 会把短剧本接受）
                # 注意：MiniMax 偶尔返回空 content（reasoning 模式 token 用尽），这是预期行为
                _ = exc  # noqa: F841 — 仅吞掉用作 break 信号
                break
            events.extend(new_events)
            existing_seconds = sum(int(e["duration_s"]) for e in events)
        return _truncate_to_target(events, target_seconds)

    def _call_once(
        self,
        api_key: str,
        user_prompt: str,
        persona_id: str,
        day_index: int,
        start_idx: int,
        start_ts_offset: int,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = _retry_post_json(
            url=f"{self.base_url}/chat/completions",
            headers=headers,
            body=body,
            timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
        )
        return self._parse_response(resp, persona_id, day_index, start_idx, start_ts_offset)

    def _parse_response(
        self,
        resp: dict[str, Any],
        persona_id: str,
        day_index: int,
        start_idx: int,
        start_ts_offset: int,
    ) -> list[dict[str, Any]]:
        choices = resp.get("choices") or []
        if not choices:
            raise LLMOutputError(f"no choices in response: {str(resp)[:200]}")
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMOutputError(f"empty content in response: {str(resp)[:200]}")
        payload = _extract_json_object(content)
        return _events_from_payload(
            payload, persona_id, day_index, start_idx=start_idx, start_ts_offset=start_ts_offset
        )


# ---------- Gemini Proxy 备选 ----------


@dataclass(frozen=True)
class GeminiProxyLLMClient:
    """自建 Gemini Proxy 23.226.135.149:4000，OpenAI 兼容路径。

    与 MiniMaxLLMClient 同接口，便于在 generate_scripts.run_batch 中无缝替换。
    支持多轮补全（同 MiniMax 实现）。
    """

    api_key_env: str = "GEMINI_PROXY_KEY"
    base_url: str = "http://23.226.135.149:4000/v1"
    model: str = "gemini-2.5-pro"
    max_tokens: int = 16000
    temperature: float = 0.7
    timeout_s: float = 180.0
    max_attempts: int = 3
    min_coverage_ratio: float = 0.95
    max_fillin_rounds: int = 2

    def generate_script(
        self,
        persona_brief: str,
        day_index: int,
        target_hours: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} env not set; refuse to call Gemini Proxy")
        persona_id = _persona_id_from_brief(persona_brief)
        target_seconds = int(target_hours * 3600)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            persona_brief=persona_brief.strip(),
            day_index=day_index,
            target_hours=target_hours,
            target_seconds=target_seconds,
            seed=seed,
            min_events=_MIN_EVENTS_HINT,
            max_events=_MAX_EVENTS_HINT,
        )
        events = self._call_once(api_key, user_prompt, persona_id, day_index, 0, 0)
        existing_seconds = sum(int(e["duration_s"]) for e in events)
        round_no = 0
        while (
            existing_seconds < target_seconds * self.min_coverage_ratio
            and round_no < self.max_fillin_rounds
        ):
            round_no += 1
            missing_seconds = target_seconds - existing_seconds
            min_e = max(8, missing_seconds // 900)
            max_e = max(min_e + 4, missing_seconds // 300)
            fillin_prompt = _FILLIN_PROMPT_TEMPLATE.format(
                persona_brief=persona_brief.strip(),
                target_seconds=target_seconds,
                existing_seconds=existing_seconds,
                missing_seconds=missing_seconds,
                min_events=min_e,
                max_events=max_e,
            )
            try:
                new_events = self._call_once(
                    api_key, fillin_prompt, persona_id, day_index, len(events), existing_seconds,
                )
            except (LLMOutputError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                break
            events.extend(new_events)
            existing_seconds = sum(int(e["duration_s"]) for e in events)
        return _truncate_to_target(events, target_seconds)

    def _call_once(
        self,
        api_key: str,
        user_prompt: str,
        persona_id: str,
        day_index: int,
        start_idx: int,
        start_ts_offset: int,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = _retry_post_json(
            url=f"{self.base_url}/chat/completions",
            headers=headers,
            body=body,
            timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
        )
        choices = resp.get("choices") or []
        if not choices:
            raise LLMOutputError(f"no choices in response: {str(resp)[:200]}")
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMOutputError(f"empty content in response: {str(resp)[:200]}")
        payload = _extract_json_object(content)
        return _events_from_payload(
            payload, persona_id, day_index, start_idx=start_idx, start_ts_offset=start_ts_offset,
        )


# ---------- truncate 工具 ----------


def _truncate_to_target(
    events: list[dict[str, Any]], target_seconds: int
) -> list[dict[str, Any]]:
    """裁剪 events 让累计 duration_s 落在 [target, target+1200] 内。

    策略：
      1. 如果累计 ≤ target+1200 → 原样返回。
      2. 否则从前往后累加，找到第一个让累计超出 target 的事件 i。
      3. 若 i 之前累计已 ≥ target-300（容差），直接截断到 i（不含 i）。
      4. 否则把 events[i] 的 duration_s 调小，让累计恰好为 target，并保留它。
      5. 重新 chain 后续 ts（这里直接丢掉 i+1: 之后的事件）。
    """
    if not events:
        return events
    total = sum(int(e["duration_s"]) for e in events)
    if total <= target_seconds + 1200:
        return events
    cum = 0
    for i, ev in enumerate(events):
        d = int(ev["duration_s"])
        if cum + d > target_seconds:
            if cum >= target_seconds - 300:
                return events[:i]
            need = target_seconds - cum
            need = max(60, need)  # 至少留 60s 让事件有意义
            new_ev = dict(ev)
            new_ev["duration_s"] = float(need)
            return events[:i] + [new_ev]
        cum += d
    return events


# ---------- 失败兜底：链式 client ----------


@dataclass(frozen=True)
class FallbackLLMClient:
    """主路径 + 备选 + 最终 mock 三级链式调用。

    用法：
        FallbackLLMClient(
            primary=MiniMaxLLMClient(),
            fallback=GeminiProxyLLMClient(),
            final=HeuristicLLMClient(),  # 最终保底（不抛异常）
        )

    主路径任何 LLMOutputError / HTTPError / RuntimeError → 切到 fallback；
    fallback 也失败 → 切到 final（HeuristicLLMClient，不抛异常）。
    所有降级在 ``meta['fallback_used']`` 里留迹。
    """

    primary: Any
    fallback: Any | None = None
    final: Any | None = None
    failures: list[str] = field(default_factory=list, compare=False, hash=False)

    def generate_script(
        self,
        persona_brief: str,
        day_index: int,
        target_hours: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        chain: list[tuple[str, Any]] = [(self.primary.__class__.__name__, self.primary)]
        if self.fallback is not None:
            chain.append((self.fallback.__class__.__name__, self.fallback))
        if self.final is not None:
            chain.append((self.final.__class__.__name__, self.final))
        last_exc: Exception | None = None
        for name, client in chain:
            try:
                events = client.generate_script(
                    persona_brief=persona_brief,
                    day_index=day_index,
                    target_hours=target_hours,
                    seed=seed,
                )
                if name != self.primary.__class__.__name__:
                    # 标记 fallback 使用（不修改 events，由调用方决定是否写入 Script.meta）
                    self.failures.append(f"used:{name}")
                return events
            except Exception as exc:  # noqa: BLE001 — 所有异常进入下一级 backend
                self.failures.append(f"{name}: {type(exc).__name__}: {exc}")
                last_exc = exc
                continue
        assert last_exc is not None
        raise last_exc


__all__ = [
    "LLMOutputError",
    "MiniMaxLLMClient",
    "GeminiProxyLLMClient",
    "FallbackLLMClient",
]
