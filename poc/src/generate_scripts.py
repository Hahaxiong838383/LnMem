"""Day 3 script generation pipeline with deterministic local LLM mock."""
from __future__ import annotations

import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import yaml

from poc.src.data.event_schema import Activity, MultimodalHint, Script, ScriptEvent, State
from poc.src.data.script_validator import validate_script

_BRIEF_MAX_CHARS = 1500
_BASE_DATE = datetime(2026, 5, 6, 9, 0, 0)


class LLMClient(Protocol):
    """LLM call interface; 测试注入 mock、运行期可替换真实后端。"""

    def generate_script(
        self,
        persona_brief: str,
        day_index: int,
        target_hours: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        """Return ScriptEvent dict list; 返回兼容 ScriptEvent.from_jsonl_dict 的字典序列。"""
        ...


class HeuristicLLMClient:
    """Deterministic heuristic generator; 6维HEXACO jointly drives activity/state/hints."""

    def generate_script(
        self,
        persona_brief: str,
        day_index: int,
        target_hours: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        """Generate synthetic event dicts; 生成满足事件契约的确定性 dict 列表。"""
        profile = _profile_from_brief(persona_brief)
        persona_id = profile["persona_id"]
        rng = random.Random(seed + day_index * 1009 + _stable_text_hash(persona_id))
        total_target_s = int(target_hours * 3600)
        current_ts = _BASE_DATE + timedelta(days=day_index)
        total_s = 0
        event_idx = 0
        long_break_used = False
        events: list[dict[str, Any]] = []

        while total_s < total_target_s:
            remaining_s = total_target_s - total_s
            elapsed_s = total_s
            activity = _pick_activity(profile, rng, elapsed_s, long_break_used)
            if activity == Activity.BREAK_LONG:
                long_break_used = True
            state = _pick_state(profile, activity, rng)
            duration_s = _pick_duration(profile, activity, remaining_s, rng)
            hint = _hint_from_state(profile, state, activity, rng)
            event = ScriptEvent(
                event_id=f"{persona_id}_d{day_index}_e{event_idx:03d}",
                ts=current_ts.isoformat(timespec="seconds"),
                persona_id=persona_id,
                day_index=day_index,
                duration_s=float(duration_s),
                activity=activity,
                state=state,
                multimodal_hint=hint,
                description=f"{profile['name']}进行{activity.value}，状态{state.value}",
                meta={},
            )
            events.append(event.to_jsonl_dict())
            current_ts += timedelta(seconds=duration_s)
            total_s += duration_s
            event_idx += 1
        return events


def load_persona_brief(persona_path: Path) -> str:
    """Load persona brief text; 提取 frontmatter + Layer4 + Layer6 到 <=1500 字上下文。"""
    text = persona_path.read_text(encoding="utf-8")
    fm_text, fm, body = _split_frontmatter(text)
    layer4 = _extract_layer_section(body, 4)
    layer6 = _extract_layer_section(body, 6)
    brief = (
        f"[persona_path]\n{persona_path.as_posix()}\n\n"
        f"[frontmatter]\n{fm_text.strip()}\n\n"
        f"[layer4_if_then]\n{layer4.strip()}\n\n"
        f"[layer6_multimodal]\n{layer6.strip()}\n"
    )
    brief = brief.strip()
    if len(brief) > _BRIEF_MAX_CHARS:
        brief = brief[:_BRIEF_MAX_CHARS]
    if "hexaco" not in brief.lower():
        brief = f"{brief}\nhexaco: {fm.get('hexaco', {})}"
    return brief


def generate_day_script(
    persona_path: Path,
    day_index: int,
    llm: LLMClient,
    target_hours: float = 8.0,
    seed: int = 0,
) -> Script:
    """Generate one day script; 生成并校验单画像单天 Script，对 error 直接失败。"""
    brief = load_persona_brief(persona_path)
    runtime_llm: LLMClient = llm
    if os.getenv("LNMEM_OFFLINE") == "1":
        runtime_llm = HeuristicLLMClient()

    raw_events = runtime_llm.generate_script(
        persona_brief=brief,
        day_index=day_index,
        target_hours=target_hours,
        seed=seed,
    )
    fm = _profile_from_brief(brief)["frontmatter"]
    persona_id = str(fm["id"])
    events = tuple(ScriptEvent.from_jsonl_dict(d) for d in raw_events)
    script = Script(
        persona_id=persona_id,
        day_index=day_index,
        generated_at=datetime.now().replace(microsecond=0).isoformat(),
        events=events,
        meta={
            "llm_class": runtime_llm.__class__.__name__,
            "seed": seed,
            "target_hours": target_hours,
        },
    )
    report = validate_script(script)
    if not report.passed:
        details = "; ".join(f"{i.code}:{i.message}" for i in report.issues if i.severity == "error")
        raise ValueError(f"script validation failed for {persona_id} day{day_index}: {details}")
    return script


def write_script(script: Script, out_dir: Path) -> Path:
    """Write script JSON; 按约定文件名落盘并返回输出路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"script_{script.persona_id}_day{script.day_index}.json"
    out_path.write_text(script.to_json(indent=2), encoding="utf-8")
    return out_path


def run_batch(
    persona_paths: list[Path],
    day_index: int,
    llm: LLMClient,
    out_dir: Path,
    seed: int = 0,
) -> list[Path]:
    """Run batch generation; 批量生成、逐个校验后落盘并返回路径列表。"""
    outputs: list[Path] = []
    for idx, persona_path in enumerate(persona_paths):
        script = generate_day_script(
            persona_path=persona_path,
            day_index=day_index,
            llm=llm,
            target_hours=8.0,
            seed=seed + idx,
        )
        outputs.append(write_script(script, out_dir))
    return outputs


def _stable_text_hash(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text))


def _split_frontmatter(text: str) -> tuple[str, dict[str, Any], str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError("persona markdown missing YAML frontmatter")
    fm_text = match.group(1)
    body = match.group(2)
    fm = yaml.safe_load(fm_text) or {}
    return fm_text, fm, body


def _extract_layer_section(body: str, layer_no: int) -> str:
    pattern = rf"##\s*Layer\s*{layer_no}[^\n]*\n(.*?)(?=\n##\s*Layer\s*\d+|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    return match.group(1).strip() if match else ""


def _profile_from_brief(persona_brief: str) -> dict[str, Any]:
    """Parse profile with HEXACO 6D; H/E/X/A/C/O -> consistency/emotion/social/co-op/focus/exploration."""
    match = re.search(r"\[frontmatter\]\n(.*?)(?:\n\[layer4_if_then\]|\Z)", persona_brief, re.DOTALL)
    if not match:
        raise ValueError("brief frontmatter block missing")
    fm = yaml.safe_load(match.group(1).strip()) or {}
    if "id" not in fm:
        raise ValueError("frontmatter.id missing in brief")
    hexaco = fm.get("hexaco") or {}
    attachment = fm.get("attachment") or {}
    density = str(fm.get("multimodal_signal_density", "medium"))
    profession = str((fm.get("demographics") or {}).get("profession", ""))
    return {
        "frontmatter": fm,
        "persona_id": str(fm["id"]),
        "name": str(fm.get("name", fm["id"])),
        "profession": profession,
        "density": density,
        "hexaco": {
            "honesty_humility": float(hexaco.get("honesty_humility", 0.5)),
            "emotionality": float(hexaco.get("emotionality", 0.5)),
            "extraversion": float(hexaco.get("extraversion", 0.5)),
            "agreeableness": float(hexaco.get("agreeableness", 0.5)),
            "conscientiousness": float(hexaco.get("conscientiousness", 0.5)),
            "openness": float(hexaco.get("openness", 0.5)),
        },
        "attachment_anxiety": float(attachment.get("anxiety", 0.5)),
        "commute_minutes": int((fm.get("demographics") or {}).get("commute_minutes", 0)),
    }


def _weighted_choice(weights: dict[Any, float], rng: random.Random) -> Any:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        raise ValueError("weighted choice has non-positive total")
    pick = rng.random() * total
    cursor = 0.0
    for key, val in weights.items():
        if val <= 0:
            continue
        cursor += val
        if pick <= cursor:
            return key
    return next(iter(weights))


def _pick_activity(
    profile: dict[str, Any],
    rng: random.Random,
    elapsed_s: int,
    long_break_used: bool,
) -> Activity:
    profession = profile["profession"]
    hx = profile["hexaco"]
    ext = hx["extraversion"]
    honesty = hx["honesty_humility"]
    agree = hx["agreeableness"]
    openness = hx["openness"]
    commute_minutes = profile["commute_minutes"]
    if 3 * 3600 <= elapsed_s <= 4 * 3600 and not long_break_used:
        return Activity.BREAK_LONG
    weights: dict[Activity, float] = {
        Activity.WORK_FOCUS: 4.0,
        Activity.WORK_MEETING: 2.0,
        Activity.BREAK_SHORT: 1.6,
        Activity.BREAK_LONG: 0.4,
        Activity.COMMUTE: 0.1 if commute_minutes <= 0 else 0.5,
        Activity.LEISURE: 0.5,
        Activity.SLEEP: 0.0,
        Activity.TRANSITION: 1.2,
    }
    if "技术" in profession:
        weights[Activity.WORK_FOCUS] += 2.0
        weights[Activity.WORK_MEETING] -= 0.5
    if "产品" in profession:
        weights[Activity.WORK_MEETING] += 1.5
        weights[Activity.TRANSITION] += 0.6
    if any(k in profession for k in ("创作者", "写作者", "插画")):
        weights[Activity.WORK_FOCUS] += 1.8
        weights[Activity.LEISURE] += 0.8
        weights[Activity.WORK_MEETING] -= 0.6
    if ext > 0.65:
        weights[Activity.WORK_MEETING] += 0.8
        weights[Activity.WORK_FOCUS] -= 0.4
    # honesty_humility: high => less social-posture switching; low => more meeting/transition drift.
    weights[Activity.TRANSITION] *= max(0.7, 1.05 - 0.4 * honesty)
    if honesty < 0.45:
        drift = 0.45 - honesty
        weights[Activity.WORK_MEETING] += 0.4 + 2.0 * drift
        weights[Activity.TRANSITION] += 0.3 + 1.8 * drift
    else:
        calm = honesty - 0.45
        weights[Activity.WORK_MEETING] -= 0.4 * calm
    # agreeableness: high => collaborative/social; low => solo focus.
    weights[Activity.WORK_MEETING] += 1.0 * (agree - 0.5)
    weights[Activity.LEISURE] += 0.8 * (agree - 0.5)
    weights[Activity.WORK_FOCUS] += 1.1 * (0.5 - agree)
    # openness: high => exploration/rest spectrum wider; low => narrower spectrum.
    weights[Activity.LEISURE] += 1.2 * (openness - 0.5)
    weights[Activity.BREAK_LONG] += 0.9 * (openness - 0.5)
    if openness < 0.45:
        narrow = 0.45 - openness
        weights[Activity.LEISURE] -= 0.6 * narrow
        weights[Activity.BREAK_LONG] -= 0.5 * narrow
        weights[Activity.WORK_FOCUS] += 0.9 * narrow
        weights[Activity.WORK_MEETING] += 0.4 * narrow
    for key, val in list(weights.items()):
        weights[key] = max(0.0, val)
    return _weighted_choice(weights, rng)


def _pick_state(profile: dict[str, Any], activity: Activity, rng: random.Random) -> State:
    hx = profile["hexaco"]
    h = hx["honesty_humility"]
    c = hx["conscientiousness"]
    e = hx["emotionality"]
    x = hx["extraversion"]
    anx = profile["attachment_anxiety"]
    weights: dict[State, float] = {
        State.FOCUSED: 1.2 + 2.0 * c,
        State.NEUTRAL: 1.0,
        State.DISTRACTED: 0.6 + (1.0 - c),
        State.FATIGUED: 0.5 + e * 0.8,
        State.EMOTIONAL_POSITIVE: 0.5 + x * 0.9,
        State.EMOTIONAL_NEGATIVE: 0.4 + e * 0.8 + anx * 0.7,
        State.LOW_ENERGY: 0.5 + (1.0 - x) * 0.5,
        State.TRANSITIONING: 0.4,
    }
    if h < 0.5:
        weights[State.EMOTIONAL_NEGATIVE] += (0.5 - h) * 0.8
    else:
        weights[State.EMOTIONAL_NEGATIVE] -= (h - 0.5) * 0.3
    if activity == Activity.WORK_FOCUS:
        weights[State.FOCUSED] += 2.0
        weights[State.DISTRACTED] -= 0.3
    if activity == Activity.WORK_MEETING:
        weights[State.EMOTIONAL_POSITIVE] += 0.8
        weights[State.TRANSITIONING] += 0.4
    if activity == Activity.BREAK_LONG:
        weights[State.LOW_ENERGY] += 0.7
        weights[State.NEUTRAL] += 0.8
    if activity in (Activity.COMMUTE, Activity.TRANSITION):
        weights[State.TRANSITIONING] += 1.8
        weights[State.NEUTRAL] += 0.6
    return _weighted_choice(weights, rng)


def _pick_duration(
    profile: dict[str, Any],
    activity: Activity,
    remaining_s: int,
    rng: random.Random,
) -> int:
    duration_options = {
        Activity.WORK_FOCUS: (420, 600, 720, 900),
        Activity.WORK_MEETING: (300, 600, 900),
        Activity.BREAK_SHORT: (300, 420, 600),
        Activity.BREAK_LONG: (900, 1200, 1500),
        Activity.COMMUTE: (300, 600, 900),
        Activity.LEISURE: (300, 420, 600),
        Activity.TRANSITION: (120, 180, 300),
        Activity.SLEEP: (900, 1200),
    }
    options = duration_options[activity]
    c = profile["hexaco"]["conscientiousness"]
    if activity == Activity.WORK_FOCUS and c >= 0.75:
        options = (420, 600, 720)
    preferred = rng.choice(options)
    if remaining_s <= 900:
        return max(5, remaining_s)
    return min(preferred, remaining_s)


def _hint_from_state(
    profile: dict[str, Any],
    state: State,
    activity: Activity,
    rng: random.Random,
) -> MultimodalHint:
    base = {
        State.FOCUSED: (0.12, 0.10, 0.08, 0.88, 0.08, 0.12, 1),
        State.DISTRACTED: (0.20, 0.18, 0.22, 0.35, 0.05, 0.06, 6),
        State.FATIGUED: (0.62, 0.28, 0.12, 0.30, 0.04, 0.04, 3),
        State.LOW_ENERGY: (0.45, 0.24, 0.10, 0.40, 0.05, 0.05, 2),
        State.EMOTIONAL_POSITIVE: (0.14, 0.12, 0.14, 0.72, 0.70, 0.55, 4),
        State.EMOTIONAL_NEGATIVE: (0.35, 0.26, 0.16, 0.38, 0.50, 0.35, 5),
        State.NEUTRAL: (0.18, 0.08, 0.07, 0.62, 0.15, 0.15, 2),
        State.TRANSITIONING: (0.20, 0.14, 0.20, 0.52, 0.18, 0.18, 5),
    }[state]
    eye, pitch, yaw, gaze, voice, utter, switch = base
    hx = profile["hexaco"]
    honesty = hx["honesty_humility"]
    openness = hx["openness"]
    density = profile["density"]
    if density == "low":
        voice *= 0.8
        utter *= 0.8
        switch = max(0, int(round(switch * 0.8)))
    elif density == "high":
        voice = min(1.0, voice * 1.1)
        utter = min(1.0, utter * 1.1)
        switch = int(round(switch * 1.2))
    elif density == "extreme_bipolar" and state in (State.FATIGUED, State.EMOTIONAL_NEGATIVE):
        eye = min(1.0, eye + 0.12)
        gaze = max(0.0, gaze - 0.08)
        switch = int(round(switch * 1.3))
    # honesty_humility: high consistency => fewer short-window app switches.
    switch_scale = 1.0 + (0.5 - honesty) * 0.6
    switch = int(round(switch * max(0.65, switch_scale)))
    # openness: high => wider utterance-density variance.
    utter_jitter = (rng.random() - 0.5) * (0.06 + 0.22 * openness)
    utter += utter_jitter
    jitter = (rng.random() - 0.5) * 0.04
    eye = _clamp01(eye + jitter)
    gaze = _clamp01(gaze - jitter)
    voice = _clamp01(voice)
    utter = _clamp01(utter)
    pitch = _clamp01(pitch)
    yaw = _clamp01(yaw)
    active_app = _active_app_for_activity(profile["profession"], activity)
    return MultimodalHint(
        eye_closure=eye,
        head_pitch_norm=pitch,
        head_yaw_norm=yaw,
        gaze_stability=gaze,
        voice_present=voice,
        utterance_density=utter,
        active_app=active_app,
        app_switch_count_5min=max(0, switch),
    )


def _active_app_for_activity(profession: str, activity: Activity) -> str:
    if activity == Activity.WORK_MEETING:
        return "Feishu"
    if activity == Activity.WORK_FOCUS:
        if "技术" in profession:
            return "VSCode"
        if "产品" in profession:
            return "Notion"
        if any(k in profession for k in ("创作者", "写作者", "插画")):
            return "WritingTool"
        return "Chrome"
    if activity in (Activity.BREAK_SHORT, Activity.BREAK_LONG, Activity.LEISURE):
        return "Chrome"
    if activity == Activity.TRANSITION:
        return "Slack"
    return ""


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, val))


__all__ = [
    "LLMClient",
    "HeuristicLLMClient",
    "load_persona_brief",
    "generate_day_script",
    "write_script",
    "run_batch",
]
