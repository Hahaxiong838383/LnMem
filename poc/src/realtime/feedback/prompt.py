"""macOS osascript 三层渐进弹窗 — 不打扰为先。

第 1 层（通知中心）：display notification，5 秒过期，不抢焦点
第 2 层（modal dialog）：display dialog，按钮选择 ✓ / ✗ / 输入
第 3 层（详细输入）：仅在 ✗ 或 输入 时弹，choose from list 或 free text

所有 osascript 调用走 subprocess.run + timeout，失败返回 None 不 raise。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

_STATE_OPTIONS = (
    "focused 专注",
    "emotional_positive 积极",
    "emotional_negative 消极",
    "low_energy 低能量",
    "fatigued 疲劳",
    "distracted 分心",
    "transitioning 过渡",
    "idle 中性",
)


@dataclass(frozen=True)
class UserResponse:
    """老板对询问的反馈（不可变）。"""
    label: str  # "match" | "mismatch" | "input" | "skip"
    true_state: str | None
    note: str | None


def _osascript(script: str, timeout_s: float = 60.0) -> str | None:
    """调 osascript 返回 stdout 文本（去尾换行）。timeout/error → None。"""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def show_notification(title: str, message: str) -> None:
    """第 1 层：通知中心提醒（毫秒级，不抢焦点）。"""
    safe_title = title.replace('"', "'")
    safe_msg = message.replace('"', "'")
    script = f'display notification "{safe_msg}" with title "{safe_title}" sound name "Tink"'
    _osascript(script, timeout_s=3.0)


def ask_match(question: str, timeout_s: float = 60.0) -> str | None:
    """第 2 层：modal dialog 三按钮。返回 'match' / 'mismatch' / 'input' / None。"""
    safe = question.replace('"', "'").replace("\\", "\\\\")
    script = (
        f'tell application "System Events" to '
        f'display dialog "{safe}" '
        f'buttons {{"✗ 不准", "📝 输入", "✓ 准确"}} '
        f'default button "✓ 准确" with title "🤖 LnMem 反馈" with icon note '
        f'giving up after {int(timeout_s)}'
    )
    out = _osascript(script, timeout_s=timeout_s + 5.0)
    if not out:
        return None  # 超时/取消
    if "准确" in out and "不准" not in out:
        return "match"
    if "不准" in out:
        return "mismatch"
    if "输入" in out:
        return "input"
    return None


def choose_state(timeout_s: float = 60.0) -> str | None:
    """第 3 层 a：list 选 state。返回 state 名（不含中文标签）或 None。"""
    items = ", ".join(f'"{s}"' for s in _STATE_OPTIONS)
    script = (
        f'tell application "System Events" to '
        f'choose from list {{{items}}} '
        f'with prompt "你刚才的真实状态是？" '
        f'with title "🤖 LnMem · 选择状态" '
        f'default items {{"focused 专注"}}'
    )
    out = _osascript(script, timeout_s=timeout_s + 5.0)
    if not out or out == "false":
        return None
    return out.split()[0]  # 取英文部分


def input_note(prompt_text: str = "请描述你的状态：", timeout_s: float = 120.0) -> str | None:
    """第 3 层 b：自由文本输入。返回老板输入的字符串或 None。"""
    safe = prompt_text.replace('"', "'")
    script = (
        f'tell application "System Events" to '
        f'display dialog "{safe}" '
        f'default answer "" '
        f'with title "🤖 LnMem · 详细描述" '
        f'giving up after {int(timeout_s)}'
    )
    out = _osascript(script, timeout_s=timeout_s + 5.0)
    if not out:
        return None
    # osascript display dialog 输出：button returned:..., text returned:...
    for part in out.split(","):
        part = part.strip()
        if part.startswith("text returned:"):
            return part[len("text returned:"):].strip()
    return None


def run_three_layer_query(
    title: str,
    short_message: str,
    detail_message: str,
    timeout_s: float = 60.0,
    *,
    runner: Any | None = None,
) -> UserResponse:
    """完整三层流程：通知 → 弹窗 → 详细。
    runner 可注入测试 mock（含 show_notification/ask_match/choose_state/input_note 4 个属性）。
    """
    r = runner or _DefaultRunner()
    r.show_notification(title, short_message)
    label = r.ask_match(detail_message, timeout_s=timeout_s)
    if label is None:
        return UserResponse(label="skip", true_state=None, note=None)
    if label == "match":
        return UserResponse(label="match", true_state=None, note=None)
    if label == "mismatch":
        true_state = r.choose_state(timeout_s=timeout_s)
        return UserResponse(label="mismatch", true_state=true_state, note=None)
    if label == "input":
        note = r.input_note(timeout_s=timeout_s * 2)
        return UserResponse(label="input", true_state=None, note=note)
    return UserResponse(label="skip", true_state=None, note=None)


class _DefaultRunner:
    show_notification = staticmethod(show_notification)
    ask_match = staticmethod(ask_match)
    choose_state = staticmethod(choose_state)
    input_note = staticmethod(input_note)
