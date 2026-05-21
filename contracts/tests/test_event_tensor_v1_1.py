"""EventTensor v1.1 升级单测（ADR-018）

验证 intent_type + multimodal_context 两个新字段：
- 向后兼容：旧代码不传也能跑
- 默认值兜底：intent_type 默认 DECLARATIVE
- 字符串自动转 Enum
- to_dict 含新字段
- set() 不可变更新保留新字段
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.event_tensor import EventTensor, IntentType, SlotValue
from contracts.schema_v1 import SlotID


def _make_tensor(**kwargs):
    """工厂：合法 tz-aware timestamp。"""
    return EventTensor(
        user_id="chuange",
        timestamp=datetime.now(timezone.utc),
        **kwargs,
    )


# ──────────────── 向后兼容（旧代码不传新字段） ────────────────

def test_backward_compat_no_new_fields():
    """旧代码不传 intent_type / multimodal_context 仍可构造，默认值兜底。"""
    e = _make_tensor()
    assert e.intent_type == IntentType.DECLARATIVE
    assert e.multimodal_context is None
    assert e.schema_version == "v1.1"


def test_backward_compat_old_test_pattern():
    """旧测试样式（只传 user_id + timestamp）继续有效。"""
    e = EventTensor(user_id="u1", timestamp=datetime.now(timezone.utc))
    assert e.user_id == "u1"
    assert isinstance(e.intent_type, IntentType)


# ──────────────── intent_type 5 类枚举 ────────────────

def test_intent_type_5_classes_exist():
    """5 类 IntentType 必须严格存在。"""
    assert IntentType.DECLARATIVE.value == "陈述"
    assert IntentType.QUESTION.value == "疑问"
    assert IntentType.RHETORICAL.value == "反问"
    assert IntentType.EXCLAMATION.value == "感叹"
    assert IntentType.REQUEST.value == "请求"
    # 不允许有第 6 类
    assert len(list(IntentType)) == 5


def test_intent_type_set_via_enum():
    """传 Enum 直接生效。"""
    e = _make_tensor(intent_type=IntentType.QUESTION)
    assert e.intent_type == IntentType.QUESTION


def test_intent_type_set_via_string_chinese():
    """传中文字符串自动转 Enum（parser 输出便利）。"""
    e = _make_tensor(intent_type="反问")
    assert e.intent_type == IntentType.RHETORICAL


def test_intent_type_invalid_string_raises():
    """非法字符串拒绝（防 LLM 输出漂移）。"""
    with pytest.raises(ValueError, match="intent_type must be one of"):
        _make_tensor(intent_type="随便编一个")


# ──────────────── multimodal_context ────────────────

def test_multimodal_context_default_none():
    """默认不抓帧 → None。"""
    e = _make_tensor()
    assert e.multimodal_context is None


def test_multimodal_context_set_dict():
    """抓帧后挂载完整 V-A + AU + 9-state。"""
    ctx = {
        "v_a": [0.3, -0.1],
        "state_label": "relaxed",
        "state_label_zh": "放松",
        "au_vector": {"AU06": 2.5, "AU12": 3.0},
        "source": "mock",
        "confidence": 0.5,
    }
    e = _make_tensor(multimodal_context=ctx)
    assert e.multimodal_context == ctx
    assert e.multimodal_context["v_a"] == [0.3, -0.1]


# ──────────────── to_dict 含新字段 ────────────────

def test_to_dict_contains_v1_1_fields():
    """to_dict 必须含 intent_type 字符串 + multimodal_context。"""
    e = _make_tensor(
        intent_type=IntentType.QUESTION,
        multimodal_context={"v_a": [0.1, 0.2]},
    )
    d = e.to_dict()
    assert d["intent_type"] == "疑问"  # 序列化为字符串
    assert d["multimodal_context"] == {"v_a": [0.1, 0.2]}
    assert d["schema_version"] == "v1.1"


def test_to_dict_backward_compat_defaults():
    """默认值的 to_dict 字段齐全。"""
    e = _make_tensor()
    d = e.to_dict()
    assert d["intent_type"] == "陈述"
    assert d["multimodal_context"] is None


# ──────────────── set() 不可变更新保留新字段 ────────────────

def test_set_preserves_intent_type():
    """slot 更新不应丢 intent_type。"""
    e = _make_tensor(intent_type=IntentType.EXCLAMATION)
    e2 = e.set(SlotID.WHO_ACTOR, SlotValue(value="川哥"))
    assert e2.intent_type == IntentType.EXCLAMATION
    assert e2.event_id == e.event_id  # 同一事件


def test_set_preserves_multimodal_context():
    """slot 更新不应丢 multimodal_context。"""
    ctx = {"v_a": [0.5, 0.5], "source": "mock"}
    e = _make_tensor(multimodal_context=ctx)
    e2 = e.set(SlotID.WHO_ACTOR, SlotValue(value="川哥"))
    assert e2.multimodal_context == ctx


def test_set_deep_copy_multimodal_context():
    """set() 必须 deep copy multimodal_context（防意外共享）。"""
    ctx = {"v_a": [0.5, 0.5]}
    e = _make_tensor(multimodal_context=ctx)
    e2 = e.set(SlotID.WHO_ACTOR, SlotValue(value="川哥"))
    # 修改原 ctx 不应影响 e2
    ctx["v_a"] = [99, 99]
    assert e2.multimodal_context["v_a"] == [0.5, 0.5]


# ──────────────── 边界 ────────────────

def test_invalid_intent_type_object_raises():
    """传非字符串非 Enum 类型 → 报错。"""
    with pytest.raises((ValueError, TypeError)):
        _make_tensor(intent_type=42)
