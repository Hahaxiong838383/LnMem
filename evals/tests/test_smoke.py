"""Smoke tests for contracts/protocols.py EvalSuite.

只验证骨架导入和 Protocol 形状，不执行真实评估。
"""
from __future__ import annotations

from typing import runtime_checkable

from contracts.event_tensor import EventTensor
from contracts.protocols import EvalCase, EvalResult, EvalSuite, TrafficLight
from contracts.schema_v1 import SlotID
from evals import LnMemEvalSuite, load_cases

RuntimeEvalSuite = runtime_checkable(EvalSuite)


def test_contracts_importable() -> None:
    """contracts.* imports remain available. 契约类型可导入。"""
    assert EvalCase is not None
    assert EvalResult is not None
    assert TrafficLight is not None
    assert EventTensor is not None
    assert SlotID.WHO_ACTOR.value == 1


def test_eval_suite_protocol_compatible() -> None:
    """LnMemEvalSuite has EvalSuite runtime shape. 实现类具备协议形状。"""
    suite = LnMemEvalSuite()
    assert isinstance(suite, RuntimeEvalSuite)


def test_stub_cases_loadable() -> None:
    """Stub JSONL cases load into EvalCase objects. 占位题可加载。"""
    cases = load_cases()
    assert len(cases) == 10
    assert all(isinstance(case, EvalCase) for case in cases)
