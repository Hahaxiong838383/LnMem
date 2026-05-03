"""Smoke tests for contracts/protocols.py::SyntheticDataGenerator.

Ensures imports and Protocol compatibility remain visible to pytest.
"""
from __future__ import annotations

import inspect
from typing import Protocol, runtime_checkable

from contracts.event_tensor import EventTensor
from contracts.protocols import SynthSpec, SyntheticDataGenerator
from contracts.schema_v1 import SCHEMA_V1
from synthetic_data_generator import LnMemSyntheticDataGenerator


@runtime_checkable
class RuntimeSyntheticDataGenerator(SyntheticDataGenerator, Protocol):
    """Runtime-checkable mirror. pytest 运行期结构兼容检查。"""


def test_contract_imports_are_available() -> None:
    """Import core contracts. 确认 contracts.* 能被本模块测试导入。"""
    assert SynthSpec(kind="type_1", n_samples=1).kind == "type_1"
    assert EventTensor
    assert SCHEMA_V1


def test_pipeline_is_runtime_protocol_compatible() -> None:
    """Check Protocol compatibility. 用 runtime_checkable 验证结构兼容。"""
    generator = LnMemSyntheticDataGenerator()
    assert isinstance(generator, RuntimeSyntheticDataGenerator)


def test_generate_signature_matches_contract() -> None:
    """Check public signature. 确认 generate 参数名与 Protocol 契约一致。"""
    params = inspect.signature(LnMemSyntheticDataGenerator.generate).parameters
    assert list(params) == ["self", "spec"]
