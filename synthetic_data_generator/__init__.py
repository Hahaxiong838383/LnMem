"""Public exports for contracts/protocols.py::SyntheticDataGenerator skeleton.

This package exposes the M0 W1 synthetic data generator API.
"""
from synthetic_data_generator.config import GenerationPlan, LLMFetchImpl, SynthKind, SyntheticGeneratorConfig
from synthetic_data_generator.monitor import BossShareMonitor, BossShareSnapshot
from synthetic_data_generator.pipeline import LnMemSyntheticDataGenerator

__all__ = [
    "BossShareMonitor",
    "BossShareSnapshot",
    "GenerationPlan",
    "LLMFetchImpl",
    "LnMemSyntheticDataGenerator",
    "SynthKind",
    "SyntheticGeneratorConfig",
]
