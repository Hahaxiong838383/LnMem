"""Phase 1 反馈闭环：主动询问触发 + macOS 弹窗 + 落盘。"""

from poc.src.realtime.feedback.dnd import DNDConfig, get_frontmost_app, should_suppress
from poc.src.realtime.feedback.engine import EngineConfig, ProactiveQueryEngine
from poc.src.realtime.feedback.persistence import append_query, queries_path
from poc.src.realtime.feedback.prompt import (
    UserResponse,
    ask_match,
    choose_state,
    input_note,
    run_three_layer_query,
    show_notification,
)
from poc.src.realtime.feedback.rate_limiter import (
    RateLimiterConfig,
    TokenBucketRateLimiter,
)
from poc.src.realtime.feedback.trigger import ProactiveQueryTrigger, TriggerEvent

__all__ = [
    "DNDConfig",
    "EngineConfig",
    "ProactiveQueryEngine",
    "ProactiveQueryTrigger",
    "RateLimiterConfig",
    "TokenBucketRateLimiter",
    "TriggerEvent",
    "UserResponse",
    "append_query",
    "ask_match",
    "choose_state",
    "get_frontmost_app",
    "input_note",
    "queries_path",
    "run_three_layer_query",
    "should_suppress",
]
