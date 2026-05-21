"""UI helper formatters for EventTensor rendering."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from contracts.event_tensor import EventTensor, SlotValue
from contracts.schema_v1 import SCHEMA_V1, SlotID


def event_to_dict(event: EventTensor) -> Dict[str, Any]:
    """Convert EventTensor into grouped dict by slot priority for UI."""
    grouped: Dict[str, Dict[str, Any]] = {"P0": {}, "P1": {}, "P2": {}, "P3": {}}
    for slot_id, slot_val in event.slots.items():
        meta = SCHEMA_V1[slot_id]
        grouped[meta.priority.name][meta.name] = _to_plain_value(slot_val)
    return {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "timestamp": event.timestamp.isoformat(),
        "valid_until": event.valid_until.isoformat() if event.valid_until else None,
        **grouped,
    }


def events_to_dataframe(events: Iterable[EventTensor]):
    """Convert EventTensor list into a concise pandas dataframe."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for event in events:
        priority = event.filled_priority_count()
        rows.append(
            {
                "event_id": event.event_id[:8],
                "timestamp": event.timestamp.strftime("%H:%M:%S"),
                "WHO": _get_slot(event, SlotID.WHO_ACTOR),
                "WHAT": _get_slot(event, SlotID.WHAT_ACTION),
                "WHEN": _get_slot(event, SlotID.WHEN_TIME),
                "WHERE": _get_slot(event, SlotID.WHERE_LOCATION),
                "P0": f"{priority['P0']}/4",
            }
        )
    return pd.DataFrame(rows)


def _get_slot(event: EventTensor, slot_id: SlotID) -> Any:
    slot = event.get(slot_id)
    return _to_plain_value(slot)


def _to_plain_value(slot_value: SlotValue | None) -> Any:
    if slot_value is None:
        return None
    value = slot_value.value
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    return str(value)
