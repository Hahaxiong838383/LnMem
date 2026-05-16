"""TDD: poc/src/lnn/feedback_dataset.py

覆盖
----
1. load_events 读真实格式 jsonl（match / mismatch / skip / input 全 4 类）
2. filter_for_training 仅保留 match + mismatch
3. count_mismatch / count_by_label 计数正确
4. 损坏行 fail-soft（warning 不抛）
5. FeedbackDataError 当目录不存在
6. date_prefix 过滤
7. FeedbackEvent.reward() 映射正确
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from poc.src.lnn.feedback_dataset import (
    count_by_label,
    count_mismatch,
    filter_for_training,
    load_events,
)
from poc.src.lnn.online_contracts import (
    REWARD_MAP,
    FeedbackDataError,
    FeedbackEvent,
)


# ---------- Fixtures ----------


@pytest.fixture
def feedback_dir(tmp_path: Path) -> Path:
    """模拟 daemon V0.2 写入的真实格式 jsonl。"""
    d = tmp_path / "feedback"
    d.mkdir()

    # 5/9 — 老格式（无 session_id / event_uuid）
    rows_0509 = [
        {
            "ts": "2026-05-09T12:56:09.782468Z",
            "frame_id": 0,
            "reason": "low_vlm_conf",
            "signal": {"brow_furrow": 0.0, "smile_intensity": 1.0,
                       "eye_closure": 0.0, "cheek_raise": 1.0},
            "vlm_label": {"confidence": 0.05, "valence": 0.0, "arousal": 0.0,
                          "model_used": "gemini-3.1-flash-lite-preview"},
            "user_response": {"label": "mismatch", "true_state": "distracted",
                              "note": None},
            "latency_s": 27.08,
        },
        {
            "ts": "2026-05-09T13:16:19.801913Z",
            "frame_id": 1210,
            "reason": "low_vlm_conf",
            "signal": {"brow_furrow": 0.0, "smile_intensity": 1.0,
                       "eye_closure": 0.0, "cheek_raise": 1.0},
            "vlm_label": None,
            "user_response": {"label": "input", "true_state": None,
                              "note": "我不在，检测错了吧"},
            "latency_s": 62.76,
        },
    ]
    (d / "queries_2026-05-09.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_0509) + "\n",
        encoding="utf-8",
    )

    # 5/10 — V0.2 新格式 (session_id + event_uuid)
    rows_0510 = [
        {
            "ts": "2026-05-10T04:05:53.388726Z",
            "frame_id": 158,
            "reason": "brow_burst",
            "signal": {"brow_furrow": 1.0, "smile_intensity": 0.0832,
                       "eye_closure": 0.0, "cheek_raise": 0.1171},
            "vlm_label": None,
            "user_response": {"label": "match", "true_state": None, "note": None},
            "latency_s": 11.42,
            "session_id": "41f00738-657e-4c90-8f36-178b9a01dafb",
            "event_uuid": "ddf525366f1a475abddf2835b5c8fd6c",
        },
        {
            "ts": "2026-05-10T04:32:17.623301Z",
            "frame_id": 1742,
            "reason": "low_vlm_conf",
            "signal": {"brow_furrow": 0.0, "smile_intensity": 1.0,
                       "eye_closure": 0.0, "cheek_raise": 1.0},
            "vlm_label": {"confidence": 0.0, "model_used": "gemini-3.1-pro-preview"},
            "user_response": {"label": "skip", "true_state": None, "note": None},
            "latency_s": 60.26,
            "session_id": "41f00738-657e-4c90-8f36-178b9a01dafb",
            "event_uuid": "72e618e188224bcdab0902b168ca4214",
        },
    ]
    (d / "queries_2026-05-10.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_0510) + "\n",
        encoding="utf-8",
    )
    return d


# ---------- 行为测试 ----------


class TestFeedbackEvent:
    """REWARD_MAP + FeedbackEvent.reward() 契约。"""

    def test_reward_map_canonical_values(self) -> None:
        assert REWARD_MAP["match"] == 1.0
        assert REWARD_MAP["mismatch"] == -1.0
        assert REWARD_MAP["skip"] == 0.0
        assert REWARD_MAP["input"] == 0.0

    def test_reward_method(self) -> None:
        ev = FeedbackEvent(
            ts="2026-05-10T00:00:00Z", frame_id=0, reason="brow_burst",
            signal={}, vlm_label=None, label="mismatch",
        )
        assert ev.reward() == -1.0


class TestLoadEvents:
    """load_events 行为契约。"""

    def test_load_all_events(self, feedback_dir: Path) -> None:
        events = load_events(feedback_dir)
        assert len(events) == 4

    def test_load_preserves_order_by_filename_then_line(
        self, feedback_dir: Path
    ) -> None:
        events = load_events(feedback_dir)
        # 文件名字典序：5/9 在前，5/10 在后
        assert events[0].ts.startswith("2026-05-09")
        assert events[-1].ts.startswith("2026-05-10")

    def test_filter_by_date_prefix(self, feedback_dir: Path) -> None:
        events = load_events(feedback_dir, date_prefix="2026-05-10")
        assert len(events) == 2
        assert all(ev.ts.startswith("2026-05-10") for ev in events)

    def test_returns_frozen_tuple(self, feedback_dir: Path) -> None:
        events = load_events(feedback_dir)
        assert isinstance(events, tuple)

    def test_raises_when_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FeedbackDataError):
            load_events(tmp_path / "nonexistent")

    def test_event_fields_parsed_correctly(self, feedback_dir: Path) -> None:
        events = load_events(feedback_dir, date_prefix="2026-05-09")
        first = events[0]
        assert first.label == "mismatch"
        assert first.true_state == "distracted"
        assert first.signal["smile_intensity"] == 1.0
        assert first.vlm_label is not None
        assert first.vlm_label["confidence"] == 0.05

    def test_v02_session_uuid_preserved(self, feedback_dir: Path) -> None:
        events = load_events(feedback_dir, date_prefix="2026-05-10")
        match_ev = next(ev for ev in events if ev.label == "match")
        assert match_ev.session_id == "41f00738-657e-4c90-8f36-178b9a01dafb"
        assert match_ev.event_uuid is not None


class TestMalformedRows:
    """Fail-soft：损坏行不抛，但发 warning。"""

    def test_skip_invalid_json_line(self, tmp_path: Path) -> None:
        d = tmp_path / "feedback"
        d.mkdir()
        (d / "queries_2026-05-10.jsonl").write_text(
            '{"ts":"2026-05-10T00:00:00Z","frame_id":0,"reason":"x",'
            '"signal":{},"vlm_label":null,'
            '"user_response":{"label":"match","true_state":null,"note":null},'
            '"latency_s":0}\n'
            "{not-json}\n"  # 损坏行
            '{"ts":"2026-05-10T00:00:01Z","frame_id":1,"reason":"x",'
            '"signal":{},"vlm_label":null,'
            '"user_response":{"label":"mismatch","true_state":null,"note":null},'
            '"latency_s":0}\n',
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            events = load_events(d)
        assert len(events) == 2
        assert any("bad jsonl" in str(item.message).lower() for item in w)

    def test_skip_non_dict_jsonl_row(self, tmp_path: Path) -> None:
        """F3-01 (codex): jsonl 行是 list/number/string 时必须 fail-soft 跳过，不抛 AttributeError。"""
        d = tmp_path / "feedback"
        d.mkdir()
        (d / "queries_2026-05-10.jsonl").write_text(
            '[1, 2, 3]\n'  # list (legal json, but raw not dict)
            '"a string"\n'  # string
            '42\n'  # number
            '{"ts":"2026-05-10T00:00:00Z","frame_id":0,"reason":"x",'
            '"signal":{},"vlm_label":null,'
            '"user_response":{"label":"match","true_state":null,"note":null},'
            '"latency_s":0}\n',
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            events = load_events(d)
        # 只有最后一条 dict 行应被解析
        assert len(events) == 1
        # 至少 1 个 warning
        assert any("non-dict" in str(item.message).lower() for item in w)

    def test_skip_unknown_label(self, tmp_path: Path) -> None:
        d = tmp_path / "feedback"
        d.mkdir()
        (d / "queries_2026-05-10.jsonl").write_text(
            '{"ts":"2026-05-10T00:00:00Z","frame_id":0,"reason":"x",'
            '"signal":{},"vlm_label":null,'
            '"user_response":{"label":"unknown_label","true_state":null,"note":null},'
            '"latency_s":0}\n',
            encoding="utf-8",
        )
        events = load_events(d)
        assert events == ()


class TestFilteringAndCounts:
    """filter_for_training / count_mismatch / count_by_label。"""

    def test_filter_keeps_only_match_mismatch(self, feedback_dir: Path) -> None:
        all_events = load_events(feedback_dir)
        train = filter_for_training(all_events)
        assert all(ev.label in {"match", "mismatch"} for ev in train)
        labels = {ev.label for ev in train}
        assert "skip" not in labels
        assert "input" not in labels

    def test_count_mismatch(self, feedback_dir: Path) -> None:
        events = load_events(feedback_dir)
        # 当前 fixture 中只有 1 条 mismatch
        assert count_mismatch(events) == 1

    def test_count_by_label_full_distribution(self, feedback_dir: Path) -> None:
        events = load_events(feedback_dir)
        dist = count_by_label(events)
        assert dist == {"match": 1, "mismatch": 1, "skip": 1, "input": 1}


class TestMinEventTsWatermark:
    """F-03 + F-06 fix (codex 二审/三审, 2026-05-16): min_event_ts 增量水位线"""

    def _write_events(self, d: Path, events: list[tuple[str, str]]) -> None:
        """events: list of (ts, label)"""
        import json as _json
        with (d / "queries_2026-05-16.jsonl").open("w", encoding="utf-8") as f:
            for ts, label in events:
                row = {
                    "ts": ts,
                    "frame_id": 0,
                    "reason": "pose_held",
                    "signal": {},
                    "vlm_label": None,
                    "user_response": {"label": label, "true_state": None, "note": None},
                    "latency_s": 0.0,
                }
                f.write(_json.dumps(row) + "\n")

    def test_min_event_ts_filters_strictly_greater(self, tmp_path: Path) -> None:
        """ev.ts <= min_event_ts → 过滤；ev.ts > min_event_ts → 保留"""
        self._write_events(tmp_path, [
            ("2026-05-16T08:00:00Z", "match"),    # 早，过滤
            ("2026-05-16T08:14:09Z", "match"),    # 等于 watermark，过滤（<= 严格）
            ("2026-05-16T08:14:10Z", "mismatch"), # 晚 1s，保留
            ("2026-05-16T09:00:00Z", "match"),    # 晚，保留
        ])
        events = load_events(tmp_path, min_event_ts="2026-05-16T08:14:09Z")
        assert len(events) == 2
        assert events[0].label == "mismatch"
        assert events[1].label == "match"

    def test_min_event_ts_handles_microseconds(self, tmp_path: Path) -> None:
        """events.ts 可能含微秒，水位线无 → 仍要数值比较正确"""
        self._write_events(tmp_path, [
            ("2026-05-16T08:14:09.123456Z", "match"),
            ("2026-05-16T08:14:09.999999Z", "match"),
            ("2026-05-16T08:14:10.000000Z", "mismatch"),
        ])
        events = load_events(tmp_path, min_event_ts="2026-05-16T08:14:09.500000Z")
        assert len(events) == 2

    def test_min_event_ts_mixed_format_z_vs_offset(self, tmp_path: Path) -> None:
        """F-06: events.ts 用 '+00:00' / 水位线用 'Z' → 数值比较应等价（无字典序错位）"""
        self._write_events(tmp_path, [
            # 用 +00:00 后缀（仓内 daemon_main.py:326 .isoformat() 输出格式）
            ("2026-05-16T08:00:00+00:00", "match"),
            ("2026-05-16T09:00:00+00:00", "mismatch"),
        ])
        events = load_events(tmp_path, min_event_ts="2026-05-16T08:30:00Z")
        assert len(events) == 1
        assert events[0].label == "mismatch"

    def test_min_event_ts_compact_format_should_NOT_work_as_iso(self, tmp_path: Path) -> None:
        """codex 三审：紧凑格式 '20260516T...' 不是合法 ISO8601 → 解析失败 → warning + 不过滤"""
        import warnings as _w
        self._write_events(tmp_path, [
            ("2026-05-16T08:00:00Z", "match"),
            ("2026-05-16T09:00:00Z", "mismatch"),
        ])
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            events = load_events(tmp_path, min_event_ts="20260516T084500Z")
        # 不可解析 → warning + 不过滤（返全部）
        assert len(events) == 2
        assert any("unparseable" in str(w.message).lower() for w in caught), "expected unparseable warning"

    def test_min_event_ts_none_returns_all(self, tmp_path: Path) -> None:
        """min_event_ts=None → 行为不变（向后兼容）"""
        self._write_events(tmp_path, [
            ("2026-05-16T08:00:00Z", "match"),
            ("2026-05-16T09:00:00Z", "mismatch"),
        ])
        events = load_events(tmp_path, min_event_ts=None)
        assert len(events) == 2

    def test_min_event_ts_filter_returns_empty_when_all_old(self, tmp_path: Path) -> None:
        """所有 events ts < watermark → 返空（cron 应判 below_threshold）"""
        self._write_events(tmp_path, [
            ("2026-05-15T08:00:00Z", "mismatch"),
            ("2026-05-16T08:00:00Z", "mismatch"),
        ])
        events = load_events(tmp_path, min_event_ts="2026-05-16T08:14:09Z")
        assert len(events) == 0
