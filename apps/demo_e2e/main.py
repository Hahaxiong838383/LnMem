"""Streamlit app for LnMem M0 — Phase α 多模态升级版（2026-05-21）

升级清单（cc + gemini 评审修订后 4-in-1）：
1. Phase α webcam snapshot → V-A + 9-state 实时显示（multimodal 模块 LNN-FIRST input signal）
2. KG 卡片展开 20 槽 + 关联 multimodal state（"LNN 符号投影"概念，gemini 评审 #1）
3. CLS 今日精华显示区（读 data/cls.sqlite.daily_summaries）
4. 记忆 UI v0.1 删除按钮（极简版 → adapter.invalidate）

设计原则（ADR-016 + gemini 评审吸收）：
- 视觉是 base v3 多模态输入信号，不是 LLM 分类
- Mock V-A 默认（毫秒级），py-feat 真推理留开关
- multimodal state 旁挂 session_state，不污染 EventTensor v1 封版 schema
- 删除按钮 streamlit rerun 风险 → 用 session_state 缓存 KG 节点列表
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Streamlit run 时 main.py 不是 package member，需手动加 repo root 到 sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

from contracts.protocols import KGQuery, ParseRequest
from contracts.schema_v1 import SCHEMA_V1, SlotID
from graphiti_zh_adapter.adapter import GraphitiZhAdapterSkeleton
from universal_grammar_parser.config import ParserConfig
from universal_grammar_parser.parser_v0_llm import SchemaGuidedLLMParser

from apps.demo_e2e.components import event_to_dict, events_to_dataframe
from apps.demo_e2e.llm_config import make_llm_client
from apps.demo_e2e.multimodal import (
    AU_NAMES_20,
    STATE_LABELS_ZH,
    SnapshotResult,
    snapshot_to_state,
)

CLS_DB_DEFAULT = Path("data/cls.sqlite")


# ──────────────────────────── 初始化 ────────────────────────────

def init_state() -> None:
    """Initialize parser, adapter, and history in Streamlit session state."""
    if "parser" not in st.session_state:
        llm_client = make_llm_client()
        st.session_state.parser = SchemaGuidedLLMParser(
            llm_client=llm_client,
            config=ParserConfig(llm_model="gemini-3-flash-preview", timeout_seconds=15.0),
        )
    if "adapter" not in st.session_state:
        st.session_state.adapter = GraphitiZhAdapterSkeleton()
    if "events_history" not in st.session_state:
        st.session_state.events_history = []
    # Phase α 旁挂：event_id → SnapshotResult（不污染 EventTensor schema）
    if "event_multimodal" not in st.session_state:
        st.session_state.event_multimodal = {}
    # 当前抓帧的 snapshot（解析前预览用）
    if "current_snapshot" not in st.session_state:
        st.session_state.current_snapshot = None
    # 已 invalidate 的 event_id（避免重复显示）
    if "invalidated_ids" not in st.session_state:
        st.session_state.invalidated_ids = set()


# ──────────────────────────── Snapshot UI 渲染 ────────────────────────────

def render_snapshot_card(snap: SnapshotResult, key_prefix: str = "") -> None:
    """渲染 multimodal SnapshotResult — 让川哥看到 base v3 input signal。"""
    v, a = snap["v_a"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("LNN 涌现状态", snap["state_label_zh"], help=f"英文: {snap['state_label']} (Russell sector)")
    with col_b:
        st.metric("Valence (V)", f"{v:+.2f}", help="情感极性 [-1, 1]")
    with col_c:
        st.metric("Arousal (A)", f"{a:+.2f}", help="唤起度 [-1, 1]")
    st.caption(
        f"📡 source: `{snap['source']}` · confidence: {snap['confidence']:.2f}"
        f" · face_detected: {snap['face_detected']}"
    )
    # AU 折叠展示
    with st.expander("🧠 20 AU 向量（base v3 多模态输入信号）", expanded=False):
        nonzero = {k: v for k, v in snap["au_vector"].items() if v > 0.01}
        if nonzero:
            st.json(nonzero)
        else:
            st.caption("当前 AU 全 0（neutral）")


# ──────────────────────────── 左面板：输入 + 解析 + 抓帧 ────────────────────────────

def left_panel() -> None:
    """输入文本 + Phase α webcam snapshot + 解析写 KG."""
    st.header("📝 输入 → 多模态融合 → 写入 KG")

    text = st.text_area(
        "中文一句话事件",
        placeholder="昨晚11点和小李在公司吃了越南粉",
        key="input_text",
    )
    user_id = st.text_input("user_id", value="chuange", key="input_user")

    # Phase α: webcam snapshot
    st.subheader("📸 表情抓拍（Phase α 多模态）")
    snap_mode = st.radio(
        "推理模式",
        ["mock (毫秒级，演示用)", "pyfeat (真模型 ~5s)"],
        index=0,
        horizontal=True,
        help="ADR-016 base v3 训练前的 quick win；真模型需 py-feat 装齐",
    )
    mode_key = "pyfeat" if snap_mode.startswith("pyfeat") else "mock"

    img_file = st.camera_input("点击抓帧（可选）", key="webcam_input")
    if img_file is not None:
        img_bytes = img_file.getvalue()
        with st.spinner(f"提取 V-A + 9-state ({mode_key} 模式)..."):
            snap = snapshot_to_state(img_bytes, mode=mode_key)  # type: ignore[arg-type]
        st.session_state.current_snapshot = snap
        st.success("📡 多模态 signal 已捕获")
        render_snapshot_card(snap, key_prefix="preview")
    elif st.session_state.current_snapshot is not None:
        st.info("📡 上次抓帧 signal 已暂存（解析时会绑定到事件）")
        render_snapshot_card(st.session_state.current_snapshot, key_prefix="cached")

    st.divider()

    if st.button("🚀 解析 + 写入 KG", type="primary"):
        if not text.strip():
            st.warning("请输入事件文本")
            return
        with st.spinner("LLM 抽取 20 槽..."):
            req = ParseRequest(text=text, user_id=user_id, timestamp=datetime.now(timezone.utc))
            event = st.session_state.parser.parse(req)

        # Phase α：绑定 multimodal context 到 event_id（旁挂 session_state）
        if st.session_state.current_snapshot is not None:
            st.session_state.event_multimodal[event.event_id] = st.session_state.current_snapshot

        st.subheader("📋 20 槽抽取结果")
        st.json(event_to_dict(event))

        with st.spinner("写入 KG..."):
            node_id = st.session_state.adapter.upsert(event)
            st.session_state.events_history.append(event)

        counts = event.filled_priority_count()
        st.success(f"✅ 已写入 KG · node_id={node_id[:12]}...")

        col1, col2 = st.columns(2)
        col1.metric("P0 填充率", f"{counts['P0']}/4")
        col2.metric("多模态绑定", "已绑定" if event.event_id in st.session_state.event_multimodal else "未抓帧")


# ──────────────────────────── 右面板：查询 + 删除 + CLS 精华 ────────────────────────────

def render_kg_hit(hit, idx: int) -> None:
    """渲染单个 KG hit：20 槽展开 + 多模态状态 + 极简删除按钮（gemini #2 + 川哥钦定）。"""
    event = hit.event
    if event.event_id in st.session_state.invalidated_ids:
        return  # 已失效不再渲染

    action = event.get(SlotID.WHAT_ACTION)
    title = action.value if action else "(no what_action)"

    with st.expander(f"#{idx} score={hit.score:.3f} · {title}", expanded=False):
        # 多模态 state（若有）
        snap = st.session_state.event_multimodal.get(event.event_id)
        if snap is not None:
            st.caption("🧠 关联的 LNN 涌现状态")
            render_snapshot_card(snap, key_prefix=f"hit_{idx}")
            st.divider()

        # 20 槽完整展开（gemini #1 命名建议：LNN 符号投影）
        st.caption("📋 20 槽符号投影（万能语法）")
        st.json(event_to_dict(event))

        if hit.explanation:
            st.caption(f"💡 {hit.explanation}")

        # 删除按钮（gemini #2 警告：rerun 风险，用 session_state 兜底）
        col_del, col_info = st.columns([1, 3])
        with col_del:
            if st.button("🗑 失效此事件", key=f"del_{event.event_id}", help="bi-temporal invalidate（不物理删除）"):
                t_invalid = datetime.now(timezone.utc)
                st.session_state.adapter.invalidate(event.event_id, t_invalid)
                st.session_state.invalidated_ids.add(event.event_id)
                st.success(f"已标记失效 · t_invalid={t_invalid.strftime('%H:%M:%S')}")
                st.rerun()
        with col_info:
            st.caption(f"event_id: `{event.event_id[:16]}...`")


def render_cls_panel(db_path: Path, user_id: str) -> None:
    """CLS 今日精华显示（gemini #3 极简：只读 + 静态）。"""
    st.subheader("🌙 CLS 夜间整合 — 今日精华")

    if not db_path.exists():
        st.info(f"CLS DB 尚未生成（{db_path}）。运行 `scripts/cls_consolidate.py` 后回看。")
        return

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.execute(
            "SELECT date, n_events, highlights_json, llm_model, created_at "
            "FROM daily_summaries WHERE user_id = ? ORDER BY date DESC LIMIT 3",
            (user_id,),
        )
        rows = cur.fetchall()
        conn.close()
    except sqlite3.OperationalError as exc:
        st.warning(f"CLS DB 读取失败: {exc}")
        return

    if not rows:
        st.caption("尚无 CLS 整合记录。今日事件累积满后 02:00 cron 自动跑。")
        return

    for date_str, n_events, highlights_json, llm_model, created_at in rows:
        with st.expander(f"📅 {date_str} (events={n_events})", expanded=(rows.index((date_str, n_events, highlights_json, llm_model, created_at)) == 0)):
            try:
                parsed = json.loads(highlights_json)
                for h in parsed.get("highlights", []):
                    h_type = h.get("type", "?")
                    icon = {"事实": "📌", "情绪": "💭", "未完成": "⏳"}.get(h_type, "🔹")
                    st.markdown(f"{icon} **【{h_type}】** {h.get('summary', '?')}")
                    evidence = h.get("evidence_event_ids", [])
                    if evidence:
                        st.caption(f"证据 event_ids: {', '.join(evidence)}")
            except Exception as exc:
                st.warning(f"highlights 解析失败: {exc}")
            st.caption(f"🤖 {llm_model} · generated {created_at}")


def right_panel() -> None:
    """KG 查询 + 删除 + 节点总览 + CLS 精华."""
    st.header("🔍 KG 查询 + 记忆 UI")

    query_text = st.text_input("查询", placeholder="昨天和谁吃饭", key="query_text")
    user_id = st.text_input("query user_id", value="chuange", key="query_user")
    top_k = st.slider("top_k", 1, 10, 5, key="query_topk")

    if st.button("🔎 查询", key="query_btn"):
        query = KGQuery(
            user_id=user_id,
            text_query=query_text,
            valid_at=datetime.now(timezone.utc),
            top_k=top_k,
        )
        hits = st.session_state.adapter.query(query)
        st.session_state.last_hits = hits

    hits = st.session_state.get("last_hits", [])
    if hits:
        active_hits = [h for h in hits if h.event.event_id not in st.session_state.invalidated_ids]
        st.subheader(f"命中 {len(active_hits)} 条（已过滤失效）")
        for i, hit in enumerate(active_hits, start=1):
            render_kg_hit(hit, i)

    st.divider()
    total = len(st.session_state.events_history)
    active = total - len(st.session_state.invalidated_ids)
    st.subheader(f"📊 KG 节点总览（{active}/{total} 活跃）")
    if total:
        st.dataframe(events_to_dataframe(st.session_state.events_history), use_container_width=True)

    st.divider()
    render_cls_panel(CLS_DB_DEFAULT, user_id)


# ──────────────────────────── 主入口 ────────────────────────────

def main() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed. Run: pip install streamlit")

    st.set_page_config(page_title="LnMem M0 Demo · Phase α", layout="wide", page_icon="🧠")
    st.title("🧠 LnMem M0 Demo · Phase α 多模态升级")
    st.caption(
        "中文输入 → 万能语法 20 槽 + 表情多模态 → 写入 bi-temporal KG → 查询/失效 · "
        "ADR-016 LNN-FIRST base v3 input signal preview"
    )

    init_state()
    col_l, col_r = st.columns([1, 1])
    with col_l:
        left_panel()
    with col_r:
        right_panel()


if __name__ == "__main__":
    main()
