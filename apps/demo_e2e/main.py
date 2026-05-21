"""Streamlit app for LnMem M0 parser→KG→query demo closure."""
from __future__ import annotations

from datetime import datetime, timezone

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

from contracts.protocols import KGQuery, ParseRequest
from contracts.schema_v1 import SlotID
from graphiti_zh_adapter.adapter import GraphitiZhAdapterSkeleton
from universal_grammar_parser.config import ParserConfig
from universal_grammar_parser.parser_v0_llm import SchemaGuidedLLMParser

from .components import event_to_dict, events_to_dataframe
from .llm_config import make_llm_client


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


def left_panel() -> None:
    """Render left panel for parse-and-upsert workflow."""
    st.header("输入 -> 解析 -> 写入 KG")
    text = st.text_area("中文一句话事件", placeholder="昨晚11点和小李在公司吃了越南粉")
    user_id = st.text_input("user_id", value="chuange")

    if st.button("解析 + 写入 KG", type="primary"):
        if not text.strip():
            st.warning("请输入事件文本")
            return
        with st.spinner("LLM 抽取 20 槽..."):
            req = ParseRequest(text=text, user_id=user_id, timestamp=datetime.now(timezone.utc))
            event = st.session_state.parser.parse(req)

        st.subheader("20 槽抽取结果")
        st.json(event_to_dict(event))

        with st.spinner("写入 KG..."):
            node_id = st.session_state.adapter.upsert(event)
            st.session_state.events_history.append(event)

        counts = event.filled_priority_count()
        st.success(f"已写入 KG, node_id={node_id}")
        st.metric("P0 填充率", f"{counts['P0']}/4")


def right_panel() -> None:
    """Render right panel for KG query and node table browsing."""
    st.header("KG 查询")
    query_text = st.text_input("查询", placeholder="昨天和谁吃饭")
    user_id = st.text_input("query user_id", value="chuange", key="query_user")
    top_k = st.slider("top_k", 1, 10, 5)

    if st.button("查询", key="query_btn"):
        query = KGQuery(
            user_id=user_id,
            text_query=query_text,
            valid_at=datetime.now(timezone.utc),
            top_k=top_k,
        )
        hits = st.session_state.adapter.query(query)
        st.subheader(f"命中 {len(hits)} 条")
        for i, hit in enumerate(hits, start=1):
            action = hit.event.get(SlotID.WHAT_ACTION)
            title = action.value if action else "(no what_action)"
            with st.expander(f"#{i} score={hit.score:.3f} {title}"):
                st.json(event_to_dict(hit.event))
                if hit.explanation:
                    st.caption(hit.explanation)

    st.divider()
    total = len(st.session_state.events_history)
    st.subheader(f"KG 节点总览 (共 {total} 条)")
    if total:
        st.dataframe(events_to_dataframe(st.session_state.events_history), use_container_width=True)


def main() -> None:
    """Render app entrypoint and top-level layout."""
    if st is None:
        raise RuntimeError("streamlit is not installed. Run: pip install streamlit")

    st.set_page_config(page_title="LnMem M0 Demo", layout="wide", page_icon="🧠")
    st.title("LnMem M0 Demo - parser + KG")
    st.caption("中文输入 -> 20 槽拆解 -> 写入 KG -> 查询返回")

    init_state()
    col_l, col_r = st.columns([1, 1])
    with col_l:
        left_panel()
    with col_r:
        right_panel()


if __name__ == "__main__":
    main()
