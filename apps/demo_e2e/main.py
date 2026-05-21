"""Streamlit app for LnMem M0 — Phase α 多模态升级版 v2（2026-05-21 修复 + UI 友好化）

修复 3 个 bug（基于 15:30 实测割裂感分析）：
  Bug 1: parser LLM 超时 → llm_config.py 加 reasoning_effort=minimal + max_tokens=4096
  Bug 2: CLS DB 路径相对 cwd 失锚 → 改用基于 __file__ 的绝对路径
  Bug 3: use_container_width deprecated → 改用 width='stretch'

UI 友好化：
  - 顶部「使用说明」展开框（4 步流程）
  - 所有专业术语带 help tooltip（LNN / Valence / Arousal / AU / KG / CLS 等）
  - 英文术语全部中文注解
  - 错误提示友好化（含修复指引）
  - 步骤数字标号（1️⃣ 输入 → 2️⃣ 抓帧 → 3️⃣ 解析 → 4️⃣ 查询/失效）
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

# Bug 2 fix: CLS DB 路径用基于 __file__ 的绝对路径，避免 streamlit 启动 cwd 不在 repo 时失锚
# __file__ = apps/demo_e2e/main.py → parents[2] = LnMem-poc 仓库根
CLS_DB_DEFAULT = _REPO_ROOT / "data" / "cls.sqlite"


# ════════════════════════════════════════════════════════════════════
# 术语注解（help tooltip 复用，专业词全部加中文注释）
# ════════════════════════════════════════════════════════════════════

HELP = {
    "lnn": (
        "**LNN（Liquid Neural Network · 液态神经网络）** —— "
        "LnMem 的心智核心。基于 LTC/CfC 微分方程，把「时间/演化/记忆/个性化」作为第一性原理。"
        "与传统 LLM（大语言模型）不同，LNN 学的是连续状态演化，不是文本概率。"
    ),
    "valence": (
        "**Valence（情感极性）** —— Russell 圆形情感模型横轴。"
        "范围 [-1, +1]：负值表示不愉快（皱眉/沮丧），正值表示愉快（开心/放松）。"
    ),
    "arousal": (
        "**Arousal（唤起度）** —— Russell 圆形情感模型纵轴。"
        "范围 [-1, +1]：负值表示平静低能量（放松/疲惫），正值表示高能量（兴奋/紧张）。"
    ),
    "9state": (
        "**9-state（Russell 9 状态）** —— "
        "用 V-A 二维平面分 1 个圆心 neutral（平静）+ 8 个 45° 扇区"
        "（happy 开心 / excited 兴奋 / alert 警觉 / tense 紧张 / sad 悲伤 / depressed 沮丧 / calm 安宁 / relaxed 放松）。"
    ),
    "au": (
        "**AU（Action Unit · 面部动作单元）** —— "
        "FACS 面部编码系统的原子动作（如 AU06=脸颊上提，AU12=嘴角上扬=笑）。"
        "py-feat 工具可从一张人脸图提取 20 个 AU 强度 [0, 5]。"
        "在 LnMem 中，AU 不是用来分类表情，是作为 base v3 LNN 的多模态输入信号。"
    ),
    "kg": (
        "**KG（Knowledge Graph · 知识图谱）** —— "
        "LnMem 用 bi-temporal（双时态）KG 存事件：每个节点带「事件发生时间 t_valid」和「失效时间 t_invalid」，"
        "永不物理删除，只标记失效，保留时间维度的真相。"
    ),
    "slot20": (
        "**万能语法 20 槽** —— LnMem 把任何一句话拆成 20 个语义槽位（who/what/when/where/why/...），"
        "按 P0-P3 优先级分层。P0 是核心 4 槽（人事时地），P1-P3 是修饰/上下文。"
        "20 槽是 LNN hidden state 向符号层投影的「观测头」，不是 LLM 实体抽取。"
    ),
    "cls": (
        "**CLS（Complementary Learning Systems · 互补学习系统）** —— "
        "海马体/新皮层双系统理论。LnMem 用每日 02:00 cron 把当天 events 浓缩为 3 条「今日精华」（事实/情绪/未完成），"
        "模拟睡眠时的记忆整合。"
    ),
    "intent": (
        "**Intent（意图类型）** —— "
        "5 类：陈述 / 疑问 / 反问 / 感叹 / 请求。"
        "Phase β（5/22+）由 LLM parser 抽取，让 AI 区分「我饿了」（陈述）vs「我饿了？」（自问/反问）。"
    ),
    "invalidate": (
        "**Invalidate（逻辑失效）** —— "
        "bi-temporal KG 不删数据，只在节点上记 t_invalid 时间戳。"
        "查询时按 valid_at 过滤即可隐藏失效节点，保留完整时间真相，支持「5/20 我以为是这样，5/21 才知道是那样」的历史溯源。"
    ),
    "confidence": (
        "**Confidence（置信度）** —— "
        "[0, 1] 区间。当前 mock 模式诚实标 0.5（不假装真识别），"
        "pyfeat 模式标 0.65（py-feat AU 真提取但 V-A 投影仍是占位算法），"
        "base v3 训练完后会替换为真模型输出。"
    ),
}


# ════════════════════════════════════════════════════════════════════
# Streamlit Session 状态初始化
# ════════════════════════════════════════════════════════════════════

def init_state() -> None:
    """初始化 Streamlit session_state（parser / KG adapter / 历史 / 多模态旁挂）。"""
    if "parser" not in st.session_state:
        try:
            llm_client = make_llm_client()
            st.session_state.parser = SchemaGuidedLLMParser(
                llm_client=llm_client,
                config=ParserConfig(llm_model="gemini-3-flash-preview", timeout_seconds=20.0),
            )
            st.session_state.parser_error = None
        except Exception as exc:
            st.session_state.parser = None
            st.session_state.parser_error = str(exc)
    if "adapter" not in st.session_state:
        # GraphitiZhAdapterSkeleton: bi-temporal KG 适配器，目前 InMemory backend（重启会丢）
        st.session_state.adapter = GraphitiZhAdapterSkeleton()
    if "events_history" not in st.session_state:
        st.session_state.events_history = []
    if "event_multimodal" not in st.session_state:
        # event_id → SnapshotResult；不污染 EventTensor v1 封版 schema（Phase β 走 ADR-018 升 schema）
        st.session_state.event_multimodal = {}
    if "current_snapshot" not in st.session_state:
        st.session_state.current_snapshot = None
    if "invalidated_ids" not in st.session_state:
        # 已 invalidate 的 event_id；用 set 兜底防 streamlit rerun 时丢状态
        st.session_state.invalidated_ids = set()


# ════════════════════════════════════════════════════════════════════
# 顶部使用说明（4 步流程引导）
# ════════════════════════════════════════════════════════════════════

def render_usage_guide() -> None:
    """顶部使用说明 — 让川哥一眼明白这个 demo 在干嘛。"""
    with st.expander("📖 这个 demo 在干什么？（点开看 4 步流程 + 术语速查）", expanded=False):
        st.markdown(
            """
### 🎯 LnMem 是什么
**LnMem = Liquid neural Memory** —— 用 LNN（液态神经网络）作为永远在线心智核心的伴侣型 AI 记忆系统。
不是传统 LLM RAG 把记忆当文本存。LNN 学的是用户连续状态的时间演化。

### 📍 这个 demo 在演示
M0 阶段的端到端最小闭环：**自然语言一句话 → 多模态融合 → 写入 bi-temporal KG → 查询 / 删除 / 夜间整合**。

### 🚶 操作 4 步走

| 步骤 | 操作 | 你会看到 |
|------|------|---------|
| **1️⃣ 输入文本** | 左侧输入一句中文事件（如"昨晚和小李在公司吃越南粉"） | 文本框记住你的输入 |
| **2️⃣ （可选）抓拍表情** | 左侧 📸 点摄像头抓帧 | 实时显示 V-A 情感坐标 + 9-state + 20 AU |
| **3️⃣ 解析+写 KG** | 点 🚀 按钮 | LLM 拆 20 槽 + 多模态绑定 + 写入 KG（2-5s）|
| **4️⃣ 查询 / 失效 / 看精华** | 右侧查询关键词 → 看卡片 → 可点🗑 失效 → 底部看 CLS 精华 | KG 节点列表 + 删除回填 + 昨日精华 |

### 📚 术语速查（鼠标悬停 ⓘ 看完整解释）
- **LNN** = Liquid Neural Network 液态神经网络（心智核心）
- **V-A** = Valence-Arousal 情感二维空间（Russell 圆形模型）
- **AU** = Action Unit 面部动作单元（FACS 编码）
- **KG** = Knowledge Graph 知识图谱（bi-temporal 双时态）
- **20 槽** = 万能语法 20 槽（who/what/when/where/why/...）
- **CLS** = Complementary Learning Systems 互补学习（每日精华整合）

### ⚠️ 当前限制（Phase α quick win）
- mock V-A 是图像 hash 出的确定性占位值，**不是真表情识别**（base v3 训练完才换真模型）
- KG 是 InMemory 后端，**刷新页面会丢**（M1 接 graphiti-core 持久化）
- 多模态 state 用 session_state 旁挂存储（Phase β 走 ADR-018 升 schema 持久化）
            """
        )


# ════════════════════════════════════════════════════════════════════
# Snapshot 卡片渲染
# ════════════════════════════════════════════════════════════════════

def render_snapshot_card(snap: SnapshotResult, key_prefix: str = "") -> None:
    """渲染 multimodal SnapshotResult —— 让川哥看到 base v3 LNN 的输入信号。"""
    v, a = snap["v_a"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            label="🧠 LNN 涌现状态",
            value=snap["state_label_zh"],
            help=HELP["9state"] + f"\n\n当前英文 label: `{snap['state_label']}`",
        )
    with col_b:
        st.metric(label="V (Valence 情感极性)", value=f"{v:+.2f}", help=HELP["valence"])
    with col_c:
        st.metric(label="A (Arousal 唤起度)", value=f"{a:+.2f}", help=HELP["arousal"])

    # 数据源 + 置信度（小字诚实标注）
    src = snap["source"]
    src_emoji = "🎭" if src == "mock" else ("📷" if "pyfeat" in src else "⚠️")
    st.caption(
        f"{src_emoji} **数据源**: `{src}` · "
        f"**置信度**: {snap['confidence']:.2f} · "
        f"**人脸检出**: {'是' if snap['face_detected'] else '否'}"
    )

    # 20 AU 折叠展示（gemini #4 警告：撑屏 → expander 收起）
    with st.expander("🔬 查看 20 AU 向量（Action Unit · base v3 多模态输入）", expanded=False):
        st.caption(HELP["au"])
        nonzero = {k: v for k, v in snap["au_vector"].items() if v > 0.01}
        if nonzero:
            st.json(nonzero)
        else:
            st.caption("当前所有 AU 为 0（neutral 平静状态，无明显面部动作）")


# ════════════════════════════════════════════════════════════════════
# 左面板：1️⃣ 输入 + 2️⃣ 抓帧 + 3️⃣ 解析
# ════════════════════════════════════════════════════════════════════

def left_panel() -> None:
    """左面板 —— 输入文本 + 抓拍表情 + 解析写 KG。"""
    st.header("📝 输入侧 · 多模态融合写入")

    # parser 健康检查（启动失败时友好提示）
    if st.session_state.get("parser") is None:
        st.error(
            f"❌ **LLM parser 初始化失败**：{st.session_state.get('parser_error')}\n\n"
            "**修复**：检查环境变量 `GEMINI_PROXY_TOKEN` 是否已设。\n"
            "启动命令应为：\n"
            "```bash\n"
            "GEMINI_PROXY_TOKEN=<token> PYTHONPATH=. .venv-poc-fresh/bin/streamlit run apps/demo_e2e/main.py\n"
            "```"
        )
        return

    # ── 1️⃣ 文本输入 ──
    st.subheader("1️⃣ 输入一句中文事件")
    text = st.text_area(
        label="中文事件文本",
        placeholder="例：昨晚 11 点和小李在公司吃了越南粉",
        help="用自然语言描述一件事，越接近你平常说话的方式越好。LLM 会拆成 20 个语义槽位。",
        key="input_text",
        height=80,
    )
    user_id = st.text_input(
        label="user_id（用户 ID，多用户场景区分）",
        value="chuange",
        help="LnMem 是 per-user 个性化的，每个用户独立 KG + LNN persona 微调",
        key="input_user",
    )

    st.divider()

    # ── 2️⃣ Webcam 抓帧（可选） ──
    st.subheader("2️⃣ 抓拍表情（可选，多模态信号）")
    st.caption(
        "📡 抓帧后会提取 **AU 面部动作 + V-A 情感坐标 + 9-state 状态标签**。"
        "这些不是分类结果，是给未来 base v3 LNN 用的**输入信号**。"
    )

    snap_mode_label = st.radio(
        label="推理模式",
        options=["mock（毫秒级·演示用）", "pyfeat（真模型·首次约 5s）"],
        index=0,
        horizontal=True,
        help=(
            "**mock**：图像 hash → 确定性占位 V-A，毫秒级返回，不是真识别。\n\n"
            "**pyfeat**：调 py-feat 真提取 20 AU，首次需下载 ~150MB 模型，CPU 每次 2-5s。"
            "失败时自动降级 mock。"
        ),
    )
    mode_key = "pyfeat" if snap_mode_label.startswith("pyfeat") else "mock"

    img_file = st.camera_input(
        label="点击下方按钮调用摄像头",
        key="webcam_input",
        help="⚠️ 浏览器会请求摄像头权限。HTTPS 或 localhost 才能调用，公网 HTTP 会被禁",
    )
    if img_file is not None:
        img_bytes = img_file.getvalue()
        with st.spinner(f"提取 AU + V-A + 9-state（{mode_key} 模式）..."):
            snap = snapshot_to_state(img_bytes, mode=mode_key)  # type: ignore[arg-type]
        st.session_state.current_snapshot = snap
        st.success("✅ 多模态信号已捕获，下一步解析时会绑定到这个事件")
        render_snapshot_card(snap, key_prefix="preview")
    elif st.session_state.current_snapshot is not None:
        st.info("📦 上次抓帧的 signal 已暂存，下次解析会自动绑定（除非重新抓帧覆盖）")
        render_snapshot_card(st.session_state.current_snapshot, key_prefix="cached")

    st.divider()

    # ── 3️⃣ 解析 + 写入 KG ──
    st.subheader("3️⃣ 解析 + 写入 KG")
    st.caption("点击下方按钮 → LLM 拆 20 槽 → 多模态 signal 绑定 → 写入 bi-temporal 知识图谱（约 2-5s）")

    if st.button("🚀 解析 + 写入 KG", type="primary", width="stretch"):
        if not text.strip():
            st.warning("⚠️ 请先输入事件文本")
            return

        try:
            with st.spinner("🧠 LLM 抽取万能语法 20 槽..."):
                req = ParseRequest(text=text, user_id=user_id, timestamp=datetime.now(timezone.utc))
                event = st.session_state.parser.parse(req)
        except Exception as exc:
            st.error(
                f"❌ **解析失败**：{exc}\n\n"
                "**可能原因**：\n"
                "- LLM proxy 网络抖动 → 稍后重试\n"
                "- 文本太长（>500 字）→ 缩短后重试\n"
                "- token 失效 → 检查 `GEMINI_PROXY_TOKEN` 环境变量"
            )
            return

        # 多模态 signal 绑定到 event_id（旁挂 session_state）
        if st.session_state.current_snapshot is not None:
            st.session_state.event_multimodal[event.event_id] = st.session_state.current_snapshot

        st.subheader("📋 解析结果 · 20 槽符号投影")
        st.caption(HELP["slot20"])
        st.json(event_to_dict(event))

        with st.spinner("💾 写入 bi-temporal KG..."):
            node_id = st.session_state.adapter.upsert(event)
            st.session_state.events_history.append(event)

        counts = event.filled_priority_count()
        st.success(f"✅ 已写入 KG · node_id = `{node_id[:16]}...`")

        col1, col2 = st.columns(2)
        col1.metric(
            "P0 核心填充率",
            f"{counts['P0']}/4",
            help="P0 = 万能语法核心 4 槽（who 谁 / what 做啥 / when 何时 / where 何地）",
        )
        col2.metric(
            "多模态绑定",
            "✅ 已绑定" if event.event_id in st.session_state.event_multimodal else "—",
            help="若步骤 2 抓了帧，会自动把 V-A + 9-state + AU 关联到此事件",
        )


# ════════════════════════════════════════════════════════════════════
# 右面板：4️⃣ 查询 + 失效 + CLS 精华
# ════════════════════════════════════════════════════════════════════

def render_kg_hit(hit, idx: int) -> None:
    """渲染单条 KG 查询命中 —— 20 槽 + 关联多模态 + 删除按钮。"""
    event = hit.event
    if event.event_id in st.session_state.invalidated_ids:
        return  # 已失效不再渲染

    action = event.get(SlotID.WHAT_ACTION)
    title = action.value if action else "(no what_action)"

    with st.expander(f"#{idx} · score={hit.score:.3f} · {title}", expanded=(idx == 1)):
        # 多模态 state（若有）
        snap = st.session_state.event_multimodal.get(event.event_id)
        if snap is not None:
            st.caption("🧠 **关联的 LNN 涌现状态**（解析时同步抓拍的多模态 signal）")
            render_snapshot_card(snap, key_prefix=f"hit_{idx}")
            st.divider()
        else:
            st.caption("（此事件未绑定多模态 signal）")

        # 20 槽完整展开
        st.caption("📋 **20 槽符号投影** · " + HELP["slot20"][:60] + "...")
        st.json(event_to_dict(event))

        if hit.explanation:
            st.caption(f"💡 检索解释: {hit.explanation}")

        # 删除按钮（极简版，gemini 评审过的 rerun 兜底已加 invalidated_ids）
        col_del, col_info = st.columns([1, 3])
        with col_del:
            if st.button(
                "🗑 失效此事件",
                key=f"del_{event.event_id}",
                help=HELP["invalidate"],
            ):
                t_invalid = datetime.now(timezone.utc)
                st.session_state.adapter.invalidate(event.event_id, t_invalid)
                st.session_state.invalidated_ids.add(event.event_id)
                st.success(f"✅ 已标记失效 · t_invalid = {t_invalid.strftime('%H:%M:%S')}（数据未物理删除）")
                st.rerun()
        with col_info:
            st.caption(f"`event_id: {event.event_id[:16]}...`")


def render_cls_panel(db_path: Path, user_id: str) -> None:
    """CLS 今日精华显示 —— 读 SQLite daily_summaries 表（只读不写）。"""
    st.subheader("🌙 CLS 夜间整合 · 今日精华")
    st.caption(HELP["cls"])

    if not db_path.exists():
        st.info(
            f"📭 CLS DB 尚未生成（路径：`{db_path}`）。\n\n"
            "**生成方式**：\n"
            "```bash\n"
            "GEMINI_PROXY_TOKEN=<token> PYTHONPATH=. \\\n"
            "  .venv-poc-fresh/bin/python scripts/cls_consolidate.py \\\n"
            "  --db data/cls.sqlite --date 2026-05-21\n"
            "```\n"
            "M1 起会改成 02:00 cron 自动跑。"
        )
        return

    try:
        # mode=ro 只读模式，不会与 cls_consolidate.py 写入冲突
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.execute(
            "SELECT date, n_events, highlights_json, llm_model, created_at "
            "FROM daily_summaries WHERE user_id = ? ORDER BY date DESC LIMIT 3",
            (user_id,),
        )
        rows = cur.fetchall()
        conn.close()
    except sqlite3.OperationalError as exc:
        st.warning(
            f"⚠️ CLS DB 读取异常: `{exc}`\n\n"
            "**可能原因**：表尚未建（先跑过一次 `cls_consolidate.py` 才会建表）。"
        )
        return

    if not rows:
        st.caption(f"📭 user_id=`{user_id}` 暂无 CLS 整合记录。今日事件累积满后 02:00 cron 自动跑。")
        return

    for i, (date_str, n_events, highlights_json, llm_model, created_at) in enumerate(rows):
        with st.expander(f"📅 {date_str} · 当日 {n_events} 个 events", expanded=(i == 0)):
            try:
                parsed = json.loads(highlights_json)
                for h in parsed.get("highlights", []):
                    h_type = h.get("type", "?")
                    icon = {"事实": "📌", "情绪": "💭", "未完成": "⏳"}.get(h_type, "🔹")
                    st.markdown(f"{icon} **【{h_type}】** {h.get('summary', '?')}")
                    evidence = h.get("evidence_event_ids", [])
                    if evidence:
                        st.caption(f"📎 证据 event_ids: `{', '.join(evidence)}`")
            except Exception as exc:
                st.warning(f"highlights JSON 解析失败: {exc}")
            st.caption(f"🤖 LLM 模型: `{llm_model}` · 生成时间: {created_at}")


def right_panel() -> None:
    """右面板 —— 查询 KG + 失效卡片 + 节点总览 + CLS 精华。"""
    st.header("🔍 查询侧 · 记忆 UI + 夜间整合")

    # ── 4️⃣ 查询 ──
    st.subheader("4️⃣ 查询 KG")
    query_text = st.text_input(
        label="查询关键词",
        placeholder="例：昨天和谁吃饭",
        help="自然语言查询，会按 user_id + 时间 + 槽位匹配排序",
        key="query_text",
    )
    user_id = st.text_input(
        label="查询的 user_id",
        value="chuange",
        key="query_user",
    )
    top_k = st.slider(
        label="返回前 K 条（top_k）",
        min_value=1,
        max_value=10,
        value=5,
        help="按相关度排序后返回前 K 条候选",
        key="query_topk",
    )

    if st.button("🔎 查询", key="query_btn", width="stretch"):
        query = KGQuery(
            user_id=user_id,
            text_query=query_text,
            valid_at=datetime.now(timezone.utc),  # bi-temporal 时间点过滤
            top_k=top_k,
        )
        hits = st.session_state.adapter.query(query)
        st.session_state.last_hits = hits

    hits = st.session_state.get("last_hits", [])
    if hits:
        active_hits = [h for h in hits if h.event.event_id not in st.session_state.invalidated_ids]
        st.success(f"✅ 查询命中 {len(active_hits)} 条（已自动过滤失效节点）")
        for i, hit in enumerate(active_hits, start=1):
            render_kg_hit(hit, i)

    st.divider()

    # KG 节点总览
    total = len(st.session_state.events_history)
    active = total - len(st.session_state.invalidated_ids)
    st.subheader(f"📊 KG 节点总览（{active} 活跃 / {total} 总数）")
    st.caption("当前 session 写入的所有事件（重启后清零，M1 接 graphiti-core 持久化）")
    if total:
        st.dataframe(events_to_dataframe(st.session_state.events_history), width="stretch")
    else:
        st.info("尚无 KG 节点 —— 左侧输入一句话 → 点 🚀 解析 + 写入 KG")

    st.divider()

    # CLS 精华
    render_cls_panel(CLS_DB_DEFAULT, user_id)


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    """Streamlit demo 主入口 —— 顶部说明 + 左右双面板布局。"""
    if st is None:
        raise RuntimeError("streamlit is not installed. Run: pip install streamlit")

    st.set_page_config(
        page_title="LnMem M0 Demo · Phase α",
        layout="wide",
        page_icon="🧠",
    )
    st.title("🧠 LnMem M0 Demo · Phase α 多模态升级")
    st.caption(
        "**LnMem** = Liquid neural Memory · 用 LNN（液态神经网络）作为永远在线心智核心的伴侣型 AI 记忆系统 · "
        "ADR-016 base v3 多模态范式 quick win 演示"
    )

    # 顶部使用说明（4 步流程引导）
    render_usage_guide()

    st.divider()
    init_state()
    col_l, col_r = st.columns([1, 1])
    with col_l:
        left_panel()
    with col_r:
        right_panel()


if __name__ == "__main__":
    main()
