"""LnMem Intent Console — Streamlit UI v0（ADR-028 v0.1 Sprint 2 钦定）

实时显示：
  - LNN 当前 3-class 意图推断（positive / negative / none）
  - 推断置信度（margin / entropy / softmax probs）
  - 反馈 DAF 弹卡：[✓确认] [✗纠正为...]

数据源：
  - ws://127.0.0.1:9876 (poc/src/realtime/start_inference_server.py)
  - POST http://127.0.0.1:9877/intent_feedback (codex Sprint 2-B 实现)

启动：
  cd /Users/rayman.chen/mycc/2-Projects/LnMem-poc
  .venv/bin/streamlit run streamlit_ui/intent_console.py --server.port 8520

设计原则：
  - 川哥是老板不是标注员（gemini finding 3-1 反馈疲劳防御）
  - 默认静默模式（置信度 < 0.8 不弹卡）
  - 反馈窗口 ≤ 2 分钟（gemini finding 2-2 防记忆污染）
  - 隐私：UI 只显示推断结果，不显示原始 AU 数值
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
import streamlit as st

# ---- 配置 ----
# 5/25 cc fix: daemon 9876 实际是 HTTP（不是 WebSocket），改用 polling /signals/recent
DAEMON_HTTP = "http://127.0.0.1:9876"
SIGNALS_URL = f"{DAEMON_HTTP}/signals/recent?limit=1"
HEALTH_URL = f"{DAEMON_HTTP}/health"
FEEDBACK_URL = "http://127.0.0.1:9877/intent_feedback"
FEEDBACK_WINDOW_SEC = 120  # gemini finding 2-2：反馈有效窗口 ≤ 2min
SILENT_CONFIDENCE_THRESHOLD = 0.8  # gemini finding 3-1：静默模式
HISTORY_MAX = 200  # 历史推断保留 200 条

INTENT_LABELS = {
    # ADR-028 3 类意图（intent_head 训完后才会出现）
    "positive": ("🟢 正反馈", "#10b981"),
    "negative": ("🔴 负反馈", "#ef4444"),
    "none": ("⚪ 无反馈", "#9ca3af"),
    # phase2 backbone 真实 8 类（STATE_LABELS，来自 poc/src/lnn/expression_dynamics.py）
    "emotional_positive": ("😊 情绪-正向", "#10b981"),
    "emotional_negative": ("😟 情绪-负向", "#ef4444"),
    "neutral": ("😐 中性", "#9ca3af"),
    "focused": ("🎯 专注", "#06b6d4"),
    "distracted": ("🌀 走神", "#9333ea"),
    "fatigued": ("😴 疲惫", "#6366f1"),
    "low_energy": ("💤 低能量", "#3b82f6"),
    "transitioning": ("🔄 过渡中", "#ec4899"),
    # 状态
    "warmup": ("🟡 暖机中", "#fbbf24"),
    "error": ("❌ 异常", "#7c3aed"),
}


@dataclass
class Inference:
    ts_ns: int
    intent_pred: str
    intent_confidence: float
    intent_probs: dict  # {"positive": 0.x, "negative": 0.y, "none": 0.z}
    margin: float
    entropy: float
    received_at: float  # local time.time()


# ---- session state 初始化 ----
def init_state():
    if "ui_session" not in st.session_state:
        st.session_state.ui_session = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=HISTORY_MAX)
    if "last_frame_id" not in st.session_state:
        st.session_state.last_frame_id = None
    if "silent_mode" not in st.session_state:
        # 5/25: intent_head 没训之前 intent_pred 全 None，静默会过滤所有 → 强制 OFF
        st.session_state.silent_mode = False
    if "pending_feedback" not in st.session_state:
        # 待反馈的推断队列（uuid → Inference）
        st.session_state.pending_feedback = {}
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "total": 0, "confirmed": 0, "corrected": 0, "ignored": 0,
        }


# ---- HTTP polling daemon /signals/recent ----
def fetch_latest_inference() -> Optional[Inference]:
    """HTTP GET daemon /signals/recent?limit=1，解析 lnn 字段返回推断。

    daemon schema (2026-05-25 实测):
      {"ts": "...", "frame_id": N, "face_detected": bool,
       "lnn": {"status": "ok|warmup|error",
               "predicted_state": "emotional_negative",   # 8 类情绪 (phase2 backbone 训过)
               "intent_pred": null,                       # 3 类意图 (head 没训之前 None)
               "intent_confidence": null,
               "intent_probs": null,
               "margin": 0.98, "entropy": 0.09}}
    """
    try:
        r = requests.get(SIGNALS_URL, timeout=2)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        rec = data[0]
        # 同帧去重（同 frame_id 不重复入队）
        fid = rec.get("frame_id")
        if fid is not None and fid == st.session_state.last_frame_id:
            return None
        st.session_state.last_frame_id = fid

        lnn = rec.get("lnn") or {}
        # 兼容：intent_pred 可能 None（head 没训），fallback 用 predicted_state 让用户能反馈
        intent_pred = lnn.get("intent_pred") or lnn.get("predicted_state") or "warmup"
        intent_confidence = lnn.get("intent_confidence")
        if intent_confidence is None:
            # 用 margin 当兜底 confidence（phase2 backbone 输出）
            intent_confidence = lnn.get("margin") or 0.0

        # ts 转 ns
        from datetime import datetime
        try:
            ts_dt = datetime.fromisoformat(rec["ts"].replace("Z", "+00:00"))
            ts_ns = int(ts_dt.timestamp() * 1e9)
        except Exception:
            ts_ns = int(time.time() * 1e9)

        # daemon 返回 intent_probs 可能是 list (intent_head 启用后) 或 dict (旧路径)
        # 统一规整为 dict {positive, negative, none}（与 poc/src/lnn/intent_head.py INTENT_LABELS 对齐）
        raw_probs = lnn.get("intent_probs")
        if isinstance(raw_probs, list) and len(raw_probs) == 3:
            probs_dict = {"positive": float(raw_probs[0]),
                          "negative": float(raw_probs[1]),
                          "none": float(raw_probs[2])}
        elif isinstance(raw_probs, dict):
            probs_dict = {k: float(v) for k, v in raw_probs.items()}
        else:
            probs_dict = {}

        return Inference(
            ts_ns=ts_ns,
            intent_pred=intent_pred,
            intent_confidence=float(intent_confidence) if intent_confidence is not None else 0.0,
            intent_probs=probs_dict,
            margin=float(lnn.get("margin") or 0.0),
            entropy=float(lnn.get("entropy") or 0.0),
            received_at=time.time(),
        )
    except Exception:
        return None


# ---- 反馈提交 ----
def submit_feedback(inf: Inference, corrected: str, note: str = "") -> bool:
    # 5/25 cc fix: daemon endpoint 限 predicted ∈ {positive,negative,none}
    # intent_head 没训之前 fallback 显示 phase2 8 类（如 transitioning），不能直接传
    valid_3class = {"positive", "negative", "none"}
    predicted = inf.intent_pred if inf.intent_pred in valid_3class else "none"
    # 真实 phase2 state 放 note，便于训练时回溯
    note_full = f"{note}|phase2_state={inf.intent_pred}|conf={inf.intent_confidence:.2f}" if note else f"phase2_state={inf.intent_pred}|conf={inf.intent_confidence:.2f}"
    payload = {
        "ts_ns": inf.ts_ns,
        "predicted": predicted,
        "corrected": corrected,
        "ui_session": st.session_state.ui_session,
        "note": note_full,
    }
    try:
        r = requests.post(FEEDBACK_URL, json=payload, timeout=3)
        r.raise_for_status()
        if corrected == inf.intent_pred:
            st.session_state.stats["confirmed"] += 1
        else:
            st.session_state.stats["corrected"] += 1
        return True
    except Exception as e:
        st.error(f"反馈提交失败：{e}")
        return False


# ---- 主 UI ----
# ---- Studio 远程状态（智能 SSH 路径） ----
@st.cache_data(ttl=60, show_spinner=False)
def detect_studio_alias() -> tuple[str, float, str]:
    """探测可达 Studio SSH alias：优先 ibro（公司内网 iBRO WiFi + 雷电网桥），fallback pub（外网 FRP）。

    场景说明：
    - 川哥在公司 → WiFi=iBRO → studio-ibro 通（Studio 物理在公司）~50ms
    - 川哥在家/外面 → 公网 → studio-pub 通（cli4 FRP + mini-pub:22022）~500-2000ms

    Returns: (alias, latency_s, status_msg) — alias 为 "" 表示都不通
    """
    import subprocess, time as _time
    for alias, label in [("studio-ibro", "🏢 公司内网 iBRO"), ("studio-pub", "🌐 外网 FRP")]:
        start = _time.time()
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=no", alias, "echo PONG"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and "PONG" in r.stdout:
                return alias, _time.time() - start, label
        except Exception:
            pass
    return "", 0.0, "❌ 都不通"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_studio_status(alias: str) -> dict:
    """ssh 拉 Studio 远端实时状态。失败返 {}"""
    if not alias:
        return {}
    import subprocess
    cmd = """
DATA_BYTES=$(du -sb ~/lnmem-daemon-sync/ 2>/dev/null | awk '{print $1}')
FEEDBACK_COUNT=$(cat ~/lnmem-daemon-sync/intent_feedback/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
CURSOR=$(cat ~/lnmem-train/.last_trained_feedback_count 2>/dev/null)
LATEST_LINK=$(readlink ~/lnmem-train/ckpt/intent_head_latest.pt 2>/dev/null)
LATEST_TS=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' ~/lnmem-train/ckpt/$LATEST_LINK 2>/dev/null)
CKPT_COUNT=$(ls ~/lnmem-train/ckpt/intent_head_*.pt 2>/dev/null | wc -l | tr -d ' ')
TODAY=$(date +%Y-%m-%d)
LOG_TAIL=$(tail -10 ~/lnmem-train/logs/train/nightly-${TODAY}.log 2>/dev/null | tail -10)
printf 'DATA_BYTES=%s\\nFEEDBACK_COUNT=%s\\nCURSOR=%s\\nLATEST_LINK=%s\\nLATEST_TS=%s\\nCKPT_COUNT=%s\\n--LOG_TAIL--\\n%s' \\
    "$DATA_BYTES" "$FEEDBACK_COUNT" "$CURSOR" "$LATEST_LINK" "$LATEST_TS" "$CKPT_COUNT" "$LOG_TAIL"
"""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", alias, cmd],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return {"error": r.stderr[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}

    out = r.stdout
    result = {}
    if "--LOG_TAIL--" in out:
        head, _, tail = out.partition("--LOG_TAIL--")
        for line in head.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                result[k] = v.strip()
        result["LOG_TAIL"] = tail.strip()
    return result


def render_studio_panel():
    """Streamlit sidebar 面板：Studio 远程状态"""
    with st.sidebar:
        st.markdown("### 📡 Studio 远程状态")
        alias, latency, label = detect_studio_alias()
        if not alias:
            st.error(f"❌ Studio 都不通\n\n试: `ssh studio-pub` / `ssh studio-ibro`")
            if st.button("🔄 重试探测"):
                st.cache_data.clear()
                st.rerun()
            return

        st.success(f"{label} `{alias}` ({latency*1000:.0f}ms)")
        if st.button("🔄 刷新 Studio 状态", help="清缓存重拉"):
            st.cache_data.clear()
            st.rerun()

        s = fetch_studio_status(alias)
        if "error" in s:
            st.warning(f"拉取失败: {s['error']}")
            return
        if not s:
            st.caption("无数据")
            return

        # 数据准备
        st.markdown("**📊 数据准备**")
        data_mb = int(s.get("DATA_BYTES", "0") or 0) / 1024 / 1024
        feedback_count = s.get("FEEDBACK_COUNT", "?")
        st.text(f"daemon-sync: {data_mb:.0f} MB")
        st.text(f"intent_feedback: {feedback_count} 条")

        # 训练进展
        st.markdown("**🎯 训练进展**")
        st.caption(
            "💡 Studio 每 8 小时（00:00/08:00/16:00）尝试训练一次，需要同时满足两个门："
            "**时间门**（距上次训练 ≥ 6 小时）+ **增量门**（积累了 ≥ 10 条新反馈）。"
        )
        cursor = s.get("CURSOR", "")
        if cursor and ":" in cursor:
            cnt, ts = cursor.split(":")
            from datetime import datetime
            try:
                ago_h = (datetime.now().timestamp() - int(ts)) / 3600
                st.text(f"⏱ 上次成功训练: {ago_h:.1f} 小时前")
                st.caption(f"（那次训练用了你截至当时的 {cnt} 条反馈）")
            except:
                st.text(f"cursor: {cursor}")
            # 计算下次能训的时间
            try:
                next_train_h = max(0, 6 - ago_h)  # 时间门 6h
                if next_train_h > 0:
                    st.text(f"🔒 时间门：{next_train_h:.1f} 小时后解锁")
                else:
                    st.text(f"🔓 时间门：✅ 已解锁")
            except: pass
        else:
            st.text("还没训过（等首次反馈累积）")

        # 增量门进度
        try:
            new_feedback = int(feedback_count) - int(cursor.split(":")[0]) if cursor and ":" in cursor else int(feedback_count)
            st.text(f"📈 增量门：新增反馈 {new_feedback} / 10 {'✅ 已满足' if new_feedback >= 10 else '⏳ 待累积'}")
        except: pass

        # ckpt = checkpoint = 训练完的模型权重快照
        st.markdown("**📦 ckpt（模型权重快照文件）**")
        st.caption(
            "💡 ckpt = checkpoint，每次 Studio 训练完会保存一个 `.pt` 文件，"
            "存这次学到的 LNN 参数。Air daemon 会自动拉最新 ckpt 加载，"
            "你的反馈就转化成了 LNN 的'记忆'。"
        )
        st.text(f"最新文件: {s.get('LATEST_LINK', '?')}")
        st.text(f"生成时间: {s.get('LATEST_TS', '?')}")
        st.text(f"历史总数: {s.get('CKPT_COUNT', '0')} 个")

        # 最近 log
        log_tail = s.get("LOG_TAIL", "")
        if log_tail:
            with st.expander("📜 最近训练 log"):
                st.code(log_tail, language="text")


def main():
    st.set_page_config(page_title="LnMem Intent Console", layout="wide", page_icon="🧠")
    init_state()

    # 侧边栏：Studio 远程状态（智能 SSH 路径）
    render_studio_panel()

    st.markdown("""
    <style>
    .intent-card { padding: 1.2rem; border-radius: 12px; margin-bottom: 0.5rem; }
    .big-label { font-size: 2.2rem; font-weight: 700; }
    .ts-label { font-size: 0.85rem; color: #6b7280; }
    </style>
    """, unsafe_allow_html=True)

    # ---- 顶部：状态条 ----
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown("### 🧠 LnMem Intent Console v0")
        st.caption(f"session: `{st.session_state.ui_session[:8]}...`  |  ADR-028 v0.1")
    with col2:
        st.metric("反馈总数", st.session_state.stats["total"])
    with col3:
        st.metric("确认", st.session_state.stats["confirmed"])
    with col4:
        st.metric("纠正", st.session_state.stats["corrected"])

    # ---- 控制条 ----
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        st.session_state.silent_mode = st.toggle(
            f"静默模式（仅置信度 > {SILENT_CONFIDENCE_THRESHOLD} 弹卡）",
            value=st.session_state.silent_mode,
            help="开：减少打扰；关：所有推断都弹卡（标注模式）",
        )
    with ctrl2:
        auto_refresh = st.toggle("自动刷新 (1s)", value=True)
    with ctrl3:
        if st.button("🔄 重置历史"):
            st.session_state.history.clear()
            st.session_state.last_frame_id = None
            st.rerun()

    # ---- 拉取最新推断 ----
    inf = fetch_latest_inference()
    if inf is not None:
        st.session_state.history.append(inf)
        st.session_state.stats["total"] += 1

        # 静默模式过滤
        should_prompt = (
            not st.session_state.silent_mode
            or inf.intent_confidence >= SILENT_CONFIDENCE_THRESHOLD
        ) and inf.intent_pred not in ("warmup", "error", "none")

        if should_prompt:
            st.session_state.pending_feedback[inf.ts_ns] = inf

    # ---- 主区：当前推断卡 ----
    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("📡 当前推断")
        if st.session_state.history:
            current = st.session_state.history[-1]
            label, color = INTENT_LABELS.get(current.intent_pred, ("未知", "#000"))
            st.markdown(
                f'<div class="intent-card" style="background:{color}22; border-left: 4px solid {color};">'
                f'<div class="big-label" style="color:{color};">{label}</div>'
                f'<div class="ts-label">置信度 {current.intent_confidence:.2f} · '
                f'margin {current.margin:.3f} · entropy {current.entropy:.3f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if current.intent_probs:
                probs_str = " · ".join([f"{k}: {v:.2f}" for k, v in current.intent_probs.items()])
                st.caption(f"三类概率分布: {probs_str}")
            with st.expander("ℹ️ 这些数字什么意思？"):
                st.markdown("""
- **置信度 (confidence)**：模型对本次判断的确定程度，越接近 1.0 越确定
- **margin**：top-1 概率减 top-2 概率的差值，越大说明 8 类表情判断越"果断"（≥0.6 强 / 0.3-0.6 中 / <0.3 模糊）
- **entropy（熵）**：分布的混乱度，0=完全确定一类，越大说明模型越纠结（>0.6 触发主动求反馈 B）
- **三类概率分布**：positive / negative / none 的预测概率，三个数加起来 = 1.0
- **intent_pred**：模型最终猜的意图（取概率最大的那个）
                """)
        else:
            st.info(f"等待 daemon 数据… 检查 `curl {SIGNALS_URL}` 是否有 lnn 字段")

    with right:
        st.subheader("✋ 待反馈队列")
        if not st.session_state.pending_feedback:
            st.caption("暂无待反馈")
        else:
            # 清理过期（>FEEDBACK_WINDOW_SEC）
            now = time.time()
            expired = []
            for ts_ns, inf in list(st.session_state.pending_feedback.items())[-3:]:  # 最多显示最近 3 张
                age = now - inf.received_at
                if age > FEEDBACK_WINDOW_SEC:
                    expired.append(ts_ns)
                    continue
                label, color = INTENT_LABELS.get(inf.intent_pred, ("未知", "#000"))
                with st.container():
                    st.markdown(
                        f'**AI 推断 [{label}] @ '
                        f'{datetime.fromtimestamp(inf.ts_ns / 1e9).strftime("%H:%M:%S")}** '
                        f'({age:.0f}s ago)'
                    )
                    bcols = st.columns(4)
                    with bcols[0]:
                        if st.button("✓ 确认", key=f"conf_{ts_ns}"):
                            if submit_feedback(inf, inf.intent_pred):
                                del st.session_state.pending_feedback[ts_ns]
                                st.rerun()
                    with bcols[1]:
                        if st.button("→正", key=f"pos_{ts_ns}", disabled=inf.intent_pred == "positive"):
                            if submit_feedback(inf, "positive"):
                                del st.session_state.pending_feedback[ts_ns]
                                st.rerun()
                    with bcols[2]:
                        if st.button("→负", key=f"neg_{ts_ns}", disabled=inf.intent_pred == "negative"):
                            if submit_feedback(inf, "negative"):
                                del st.session_state.pending_feedback[ts_ns]
                                st.rerun()
                    with bcols[3]:
                        if st.button("→无", key=f"none_{ts_ns}", disabled=inf.intent_pred == "none"):
                            if submit_feedback(inf, "none"):
                                del st.session_state.pending_feedback[ts_ns]
                                st.rerun()
            for ts_ns in expired:
                del st.session_state.pending_feedback[ts_ns]
                st.session_state.stats["ignored"] += 1

    # ---- 历史折线 ----
    st.markdown("---")
    st.subheader("📈 近 200 条推断历史（纵轴 = 置信度 0~1）")
    st.caption(
        "**纵轴含义**：每帧 LNN 推断的 `intent_confidence`（置信度），范围 0.0-1.0。"
        "越接近 1.0 = 模型对本帧判断越确定；越接近 0.0 = 越不确定（接近随机猜）。"
        "经验阈值：≥0.85 高可信，0.70-0.85 中等，<0.70 弱信号（notify_watcher 弹窗也用此阈值过滤）。"
    )
    if len(st.session_state.history) >= 5:
        import pandas as pd
        import altair as alt
        df = pd.DataFrame([
            {"ts": datetime.fromtimestamp(h.ts_ns / 1e9),
             "confidence": h.intent_confidence,
             "intent": h.intent_pred}
            for h in st.session_state.history
        ])
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("ts:T", title="时间"),
                y=alt.Y(
                    "confidence:Q",
                    title="置信度 (0=不确定, 1=完全确定)",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=["ts:T", "confidence:Q", "intent:N"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)
        # 意图分布
        from collections import Counter
        dist = Counter([h.intent_pred for h in st.session_state.history])
        st.caption(f"意图分布: {dict(dist)}")
    else:
        st.caption(f"已积累 {len(st.session_state.history)} 条，攒到 5 条再画图")

    # ---- auto refresh ----
    if auto_refresh:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
