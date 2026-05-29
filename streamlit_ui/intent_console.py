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
SIGNALS_URL = f"{DAEMON_HTTP}/signals/recent?seconds=30"  # ADR-041: daemon 不认 limit; 实测 daemon 帧率抖动大(只在检测到人脸时记帧, 常 idle 10+s), seconds=30 留充足容差; data[-1] 总取最新, ts_ns 锚到该帧时刻 (即使是几秒到几十秒前, 仍精确, 修了原 5min 错位 bug)
HEALTH_URL = f"{DAEMON_HTTP}/health"
FEEDBACK_URL = "http://127.0.0.1:9877/intent_feedback"
FEEDBACK_WINDOW_SEC = 120  # gemini finding 2-2：反馈有效窗口 ≤ 2min
SILENT_CONFIDENCE_THRESHOLD = 0.8  # gemini finding 3-1：静默模式
HISTORY_MAX = 200  # 历史推断保留 200 条

INTENT_LABELS = {
    # ADR-028 3 类意图（intent_head 训完后才会出现）
    "positive": ("🟢 正反馈", "#10b981"),
    "negative": ("🔴 负反馈", "#ef4444"),
    "none": ("⚪ 中性", "#9ca3af"),
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
    phase2_state: str = "unknown"  # 8 类情绪 top-1 (emotional_negative / focused / ...)
    top_prob: float = 0.0          # phase2 top-1 概率
    state_probs: list = None       # 完整 8 类概率（与 STATE_LABELS 顺序一致）


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
        rec = data[-1]  # ADR-041: daemon /signals/recent 返回 oldest→newest, 必须取 [-1] 拿最新, 否则 ts_ns 锚到 5min 前最旧帧
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
            phase2_state=str(lnn.get("predicted_state") or "unknown"),
            top_prob=float(lnn.get("top_prob") or 0.0),
            state_probs=lnn.get("state_probs") or None,
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


@st.cache_data(ttl=10, show_spinner=False)
def fetch_ltc_predictions(steps: int = 10) -> tuple[list[dict], str]:
    """拉 daemon /predict_future 真 LNN 预测，返 (predictions, method)。

    fallback: 失败返 ([], "mean_extrapolation") 让 UI 降级用 mean 外推。
    """
    import requests
    try:
        r = requests.get(f"http://127.0.0.1:9876/predict_future?steps={steps}", timeout=2)
        if r.status_code == 200:
            data = r.json()
            preds = data.get("predictions", [])
            # 第一项是 error dict → fallback
            if preds and "error" in preds[0]:
                return [], f"mean_fallback ({preds[0]['error']})"
            return preds, data.get("method", "ltc_ode_persistence")
        elif r.status_code == 503:
            return [], "mean_fallback (endpoint disabled)"
    except Exception as e:
        return [], f"mean_fallback ({type(e).__name__})"
    return [], "mean_fallback (unknown)"


def render_evolution_panel(history):
    """历史表情 + 意图演化曲线 + LTC ODE 真预测（fallback 到 mean 外推）"""
    if len(history) < 10:
        st.info(f"演化曲线需要 ≥10 帧数据（当前 {len(history)}）")
        return

    import pandas as pd
    import altair as alt
    from datetime import datetime, timedelta

    # 拉真 LNN 预测
    ltc_preds, method = fetch_ltc_predictions(steps=10)
    use_ltc = bool(ltc_preds)

    # 显示预测方法标识
    if use_ltc:
        st.success(f"🧠 预测方法: **LTC ODE 真预测**（method={method}），右侧 10 帧来自 LNN 微分方程演化")
    else:
        st.warning(f"⚠️ 预测方法: **mean 外推 fallback**（{method}），LTC endpoint 不可用")

    # ---- 1. 意图概率演化 (3 类 stacked area + 真预测 or mean fallback) ----
    rows = []
    for h in history:
        ts = datetime.fromtimestamp(h.ts_ns / 1e9)
        for cls in ("positive", "negative", "none"):
            rows.append({
                "ts": ts,
                "意图类别": {"positive": "🟢 正反馈", "negative": "🔴 负反馈", "none": "⚪ 中性"}[cls],
                "概率": h.intent_probs.get(cls, 0.0),
                "是否预测": False,
            })

    last_ts = datetime.fromtimestamp(history[-1].ts_ns / 1e9)

    if use_ltc:
        # 真 LTC 预测：每帧有完整 intent_probs
        cls_names = ("positive", "negative", "none")
        for i, p in enumerate(ltc_preds[:10], 1):
            future_ts = last_ts + timedelta(seconds=i)
            ip = p.get("intent_probs")
            if not ip or len(ip) != 3:
                continue
            for j, cls in enumerate(cls_names):
                rows.append({
                    "ts": future_ts,
                    "意图类别": {"positive": "🟢 正反馈", "negative": "🔴 负反馈", "none": "⚪ 中性"}[cls],
                    "概率": float(ip[j]),
                    "是否预测": True,
                })
    else:
        # fallback: 用最近 10 帧 mean 外推
        recent = list(history)[-10:]
        mean_probs = {
            cls: sum(h.intent_probs.get(cls, 0.0) for h in recent) / len(recent)
            for cls in ("positive", "negative", "none")
        }
        for i in range(1, 11):
            future_ts = last_ts + timedelta(seconds=i)
            for cls in ("positive", "negative", "none"):
                rows.append({
                    "ts": future_ts,
                    "意图类别": {"positive": "🟢 正反馈", "negative": "🔴 负反馈", "none": "⚪ 中性"}[cls],
                    "概率": mean_probs[cls],
                    "是否预测": True,
                })

    df_intent = pd.DataFrame(rows)
    # 川哥钦定：预测占右 1/3 → 历史显示最近 2× 预测时长（10s 预测 → 20s 历史 → 总 30s 窗口）
    intent_pred_duration = 10  # LTC 预测时长
    intent_history_window = intent_pred_duration * 2
    intent_x_domain_start = last_ts - timedelta(seconds=intent_history_window)
    intent_x_domain_end = last_ts + timedelta(seconds=intent_pred_duration)

    # 主面积图
    intent_area = (
        alt.Chart(df_intent)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("ts:T", title=None,
                    scale=alt.Scale(domain=[intent_x_domain_start, intent_x_domain_end], clamp=True)),
            y=alt.Y("概率:Q", stack="normalize", title="意图概率分布"),
            color=alt.Color(
                "意图类别:N",
                scale=alt.Scale(
                    domain=["🟢 正反馈", "🔴 负反馈", "⚪ 中性"],
                    range=["#10b981", "#ef4444", "#9ca3af"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            opacity=alt.condition(alt.datum["是否预测"], alt.value(0.45), alt.value(0.92)),
            tooltip=["ts:T", "意图类别:N", alt.Tooltip("概率:Q", format=".3f"), "是否预测:N"],
        )
    )
    # 预测段灰黄背景
    future_max_ts = last_ts + timedelta(seconds=10)
    intent_bg = (
        alt.Chart(pd.DataFrame([{"x_start": last_ts, "x_end": future_max_ts}]))
        .mark_rect(color="#fbbf24", opacity=0.08)
        .encode(x="x_start:T", x2="x_end:T")
    )
    # "现在" 垂直红虚线
    intent_now_rule = (
        alt.Chart(pd.DataFrame([{"x": last_ts}]))
        .mark_rule(color="#dc2626", strokeWidth=2, strokeDash=[6, 3])
        .encode(x="x:T")
    )
    intent_now_text = (
        alt.Chart(pd.DataFrame([{"x": last_ts}]))
        .mark_text(text="⏱ 现在", align="left", dx=4, dy=-4,
                   color="#dc2626", fontSize=12, fontWeight="bold")
        .encode(x="x:T", y=alt.value(0))
    )
    intent_future_text = (
        alt.Chart(pd.DataFrame([{"x": last_ts + timedelta(seconds=5)}]))
        .mark_text(text=("🔮 LTC 真预测" if use_ltc else "🔮 mean fallback"),
                   align="center", dy=14, color="#92400e",
                   fontSize=12, fontWeight="bold", fontStyle="italic")
        .encode(x="x:T", y=alt.value(0))
    )

    intent_chart = (intent_bg + intent_area + intent_now_rule + intent_now_text + intent_future_text).properties(
        height=200,
        title=f"🎯 意图概率演化（红虚线 = '现在'，右侧黄色区 = {'LTC 真预测' if use_ltc else 'mean fallback'}）"
    )
    st.altair_chart(intent_chart, use_container_width=True)

    # ---- 2. 表情状态时间线 (ADR-033 分情绪 + 注意力两组 + valence) ----
    # 模型层仍 8 类，UI 层 mapping 到两组
    state_meta = {
        "emotional_positive": {"group": "情绪", "label": "😊 情绪-正向", "valence": "✅ 正向", "color": "#10b981"},
        "emotional_negative": {"group": "情绪", "label": "😟 情绪-负向", "valence": "❌ 负向", "color": "#ef4444"},
        "neutral":            {"group": "情绪", "label": "😐 中立",     "valence": "⚪ 中性", "color": "#9ca3af"},
        "focused":            {"group": "注意力", "label": "🎯 专注",   "valence": "✅ 正向", "color": "#3b82f6"},
        "distracted":         {"group": "注意力", "label": "🌀 分心",   "valence": "⚠️ 偏负", "color": "#f59e0b"},
        "fatigued":           {"group": "注意力", "label": "😴 疲惫",   "valence": "❌ 负向", "color": "#a78bfa"},
        "low_energy":         {"group": "注意力", "label": "🪫 低能量", "valence": "❌ 负向", "color": "#6b7280"},
        "transitioning":      {"group": "注意力", "label": "🔄 过渡中", "valence": "⚪ 中性", "color": "#ec4899"},
        "unknown":            {"group": "其他",   "label": "❓ 未知",   "valence": "❓",      "color": "#000000"},
    }

    def _build_rows(history, group_filter):
        """构造某分组的色块行"""
        rows = []
        for i, h in enumerate(history):
            meta = state_meta.get(h.phase2_state, state_meta["unknown"])
            if meta["group"] != group_filter:
                continue
            ts = datetime.fromtimestamp(h.ts_ns / 1e9)
            next_ts = (datetime.fromtimestamp(history[i+1].ts_ns / 1e9)
                       if i + 1 < len(history) else ts + timedelta(seconds=1))
            rows.append({
                "ts_start": ts, "ts_end": next_ts,
                "状态": meta["label"], "valence": meta["valence"],
                "置信度": h.top_prob, "是否预测": False,
            })
        return rows

    # 外推：优先用 LTC 真预测 5 步，fallback 最近 5 帧 mode
    last_ts_dt = datetime.fromtimestamp(history[-1].ts_ns / 1e9)
    pred_rows = []
    if use_ltc:
        # LTC 每帧给出 predicted_state，把连续相同 state 合成一个色块
        prev_state = None
        block_start = last_ts_dt
        for i, p in enumerate(ltc_preds[:5], 1):
            st_name = p.get("predicted_state", "unknown")
            ts_i = last_ts_dt + timedelta(seconds=i)
            if prev_state is None:
                prev_state = st_name; block_start = last_ts_dt
            elif st_name != prev_state:
                # 切块
                meta = state_meta.get(prev_state, state_meta["unknown"])
                pred_rows.append({
                    "ts_start": block_start, "ts_end": ts_i,
                    "状态": meta["label"], "valence": meta["valence"],
                    "置信度": float(p.get("top_prob", 0.0)), "是否预测": True,
                })
                prev_state = st_name; block_start = ts_i
        # 最后一块
        if prev_state is not None:
            meta = state_meta.get(prev_state, state_meta["unknown"])
            pred_rows.append({
                "ts_start": block_start, "ts_end": last_ts_dt + timedelta(seconds=5),
                "状态": meta["label"], "valence": meta["valence"],
                "置信度": float(ltc_preds[-1].get("top_prob", 0.0)) if ltc_preds else 0.0,
                "是否预测": True,
            })
    else:
        from collections import Counter
        recent_5 = [h.phase2_state for h in list(history)[-5:]]
        pred_state = Counter(recent_5).most_common(1)[0][0] if recent_5 else "unknown"
        pred_meta = state_meta.get(pred_state, state_meta["unknown"])
        pred_rows.append({
            "ts_start": last_ts_dt, "ts_end": last_ts_dt + timedelta(seconds=5),
            "状态": pred_meta["label"], "valence": pred_meta["valence"],
            "置信度": 0.0, "是否预测": True,
        })

    # STATE_LABELS 必须与 daemon 端 lnn_inferencer 一致（来自 expression_dynamics.py）
    STATE_LABELS_ORDER = (
        "low_energy", "focused", "distracted", "fatigued",
        "emotional_positive", "emotional_negative", "neutral", "transitioning",
    )

    def _render_group_chart(group_name, group_emoji, height=240):
        """Stacked area 概率图：每个 state 一条色带，y=该 state 概率，总和=1.0"""
        group_states = [s for s in STATE_LABELS_ORDER if state_meta.get(s, {}).get("group") == group_name]
        if not group_states:
            return

        # 历史段：从 h.state_probs 取该 group 内每个 state 的概率
        rows = []
        for h in history:
            if not h.state_probs or len(h.state_probs) != 8:
                # 旧数据没 state_probs → 用 top-1 fake：top_prob 给 top state，其余等分剩余
                top_state = h.phase2_state
                remaining = (1.0 - h.top_prob) / max(1, len(STATE_LABELS_ORDER) - 1)
                state_probs_dict = {
                    s: (h.top_prob if s == top_state else remaining)
                    for s in STATE_LABELS_ORDER
                }
            else:
                state_probs_dict = dict(zip(STATE_LABELS_ORDER, h.state_probs))

            ts = datetime.fromtimestamp(h.ts_ns / 1e9)
            for s in group_states:
                meta = state_meta[s]
                rows.append({
                    "ts": ts,
                    "状态": meta["label"],
                    "valence": meta["valence"],
                    "概率": state_probs_dict.get(s, 0.0),
                    "是否预测": False,
                })

        # 预测段：从 ltc_preds[i].state_probs 取
        if use_ltc:
            for i, p in enumerate(ltc_preds[:10], 1):
                sp = p.get("state_probs")
                if not sp or len(sp) != 8:
                    continue
                state_probs_dict = dict(zip(STATE_LABELS_ORDER, sp))
                future_ts = last_ts_dt + timedelta(seconds=i)
                for s in group_states:
                    meta = state_meta[s]
                    rows.append({
                        "ts": future_ts,
                        "状态": meta["label"],
                        "valence": meta["valence"],
                        "概率": state_probs_dict.get(s, 0.0),
                        "是否预测": True,
                    })

        if not rows:
            st.caption(f"{group_emoji} {group_name}: 本批历史无此类记录")
            return

        df = pd.DataFrame(rows)
        group_labels = [state_meta[s]["label"] for s in group_states]
        group_colors = [state_meta[s]["color"] for s in group_states]

        # 川哥钦定：预测占右 1/3 → 历史显示最近 2× 预测时长
        group_pred_duration = 10 if use_ltc else 5
        group_history_window = group_pred_duration * 2
        group_x_start = last_ts_dt - timedelta(seconds=group_history_window)
        group_x_end = last_ts_dt + timedelta(seconds=group_pred_duration)

        # 主图：stacked area
        main_chart = (
            alt.Chart(df)
            .mark_area(opacity=0.85)
            .encode(
                x=alt.X("ts:T", title=None,
                        scale=alt.Scale(domain=[group_x_start, group_x_end], clamp=True)),
                y=alt.Y("概率:Q", stack="normalize",
                        title=f"{group_name}组内概率占比 (总=100%)"),
                color=alt.Color(
                    "状态:N",
                    scale=alt.Scale(domain=group_labels, range=group_colors),
                    legend=alt.Legend(orient="top", columns=len(group_labels), title=None),
                ),
                opacity=alt.condition(alt.datum["是否预测"],
                                       alt.value(0.45),  # 预测段半透明
                                       alt.value(0.92)),  # 历史段实色
                tooltip=["ts:T", "状态:N", "valence:N",
                         alt.Tooltip("概率:Q", format=".3f"), "是否预测:N"],
            )
        )

        # 预测段背景灰带（用一帧矩形覆盖预测时间范围）
        future_max_ts = last_ts_dt + timedelta(seconds=10 if use_ltc else 5)
        bg_df = pd.DataFrame([{"x_start": last_ts_dt, "x_end": future_max_ts}])
        bg_layer = (
            alt.Chart(bg_df)
            .mark_rect(color="#fbbf24", opacity=0.08)  # 浅黄背景
            .encode(x="x_start:T", x2="x_end:T")
        )

        # "现在" 垂直分割线
        now_line_df = pd.DataFrame([{"x": last_ts_dt}])
        now_rule = (
            alt.Chart(now_line_df)
            .mark_rule(color="#dc2626", strokeWidth=2, strokeDash=[6, 3])
            .encode(x="x:T")
        )

        # 顶部"现在"标签
        now_text = (
            alt.Chart(now_line_df)
            .mark_text(text="⏱ 现在", align="left", dx=4, dy=-4,
                       color="#dc2626", fontSize=12, fontWeight="bold")
            .encode(x="x:T", y=alt.value(0))
        )

        # 预测段顶部 "🔮 LTC 预测" 标签
        if use_ltc or pred_rows:
            mid_future = last_ts_dt + timedelta(seconds=5 if use_ltc else 2.5)
            label_text = "🔮 LTC 真预测" if use_ltc else "🔮 mean fallback"
            future_text = (
                alt.Chart(pd.DataFrame([{"x": mid_future}]))
                .mark_text(text=label_text, align="center", dy=14,
                           color="#92400e", fontSize=12, fontWeight="bold",
                           fontStyle="italic")
                .encode(x="x:T", y=alt.value(0))
            )
        else:
            future_text = alt.Chart(pd.DataFrame()).mark_text()

        # 历史段 "← 历史" 标签
        if len(history) > 0:
            mid_history = datetime.fromtimestamp(
                history[max(0, len(history) // 2)].ts_ns / 1e9
            )
            history_text = (
                alt.Chart(pd.DataFrame([{"x": mid_history}]))
                .mark_text(text="← 历史", align="center", dy=14,
                           color="#1f2937", fontSize=12, fontWeight="bold")
                .encode(x="x:T", y=alt.value(0))
            )
        else:
            history_text = alt.Chart(pd.DataFrame()).mark_text()

        chart = (bg_layer + main_chart + now_rule + now_text + future_text + history_text).properties(
            height=height,
            title=f"{group_emoji} {group_name} 概率演化（红虚线 = '现在'，右侧黄色区 = 预测）"
        )
        st.altair_chart(chart, use_container_width=True)
        return  # 下面的旧 mark_line 代码忽略

        # === 以下保留旧 mark_line 代码作为参考，永不执行 ===
        # 历史点
        history_pts = []
        for h in history:
            meta = state_meta.get(h.phase2_state, state_meta["unknown"])
            if meta["group"] != group_name:
                continue
            history_pts.append({
                "ts": datetime.fromtimestamp(h.ts_ns / 1e9),
                "置信度": h.top_prob,
                "状态": meta["label"],
                "valence": meta["valence"],
                "是否预测": False,
            })
        # 预测点（从 ltc_preds 取这个 group 的每步预测）
        pred_pts = []
        if use_ltc:
            for i, p in enumerate(ltc_preds[:10], 1):
                st_name = p.get("predicted_state", "unknown")
                meta = state_meta.get(st_name, state_meta["unknown"])
                if meta["group"] != group_name:
                    continue
                pred_pts.append({
                    "ts": last_ts_dt + timedelta(seconds=i),
                    "置信度": float(p.get("top_prob", 0.0)),
                    "状态": meta["label"],
                    "valence": meta["valence"],
                    "是否预测": True,
                })
        elif pred_rows:
            for pr in pred_rows:
                for orig_state, m in state_meta.items():
                    if m["label"] == pr["状态"] and m["group"] == group_name:
                        pred_pts.append({
                            "ts": pr["ts_start"],
                            "置信度": pr["置信度"],
                            "状态": pr["状态"],
                            "valence": pr["valence"],
                            "是否预测": True,
                        })
                        break

        rows = history_pts + pred_pts
        if not rows:
            st.caption(f"{group_emoji} {group_name}: 本批历史无此类记录")
            return

        df = pd.DataFrame(rows)
        group_labels = [m["label"] for m in state_meta.values() if m["group"] == group_name]
        group_colors = [m["color"] for m in state_meta.values() if m["group"] == group_name]

        # 历史段：实线 + 点
        history_chart = (
            alt.Chart(df)
            .transform_filter(alt.datum["是否预测"] == False)
            .mark_line(interpolate="step-after", strokeWidth=2.5)
            .encode(
                x=alt.X("ts:T", title=None),
                y=alt.Y("置信度:Q", scale=alt.Scale(domain=[0, 1]), title="state 置信度 (0-1)"),
                color=alt.Color(
                    "状态:N",
                    scale=alt.Scale(domain=group_labels, range=group_colors),
                    legend=alt.Legend(orient="top", columns=len(group_labels), title=None),
                ),
                tooltip=["ts:T", "状态:N", "valence:N",
                         alt.Tooltip("置信度:Q", format=".3f"), "是否预测:N"],
            )
        )
        history_dots = (
            alt.Chart(df)
            .transform_filter(alt.datum["是否预测"] == False)
            .mark_circle(size=30)
            .encode(
                x="ts:T", y="置信度:Q",
                color=alt.Color("状态:N", scale=alt.Scale(domain=group_labels, range=group_colors), legend=None),
                tooltip=["ts:T", "状态:N", alt.Tooltip("置信度:Q", format=".3f")],
            )
        )
        # 预测段：虚线 + 半透明点
        pred_chart = (
            alt.Chart(df)
            .transform_filter(alt.datum["是否预测"] == True)
            .mark_line(interpolate="step-after", strokeWidth=2, strokeDash=[5, 3], opacity=0.6)
            .encode(
                x="ts:T", y="置信度:Q",
                color=alt.Color("状态:N", scale=alt.Scale(domain=group_labels, range=group_colors), legend=None),
                tooltip=["ts:T", "状态:N", alt.Tooltip("置信度:Q", format=".3f"), "是否预测:N"],
            )
        )
        pred_dots = (
            alt.Chart(df)
            .transform_filter(alt.datum["是否预测"] == True)
            .mark_circle(size=40, opacity=0.5)
            .encode(x="ts:T", y="置信度:Q",
                    color=alt.Color("状态:N", scale=alt.Scale(domain=group_labels, range=group_colors), legend=None))
        )
        combined = (history_chart + history_dots + pred_chart + pred_dots).properties(
            height=height,
            title=f"{group_emoji} {group_name} 演化（实线=历史，虚线=右侧 {'10s LTC 预测' if use_ltc else '5s mean fallback'}，颜色=state，y=top_prob）"
        )
        st.altair_chart(combined, use_container_width=True)

    _render_group_chart("情绪", "😊", height=200)
    _render_group_chart("注意力", "🎯", height=200)

    # valence 图例说明
    st.caption(
        "🎨 **valence 标签**："
        "✅ 正向（专注/情绪-正向）· "
        "❌ 负向（疲惫/低能量/情绪-负向）· "
        "⚠️ 偏负（分心）· "
        "⚪ 中性（中立/过渡中）"
    )

    # 说明
    with st.expander("ℹ️ 预测是怎么算的？"):
        st.markdown(f"""
**当前算法：{'🧠 LTC ODE 真预测' if use_ltc else '⚠️ mean 外推 fallback'}**（ADR-033）

### LTC ODE 真预测（推荐）
- daemon `/predict_future?steps=N` endpoint
- 取最近一帧的 `hidden_state` (64 维) + `fused_vector` (18 维 input)
- **persistence assumption**: 未来 N 步 input 持续 = 最后一帧
- LTC ODE 自由演化：`hidden_t+1 = backbone.rnn_cell(last_input, hidden_t, timespans)`
- 每步过 `readout(hidden)` 算 state_probs，过 `intent_head(hidden)` 算 intent_probs
- 能捕获 "疲劳累积 / 注意力衰减" 这种**内在动力学**，能预测状态切换时机

### Mean 外推 fallback
- LTC endpoint 失败时自动降级
- 取最近 10 帧 `intent_probs` mean / 最近 5 帧 state mode
- 不能预测突变 / 不利用 LNN ODE

### 当前限制（真 LTC 预测仍有）
- **persistence assumption** 不能预测外部刺激驱动的突变（比如有人敲门）
- 长期预测会衰减到 ODE attractor（不是真未来），**5-15s 内可信，>30s 不要看**

### 切换方法
- 关 LTC 预测: `export LNMEM_DISABLE_PREDICT_FUTURE=1` + 重启 daemon → 自动降 mean
- 完整 revert: `git revert b356847` (ADR-033 commit)
        """)


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
        # ADR-041 UX 修复: 用 st.fragment(run_every) 替代 streamlit_autorefresh,
        # 只重绘推断区域, 整页布局不再闪灰 (streamlit 1.33+ 原生)。
        st.caption("⚡ 推断区自动刷新 (1s, 局部不闪灰)")
    with ctrl3:
        if st.button("🔄 重置历史"):
            st.session_state.history.clear()
            st.session_state.last_frame_id = None
            st.rerun()

    # ---- toast 兜底显示 (老 streamlit 无 st.toast 时) ----
    if "_last_toast" in st.session_state:
        st.success(st.session_state.pop("_last_toast"))

    # ---- 推断 + 待反馈队列 (fragment 局部刷新) ----
    _render_inference_block()


@st.fragment(run_every=1.0)
def _render_inference_block() -> None:
    """ADR-041 UX 修复: fragment 局部刷新, 整页布局不闪灰。

    内含: 拉取最新推断 / 主区当前推断卡 / 右栏待反馈队列。
    fragment 内 st.rerun() 默认只重跑 fragment, 不重跑整页 (1.36+)。
    """
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
- **三类概率分布**：positive（正反馈）/ negative（负反馈）/ none（中性）的预测概率，三个数加起来 = 1.0
- **intent_pred**：模型最终猜的意图（取概率最大的那个）

> 💡 三类含义：**正反馈 = 你认可推断方向**，**负反馈 = 你不认可**，**中性 = 不明显倾向/中间态/无法判断**
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
                    def _commit(label: str) -> None:
                        """提交后给视觉反馈 (st.toast 1.27+, 兜底 session_state)。"""
                        try:
                            st.toast(f"✅ 已提交 → {label}", icon="✅")
                        except Exception:
                            st.session_state["_last_toast"] = f"✅ 已提交 → {label}"
                        del st.session_state.pending_feedback[ts_ns]
                        st.rerun()

                    bcols = st.columns(4)
                    with bcols[0]:
                        if st.button("✓ 确认", key=f"conf_{ts_ns}", type="primary"):
                            if submit_feedback(inf, inf.intent_pred):
                                _commit(f"确认 ({inf.intent_pred})")
                    with bcols[1]:
                        if st.button("→正", key=f"pos_{ts_ns}", disabled=inf.intent_pred == "positive"):
                            if submit_feedback(inf, "positive"):
                                _commit("positive")
                    with bcols[2]:
                        if st.button("→负", key=f"neg_{ts_ns}", disabled=inf.intent_pred == "negative"):
                            if submit_feedback(inf, "negative"):
                                _commit("negative")
                    with bcols[3]:
                        if st.button("→中性", key=f"none_{ts_ns}", disabled=inf.intent_pred == "none"):
                            if submit_feedback(inf, "none"):
                                _commit("none")
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

    # ---- 状态演化时间线 + 外推预测 ----
    st.markdown("---")
    st.subheader("🔮 状态演化 + 预测")
    render_evolution_panel(list(st.session_state.history))

    # 不再用 time.sleep+st.rerun，刷新由 streamlit-autorefresh 通过 JavaScript 触发（顶部 toggle）


if __name__ == "__main__":
    main()
