#!/usr/bin/env bash
# notify_watcher.sh v2 — Active Learning 主动反馈触发（川哥 5/25 12:25 钦定）
#
# v1 → v2 范式转变：不再用单一阈值，由 LNN 自己判断"需要反馈"才弹，并附诚实理由。
#
# 6 触发器（codex 对抗审后整改，按优先级；同时段只取最高优先级一个）：
#   F. 冷启动校准 (ckpt 刚换 5min 内 + 任意不确定信号)
#   A. 意图不确定 (intent_probs top-2 差 < 0.10) 持续 ≥ 3 帧
#   B. 表情不确定 (phase2 margin < 0.30 或 entropy > 0.60) 持续 ≥ 3 帧
#   C. 状态突变 (高置信 top_prob ≥ 0.80 的 state 切到不同 state) cooldown 90s
#   D. 稀疏类  (此 state 历史 feedback < 5 条) cooldown 600s 防长尾尖刺
#   E. 预测翻转 (同 state 历史 ≥ 10 条 且主类占比在 40%-60% 之间) cooldown 600s
#
# 防抖：
#   - 同一触发器+同一 state 最小间隔 120s
#   - 全局最小间隔 30s
#   - lnn.error 非 null / 字段缺失 → 跳过（信号质量异常，不弹）
#
# 文案诚实性：
#   - 弹窗必带「依据：触发器 X + 关键字段=值」，不写"我觉得"心理化表述
#   - 多触发器并发只展示最高优先级 1 个主因（codex 钦定避免编故事拼接）
#
# 用法：
#   ./scripts/notify_watcher.sh                        # 前台
#   通过 LaunchAgent com.lnmem.notify-watcher 自动起   # 推荐生产方式

set -uo pipefail

DAEMON_HOST="${DAEMON_HOST:-127.0.0.1}"
SIGNALS_URL="http://${DAEMON_HOST}:9876/signals/recent?limit=1"
HEALTH_URL="http://${DAEMON_HOST}:9876/health"
FEEDBACK_URL="http://${DAEMON_HOST}:9877/intent_feedback"
POLL_INTERVAL="${POLL_INTERVAL:-3}"
GLOBAL_COOLDOWN_S="${GLOBAL_COOLDOWN_S:-300}"        # 5 分钟（之前 30s 太频繁）
PER_REASON_COOLDOWN_S="${PER_REASON_COOLDOWN_S:-1800}"  # 30 分钟（之前 120s 同问重问）
RECENT_FEEDBACK_DEDUP_S="${RECENT_FEEDBACK_DEDUP_S:-1800}"  # 30 分钟内反馈过的 (state, intent_pred) 不再问
DAILY_MAX="${DAILY_MAX:-60}"                         # 日上限 60（之前 150）

# active learning 触发阈值（codex 对抗审后定）
INTENT_TOP2_DIFF_MAX="${INTENT_TOP2_DIFF_MAX:-0.10}"
PHASE2_MARGIN_MIN="${PHASE2_MARGIN_MIN:-0.30}"
PHASE2_ENTROPY_MAX="${PHASE2_ENTROPY_MAX:-0.60}"
HIGH_CONF_TOP_PROB="${HIGH_CONF_TOP_PROB:-0.80}"
PERSIST_FRAMES="${PERSIST_FRAMES:-3}"        # A/B 要持续 N 帧防瞬时噪声
COLD_START_WINDOW_S="${COLD_START_WINDOW_S:-300}"  # F 冷启动窗口
SPARSE_CLASS_MAX="${SPARSE_CLASS_MAX:-5}"
FLIP_MIN_SAMPLES="${FLIP_MIN_SAMPLES:-10}"

UI_SESSION="notify-watcher-v2-$(date +%s)"
FEEDBACK_JSONL="${HOME}/.lnmem/data/intent_feedback/$(date +%Y-%m-%d).jsonl"
QUERY_DECISIONS_JSONL="${HOME}/.lnmem/data/query_decisions/$(date +%Y-%m-%d).jsonl"  # P-AL.1
mkdir -p "$(dirname $QUERY_DECISIONS_JSONL)"

# P-AL.2 online cooldown adaptation 参数
ADAPT_WINDOW_N="${ADAPT_WINDOW_N:-20}"       # 滑窗 20 次弹窗
ADAPT_LOW_THRESH="${ADAPT_LOW_THRESH:-0.3}"  # avg_reward < 0.3 → cooldown × 1.5
ADAPT_HIGH_THRESH="${ADAPT_HIGH_THRESH:-0.7}" # avg_reward > 0.7 → cooldown × 0.8
ADAPT_MIN_COOLDOWN="${ADAPT_MIN_COOLDOWN:-180}"   # 下限 3 分钟
ADAPT_MAX_COOLDOWN="${ADAPT_MAX_COOLDOWN:-3600}"  # 上限 1 小时
ADAPT_EVERY_N="${ADAPT_EVERY_N:-5}"          # 每 5 次弹窗重算一次
ADAPTED_COOLDOWN_FILE="${HOME}/.lnmem/watcher_state/adapted_cooldown"

# 状态（持久化在 ~/.lnmem/watcher_state/）
STATE_DIR="${HOME}/.lnmem/watcher_state"
mkdir -p "$STATE_DIR"
LAST_GLOBAL_TS_FILE="${STATE_DIR}/last_global_ts"
LAST_REASON_TS_FILE="${STATE_DIR}/last_reason_ts"   # key=reason|state value=ts
LAST_CKPT_SHA_FILE="${STATE_DIR}/last_ckpt_sha8"
LAST_CKPT_CHANGE_TS_FILE="${STATE_DIR}/last_ckpt_change_ts"
PERSIST_COUNTER_FILE="${STATE_DIR}/persist_counter"  # JSON {A:n, B:n}
LAST_STATE_FILE="${STATE_DIR}/last_high_conf_state"
DAILY_COUNT_FILE="${STATE_DIR}/daily_count_$(date +%Y-%m-%d)"

# helpers
read_ts() { [ -f "$1" ] && cat "$1" 2>/dev/null || echo 0; }
write_ts() { echo "$1" > "$2"; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }
now_s() { date +%s; }

# trigger_dialog: 弹窗 + 提交反馈
trigger_dialog() {
    local reason="$1" reason_desc="$2" state="$3" evidence="$4" ts_ns="$5"

    # 诚实文案：触发依据 + 关键字段 + 反馈作用
    local prompt="🤔 LNN 主动求反馈

【为什么这次问你】${reason_desc}

【依据】${evidence}

【你的反馈如何帮我】${reason_help}

当前推断状态：${state}"

    local dialog_start=$(date +%s)
    local result
    result=$(osascript -e "display dialog \"${prompt}\" \
        with title \"LnMem 主动学习\" \
        buttons {\"→ 负反馈\", \"→ 中性\", \"→ 正反馈\"} \
        default button \"→ 中性\" \
        with icon note \
        giving up after 60" 2>&1)
    local osa_exit=$?
    local dialog_end=$(date +%s)
    local response_time_s=$((dialog_end - dialog_start))

    log "osascript [$reason] result: $result (exit=$osa_exit)"

    # P-AL.1: 落 query_decisions.jsonl（包括 gave_up 也记，作为负 reward）
    local user_response="" corrected="" reward=0
    if echo "$result" | grep -q "gave up:true"; then
        user_response="gave_up"
        reward=0
        log "user 60s 内未响应，跳过 reward=0"
    elif [ $osa_exit -ne 0 ] || echo "$result" | grep -q "User canceled"; then
        user_response="canceled"
        reward=0
        log "user 取消 / osascript 错误 reward=0"
    else
        case "$result" in
            *正*) corrected="positive" ;;
            *中*) corrected="none" ;;       # "→ 中性" 按钮 → 内部 none
            *负*) corrected="negative" ;;
            *) log "unknown button: $result"; corrected="?"; ;;
        esac
        user_response="$corrected"
    fi

    # predicted 填真 intent_pred（有 intent_head 后是 positive/negative/none，否则 none 占位）
    local predicted="${INTENT_PRED:-none}"

    # 二元 reward 计算（川哥钦定）：响应 + 纠错 LNN = 正反馈；gave_up = 负
    if [ -n "$corrected" ] && [ "$corrected" != "?" ]; then
        if [ "$corrected" != "$predicted" ]; then
            reward=1   # 川哥纠错了 LNN（真正有学习价值）
        else
            reward=1   # 响应了即正 reward（确认 LNN 也算正）
        fi
    fi

    # 写 query_decisions.jsonl（无论是否响应都记）
    python3 - <<PYEOF >/dev/null 2>&1 || true
import json
from pathlib import Path
entry = {
    "ts_ns": ${ts_ns},
    "ui_session": "${UI_SESSION}",
    "trigger_reason": "${reason}",
    "lnn_state": {
        "phase2_state": "${state}",
        "intent_pred": "${predicted}",
        "evidence_str": "${evidence}",
    },
    "user_response": "${user_response}",
    "corrected": "${corrected}",
    "response_time_s": ${response_time_s},
    "reward": ${reward},
}
Path("${QUERY_DECISIONS_JSONL}").parent.mkdir(parents=True, exist_ok=True)
with open("${QUERY_DECISIONS_JSONL}", "a") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PYEOF

    # 如果是 gave_up/canceled，不 POST 反馈，提前 return
    if [ -z "$corrected" ] || [ "$corrected" = "?" ]; then
        return 0
    fi

    local resp
    resp=$(curl -s -X POST "$FEEDBACK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"ts_ns\":${ts_ns},\"predicted\":\"${predicted}\",\"corrected\":\"${corrected}\",\"ui_session\":\"${UI_SESSION}\",\"note\":\"v2|reason=${reason}|phase2=${state}|${evidence}|reward=${reward}\"}" 2>&1)
    log "POST: ${predicted}→${corrected} ($reason) reward=$reward → $resp"

    # 成功通知
    osascript -e "display notification \"已记录: ${corrected} ($reason)\" with title \"LnMem 反馈\" sound name \"Tink\"" 2>/dev/null

    # 更新 cooldown
    local now=$(now_s)
    write_ts "$now" "$LAST_GLOBAL_TS_FILE"
    echo "${reason}|${state}=${now}" >> "$LAST_REASON_TS_FILE"
    # 日计数 +1
    local cnt=$(read_ts "$DAILY_COUNT_FILE")
    write_ts "$((cnt + 1))" "$DAILY_COUNT_FILE"
}

# 主决策（Python 写更稳）
decide_trigger() {
    local lnn_json="$1"
    python3 - <<PYEOF
import json, os, sys
from collections import Counter
from pathlib import Path

lnn = json.loads('''$lnn_json''')
if not lnn or lnn.get('error'):
    print("SKIP|signal_quality|lnn block error or empty")
    sys.exit(0)

state = lnn.get('predicted_state')
top_prob = lnn.get('top_prob') or 0.0
margin = lnn.get('margin') or 0.0
entropy = lnn.get('entropy') or 0.0
intent_probs = lnn.get('intent_probs') or []
intent_pred = lnn.get('intent_pred')
ckpt_sha = lnn.get('ckpt_sha8')

# 缺关键字段 → skip
if not state or not isinstance(intent_probs, list) or len(intent_probs) != 3:
    print(f"SKIP|signal_quality|fields_missing state={state} probs_len={len(intent_probs) if isinstance(intent_probs, list) else '?'}")
    sys.exit(0)

# 读历史
state_dir = Path(os.path.expanduser('~/.lnmem/watcher_state'))
last_ckpt_sha = (state_dir / 'last_ckpt_sha8').read_text().strip() if (state_dir / 'last_ckpt_sha8').exists() else ''
last_ckpt_change_ts = int((state_dir / 'last_ckpt_change_ts').read_text().strip()) if (state_dir / 'last_ckpt_change_ts').exists() else 0
last_high_conf_state = (state_dir / 'last_high_conf_state').read_text().strip() if (state_dir / 'last_high_conf_state').exists() else ''
persist_file = state_dir / 'persist_counter'
persist = json.loads(persist_file.read_text()) if persist_file.exists() else {'A': 0, 'B': 0}

import time
now = int(time.time())

# 持久化 ckpt 变更时间
# 修：watcher 第一次启动看到 last_ckpt_sha 空，不能当作"刚换"
# 必须区分 "初次见到" vs "真换了"
if ckpt_sha and ckpt_sha != last_ckpt_sha:
    (state_dir / 'last_ckpt_sha8').write_text(ckpt_sha)
    if last_ckpt_sha:
        # 真换了（之前记过 sha 但不同）
        (state_dir / 'last_ckpt_change_ts').write_text(str(now))
        last_ckpt_change_ts = now
    # 初次见到不更新 last_ckpt_change_ts，保持 0 → F 不触发

# 读历史 feedback 分布（今天 jsonl）
jsonl = Path(os.path.expanduser(f"~/.lnmem/data/intent_feedback/{__import__('datetime').date.today()}.jsonl"))
state_count = Counter()
state_corrected = {}  # state -> Counter of corrected
recent_combos = set()  # 最近 N 分钟内反馈过的 (state, predicted) 组合 → 不再问
dedup_window = int(os.environ.get('RECENT_FEEDBACK_DEDUP_S', '1800'))
cutoff_ts_ns = (now - dedup_window) * 1_000_000_000
if jsonl.exists():
    with jsonl.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                note = d.get('note', '')
                ps = None
                for token in note.split('|'):
                    if token.startswith('phase2=') or token.startswith('phase2_state='):
                        ps = token.split('=', 1)[1]
                        break
                if ps:
                    state_count[ps] += 1
                    state_corrected.setdefault(ps, Counter())[d.get('corrected', '?')] += 1
                # 提取这条反馈的 ts_ns + 组合
                ts_ns_val = d.get('ts_ns') or 0
                if ts_ns_val and ts_ns_val >= cutoff_ts_ns:
                    pred = d.get('predicted', 'none')
                    if ps:
                        recent_combos.add((ps, pred))
            except: pass

# 抑制：当前 (state, intent_pred) 已经在最近 N 分钟反馈过 → SKIP
current_combo = (state, intent_pred)
if current_combo in recent_combos:
    print(f"SKIP|recent_feedback|组合 {current_combo} 最近 {dedup_window//60}min 内已反馈过，不重问")
    sys.exit(0)

# 计算 6 触发器，按优先级返回第一个命中
intent_sorted = sorted(intent_probs, reverse=True)
top2_diff = intent_sorted[0] - intent_sorted[1]
is_intent_uncertain = top2_diff < float(os.environ.get('INTENT_TOP2_DIFF_MAX', '0.10'))
is_phase2_uncertain = margin < float(os.environ.get('PHASE2_MARGIN_MIN', '0.30')) or entropy > float(os.environ.get('PHASE2_ENTROPY_MAX', '0.60'))
is_high_conf = top_prob >= float(os.environ.get('HIGH_CONF_TOP_PROB', '0.80'))
persist_n = int(os.environ.get('PERSIST_FRAMES', '3'))
cold_window = int(os.environ.get('COLD_START_WINDOW_S', '300'))
sparse_max = int(os.environ.get('SPARSE_CLASS_MAX', '5'))
flip_min = int(os.environ.get('FLIP_MIN_SAMPLES', '10'))

# 更新持续帧计数
if is_intent_uncertain:
    persist['A'] = persist.get('A', 0) + 1
else:
    persist['A'] = 0
if is_phase2_uncertain:
    persist['B'] = persist.get('B', 0) + 1
else:
    persist['B'] = 0
persist_file.write_text(json.dumps(persist))

# 更新 last_high_conf_state
state_changed = is_high_conf and last_high_conf_state and state != last_high_conf_state
if is_high_conf:
    (state_dir / 'last_high_conf_state').write_text(state)

# 触发器优先级判断
# F. 冷启动校准（最高优先级，仅当 ckpt 5min 内换过 + 有任意不确定信号）
if last_ckpt_change_ts > 0 and (now - last_ckpt_change_ts) < cold_window:
    if is_intent_uncertain or is_phase2_uncertain:
        ago = (now - last_ckpt_change_ts) // 60
        print(f"F|冷启动校准期|我刚学完一批反馈（{ago} 分钟前 ckpt 换到 {ckpt_sha[:6]}），需要你验证学得对|intent_top2_diff={top2_diff:.3f} phase2_margin={margin:.3f}|帮我确认新模型的判断方向，避免学偏")
        sys.exit(0)

# A. 意图不确定（持续 N 帧）
if persist['A'] >= persist_n:
    print(f"A|意图不确定|我对你这一刻意图判断很犹豫（正反馈={intent_probs[0]:.2f} / 负反馈={intent_probs[1]:.2f} / 中性={intent_probs[2]:.2f}，top2 差仅 {top2_diff:.3f}）|连续 {persist['A']} 帧三类概率差 < {os.environ.get('INTENT_TOP2_DIFF_MAX', '0.10')}|你告诉我到底是正反馈/负反馈/中性，我下次同情境能学会")
    sys.exit(0)

# B. 表情不确定（持续 N 帧）
if persist['B'] >= persist_n:
    issue = "margin 太小" if margin < 0.30 else "entropy 太高"
    print(f"B|表情不确定|我对你当前表情 ({state}) 不确定（{issue}：margin={margin:.3f} entropy={entropy:.3f}）|连续 {persist['B']} 帧 phase2 head 模糊|你的反馈能帮 phase2 backbone 校准表情判断")
    sys.exit(0)

# C. 状态突变
if state_changed:
    print(f"C|高置信状态突变|刚才你是 {last_high_conf_state}，现在切到 {state}（两次都 top_prob ≥ 0.80），是真切换还是误判？|state_change top_prob={top_prob:.3f}|确认是否真切换，避免我把临时表情学成长期偏好")
    sys.exit(0)

# D. 稀疏类
n = state_count.get(state, 0)
if n < sparse_max:
    print(f"D|稀疏类样本|你当前是 {state} 状态，但历史反馈仅 {n} 条（< {sparse_max}），这类样本对我极宝贵|sparse_class_count={n}|多 1 条样本，我对这种状态的判断准确率能提升明显")
    sys.exit(0)

# E. 预测翻转（同 state 历史 ≥ 10 条 且主类占比 40-60）
if n >= flip_min:
    c = state_corrected.get(state, Counter())
    total = sum(c.values()) or 1
    top_share = max(c.values()) / total
    if 0.4 <= top_share <= 0.6:
        breakdown = ', '.join([f'{k}={v}' for k,v in c.most_common()])
        print(f"E|历史标注分歧|你之前对 {state} 状态的反馈分布很均匀（{breakdown}），我拿不准|flip_top_share={top_share:.2f}|这次你的选择能帮我决定这种状态默认偏向哪类")
        sys.exit(0)

print("NONE|all triggers idle")
PYEOF
}

# 启动 log
log "===== notify_watcher v2.2 (active learning + cooldown adaptation) 启动 ====="
log "POLL=${POLL_INTERVAL}s GLOBAL_COOLDOWN=${GLOBAL_COOLDOWN_S}s (initial, will adapt) PER_REASON=${PER_REASON_COOLDOWN_S}s DAILY_MAX=${DAILY_MAX}"
log "adaptation: window=${ADAPT_WINDOW_N} 每 ${ADAPT_EVERY_N} 次重算 range=[${ADAPT_MIN_COOLDOWN}s, ${ADAPT_MAX_COOLDOWN}s]"
log "阈值：intent_top2_diff<${INTENT_TOP2_DIFF_MAX} phase2_margin<${PHASE2_MARGIN_MIN} entropy>${PHASE2_ENTROPY_MAX}"
log "触发器: F(冷启动) > A(意图不确定) > B(表情不确定) > C(状态突变) > D(稀疏类) > E(翻转)"

# P-AL.2 cooldown adaptation 函数
adapt_cooldown() {
    local current_cooldown="$1"
    python3 - <<PYEOF
import json, os, sys
from pathlib import Path

jsonl = Path("${QUERY_DECISIONS_JSONL}")
if not jsonl.exists():
    print(${current_cooldown}); sys.exit(0)

window_n = int(os.environ.get('ADAPT_WINDOW_N', '20'))
low_thresh = float(os.environ.get('ADAPT_LOW_THRESH', '0.3'))
high_thresh = float(os.environ.get('ADAPT_HIGH_THRESH', '0.7'))
cd_min = int(os.environ.get('ADAPT_MIN_COOLDOWN', '180'))
cd_max = int(os.environ.get('ADAPT_MAX_COOLDOWN', '3600'))

# 读最近 window_n 条
rewards = []
with jsonl.open() as f:
    for line in f:
        try:
            d = json.loads(line)
            r = d.get('reward')
            if r is not None:
                rewards.append(int(r))
        except: pass
rewards = rewards[-window_n:]
n = len(rewards)
if n < 5:
    print(${current_cooldown}); sys.exit(0)

avg = sum(rewards) / n
cd = ${current_cooldown}
old_cd = cd

if avg < low_thresh:
    cd = int(cd * 1.5)   # 川哥嫌烦，拉长
elif avg > high_thresh:
    cd = int(cd * 0.8)   # 川哥积极参与，缩短
# 中间区不变

cd = max(cd_min, min(cd_max, cd))
print(cd, file=sys.stderr)
print(f"[adapt] n={n} avg_reward={avg:.2f} {old_cd}s → {cd}s", file=sys.stderr)
print(cd)
PYEOF
}

# 初始 cooldown
EFFECTIVE_COOLDOWN="$GLOBAL_COOLDOWN_S"
if [ -f "$ADAPTED_COOLDOWN_FILE" ]; then
    stored=$(cat "$ADAPTED_COOLDOWN_FILE" 2>/dev/null)
    if [ -n "$stored" ] && [ "$stored" -gt 0 ] 2>/dev/null; then
        EFFECTIVE_COOLDOWN="$stored"
        log "复用上次 adapted cooldown: ${EFFECTIVE_COOLDOWN}s"
    fi
fi
TRIGGER_COUNT=0

while true; do
    # 日上限检查
    cnt=$(read_ts "$DAILY_COUNT_FILE")
    if [ "$cnt" -ge "$DAILY_MAX" ]; then
        log "DAILY_MAX=$DAILY_MAX 已达上限，今天不再弹窗"
        sleep 300
        continue
    fi

    # 拉数据
    JSON=$(curl -s -m 2 "$SIGNALS_URL" 2>/dev/null)
    if [ -z "$JSON" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # 提取 lnn 块 + ts_ns + intent_pred
    META=$(echo "$JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if not data: sys.exit(0)
    rec = data[0]
    lnn = rec.get('lnn') or {}
    ts = rec.get('ts', '')
    from datetime import datetime
    try:
        ts_ns = int(datetime.fromisoformat(ts.replace('Z','+00:00')).timestamp() * 1e9)
    except: ts_ns = 0
    print(f'{ts_ns}|{lnn.get(\"intent_pred\") or \"none\"}|{json.dumps(lnn)}')
except: pass
" 2>/dev/null)

    if [ -z "$META" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    TS_NS=$(echo "$META" | cut -d'|' -f1)
    INTENT_PRED=$(echo "$META" | cut -d'|' -f2)
    export INTENT_PRED
    LNN_JSON=$(echo "$META" | cut -d'|' -f3-)

    DECISION=$(decide_trigger "$LNN_JSON")
    REASON=$(echo "$DECISION" | cut -d'|' -f1)

    if [ "$REASON" = "NONE" ] || [ "$REASON" = "SKIP" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    REASON_DESC=$(echo "$DECISION" | cut -d'|' -f2)
    REASON_HELP_RAW=$(echo "$DECISION" | cut -d'|' -f3)
    EVIDENCE=$(echo "$DECISION" | cut -d'|' -f4)
    HELP_TEXT=$(echo "$DECISION" | cut -d'|' -f5)

    # state 用 lnn predicted_state
    STATE=$(echo "$LNN_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin).get('predicted_state','?'))")

    # 全局 cooldown（用 adapted cooldown 不用静态 GLOBAL_COOLDOWN_S）
    NOW=$(now_s)
    LAST_GLOBAL=$(read_ts "$LAST_GLOBAL_TS_FILE")
    ELAPSED=$((NOW - LAST_GLOBAL))
    if [ "$ELAPSED" -lt "$EFFECTIVE_COOLDOWN" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # 同一原因+state cooldown（grep last_reason_ts_file）
    LAST_RS=$(grep "^${REASON}|${STATE}=" "$LAST_REASON_TS_FILE" 2>/dev/null | tail -1 | cut -d'=' -f2)
    LAST_RS=${LAST_RS:-0}
    RS_ELAPSED=$((NOW - LAST_RS))
    if [ "$RS_ELAPSED" -lt "$PER_REASON_COOLDOWN_S" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    log "TRIGGER [${REASON}] state=${STATE} elapsed_global=${ELAPSED}s elapsed_reason=${RS_ELAPSED}s effective_cooldown=${EFFECTIVE_COOLDOWN}s"
    log "  evidence: $EVIDENCE"

    reason_help="$HELP_TEXT"
    # 同步等待 trigger_dialog（不再 &），让 query_decisions 落盘后再算 adaptation
    trigger_dialog "$REASON" "$REASON_DESC" "$STATE" "$EVIDENCE" "$TS_NS"
    TRIGGER_COUNT=$((TRIGGER_COUNT + 1))

    # P-AL.2 每 ADAPT_EVERY_N 次触发后重算 cooldown
    if [ $((TRIGGER_COUNT % ADAPT_EVERY_N)) -eq 0 ]; then
        NEW_CD=$(adapt_cooldown "$EFFECTIVE_COOLDOWN" 2>/tmp/adapt_$$.log)
        if [ -n "$NEW_CD" ] && [ "$NEW_CD" != "$EFFECTIVE_COOLDOWN" ]; then
            cat /tmp/adapt_$$.log >> ~/.lnmem/logs/notify_watcher/launchd.out.log
            EFFECTIVE_COOLDOWN="$NEW_CD"
            echo "$EFFECTIVE_COOLDOWN" > "$ADAPTED_COOLDOWN_FILE"
            log "🎯 cooldown adapted → ${EFFECTIVE_COOLDOWN}s（基于最近 ${ADAPT_WINDOW_N} 次反馈 reward）"
        fi
        rm -f /tmp/adapt_$$.log
    fi

    sleep "$POLL_INTERVAL"
done
