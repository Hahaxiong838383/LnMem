#!/usr/bin/env bash
# notify_watcher.sh — 后台 poll daemon /signals/recent，触发 macOS 系统通知 + 弹窗反馈
#
# 解决：川哥不能一直开 Streamlit web UI，但又要给反馈攒训练数据
# 方案：后台守护进程，每 N 秒 poll daemon，触发条件：
#   - margin/confidence > NOTIFY_THRESHOLD（默认 0.85）
#   - 距上次通知 ≥ NOTIFY_COOLDOWN_S 秒（默认 180，3 分钟，防骚扰）
#   - state 与上次通知不同（防同状态连发）
# 弹窗后：川哥点 [正/负/无/跳过]，自动 POST 到 daemon /intent_feedback
#
# 用法：
#   ./scripts/notify_watcher.sh                  # 前台跑（看 log）
#   nohup ./scripts/notify_watcher.sh > /tmp/notify_watcher.log 2>&1 &
#                                                # 后台跑
#
# 停止：pkill -f notify_watcher.sh

set -uo pipefail   # 不用 -e，poll 单次失败不退出整个 watcher

DAEMON_HOST="${DAEMON_HOST:-127.0.0.1}"
SIGNALS_URL="http://${DAEMON_HOST}:9876/signals/recent?limit=1"
FEEDBACK_URL="http://${DAEMON_HOST}:9877/intent_feedback"
POLL_INTERVAL="${POLL_INTERVAL:-3}"           # 每 3s 看一次
NOTIFY_THRESHOLD="${NOTIFY_THRESHOLD:-0.85}"  # margin 高于此触发
NOTIFY_COOLDOWN_S="${NOTIFY_COOLDOWN_S:-180}" # 通知冷却 3min
UI_SESSION="notify-watcher-$(date +%s)"

LAST_NOTIFY_TS=0
LAST_NOTIFY_STATE=""

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

trigger_dialog() {
    local state="$1" margin="$2" ts_ns="$3"

    # AppleScript 弹窗：osascript display dialog 最多 3 按钮（5/25 川哥实测踩坑）
    # 设计：[→ 负] [→ 中] [→ 正]，「跳过」靠 ESC / 60s 超时
    local prompt="LnMem 检测到状态：${state} (置信度 ${margin})

这个推断准吗？给反馈让 AI 学得更快
（不想反馈点叉关闭或等 60s 超时）"

    local result
    result=$(osascript -e "display dialog \"${prompt}\" \
        with title \"LnMem 反馈\" \
        buttons {\"→ 负\", \"→ 中\", \"→ 正\"} \
        default button \"→ 正\" \
        with icon note \
        giving up after 60" 2>&1)
    local osa_exit=$?

    log "osascript result: $result (exit=$osa_exit)"

    # gave up=true 或 user canceled → 跳过
    if echo "$result" | grep -q "gave up:true"; then
        log "user did not respond in 60s, skip"
        return 0
    fi
    if [ $osa_exit -ne 0 ] || echo "$result" | grep -q "User canceled"; then
        log "user canceled / osascript error"
        return 0
    fi

    local corrected=""
    case "$result" in
        *正*) corrected="positive" ;;
        *中*) corrected="none" ;;
        *负*) corrected="negative" ;;
        *) log "unknown button choice"; return 0 ;;
    esac

    # POST 反馈
    local resp
    resp=$(curl -s -X POST "$FEEDBACK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"ts_ns\":${ts_ns},\"predicted\":\"${state}\",\"corrected\":\"${corrected}\",\"ui_session\":\"${UI_SESSION}\",\"note\":\"notify_watcher\"}" 2>&1)
    log "feedback POST: $corrected → $resp"

    # 成功通知
    osascript -e "display notification \"已记录: ${corrected}\" with title \"LnMem 反馈\" sound name \"Tink\"" 2>/dev/null
}

log "===== notify_watcher 启动 ====="
log "POLL_INTERVAL=${POLL_INTERVAL}s NOTIFY_THRESHOLD=${NOTIFY_THRESHOLD} COOLDOWN=${NOTIFY_COOLDOWN_S}s"
log "等待 daemon 推断..."

while true; do
    # 拉最新 1 条
    JSON=$(curl -s -m 2 "$SIGNALS_URL" 2>/dev/null)
    if [ -z "$JSON" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # python 解析（macOS bash 不擅长 json）
    PARSED=$(python3 -c "
import json, sys
try:
    data = json.loads('''$JSON''')
    if not data: sys.exit(0)
    rec = data[0]
    lnn = rec.get('lnn') or {}
    state = lnn.get('predicted_state', 'unknown')
    margin = lnn.get('margin') or 0.0
    confidence = lnn.get('intent_confidence') or margin   # fallback margin
    ts = rec.get('ts', '')
    from datetime import datetime
    try:
        ts_ns = int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp() * 1e9)
    except Exception:
        ts_ns = 0
    print(f'{state}|{confidence:.3f}|{ts_ns}')
except Exception:
    pass
" 2>/dev/null)

    if [ -z "$PARSED" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    STATE=$(echo "$PARSED" | cut -d'|' -f1)
    CONF=$(echo "$PARSED" | cut -d'|' -f2)
    TS_NS=$(echo "$PARSED" | cut -d'|' -f3)
    NOW=$(date +%s)
    ELAPSED=$((NOW - LAST_NOTIFY_TS))

    # 触发条件：置信度高 + 冷却已过 + state 跟上次不同
    if awk "BEGIN {exit !($CONF >= $NOTIFY_THRESHOLD)}" \
       && [ "$ELAPSED" -ge "$NOTIFY_COOLDOWN_S" ] \
       && [ "$STATE" != "$LAST_NOTIFY_STATE" ]; then
        log "trigger: state=$STATE conf=$CONF (elapsed=${ELAPSED}s)"
        trigger_dialog "$STATE" "$CONF" "$TS_NS" &
        LAST_NOTIFY_TS=$NOW
        LAST_NOTIFY_STATE="$STATE"
    fi

    sleep "$POLL_INTERVAL"
done
