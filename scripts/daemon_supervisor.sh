#!/usr/bin/env bash
# LnMem daemon supervisor (M0 W1 P0-4 第二阶段，2026-05-11)
#
# 用途：检测 watchdog 写入的 ~/.lnmem/RESTART_NEEDED 标记，自动重启 daemon。
#
# 设计要点：
# 1. supervisor 由 launchd 管理（LaunchAgent + KeepAlive）
# 2. 主循环每 INTERVAL 秒检查 RESTART_FLAG 文件
# 3. flag 存在 → kill 当前 daemon + osascript 启 Terminal.app 跑 start-daemon.sh
# 4. cooldown 机制：上次重启后 RESTART_COOLDOWN_S 秒内不再触发，防 thrashing
# 5. 失败时降级：osascript 失败 → 写 SUPERVISOR_ERROR 标记 + 大声 log
#
# TCC 授权链：
# - supervisor 由 launchd 启动 → 有 GUI session 但默认无 Apple Events Automation 授权
# - 首次 osascript 调用会弹"允许 supervisor 控制 Terminal" 对话框（川哥点允许）
# - 一旦授权，后续重启无需交互
#
# 退出码：
# - 0 正常退出（信号）
# - 1 配置错误
# - 2 路径不可写

set -uo pipefail

# ---------- 配置 ----------

INTERVAL="${SUPERVISOR_INTERVAL:-30}"               # 检查 flag 间隔（秒）
RESTART_COOLDOWN_S="${RESTART_COOLDOWN_S:-300}"     # 重启冷却（防 thrash，5min）
DAEMON_WARMUP_S="${DAEMON_WARMUP_S:-30}"            # daemon 启动到 healthy 等待
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:9876/health}"

# 默认 START_DAEMON_SCRIPT 用 supervisor 自身位置推导（响应 codex F6 — 不硬编码 $HOME 路径）
SCRIPT_SRC="${BASH_SOURCE[0]:-$0}"
if command -v realpath >/dev/null 2>&1; then
    SCRIPT_REAL="$(realpath "$SCRIPT_SRC" 2>/dev/null || echo "$SCRIPT_SRC")"
else
    SCRIPT_REAL="$(cd "$(dirname "$SCRIPT_SRC")" && pwd)/$(basename "$SCRIPT_SRC")"
fi
SCRIPT_DIR="$(dirname "$SCRIPT_REAL")"
START_DAEMON_SCRIPT="${START_DAEMON_SCRIPT:-$SCRIPT_DIR/start-daemon.sh}"

# 响应 codex F2 — kill 失败 backoff/熔断（防长期日志风暴）
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-3}"   # 连续 N 次失败后进入 backoff
BACKOFF_INTERVAL="${BACKOFF_INTERVAL:-600}"                  # backoff 模式间隔（10min）
HIBERNATE_AFTER_FAILURES="${HIBERNATE_AFTER_FAILURES:-10}"  # 连续 N 次后进入 hibernation
HIBERNATE_INTERVAL="${HIBERNATE_INTERVAL:-1800}"             # hibernation 间隔（30min）

RESTART_FLAG="${RESTART_FLAG:-$HOME/.lnmem/RESTART_NEEDED}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$HOME/.lnmem/supervisor.log}"
SUPERVISOR_ERROR_FLAG="${SUPERVISOR_ERROR_FLAG:-$HOME/.lnmem/SUPERVISOR_ERROR}"
DAEMON_NAME="${DAEMON_NAME:-poc.src.realtime.daemon_main}"

# ---------- 工具函数 ----------

log() {
    local level="$1"; shift
    local ts
    ts=$(date '+%Y-%m-%dT%H:%M:%S%z')
    printf '%s [%s] %s\n' "$ts" "$level" "$*" | tee -a "$SUPERVISOR_LOG"
}

get_daemon_pid() {
    pgrep -f "$DAEMON_NAME" 2>/dev/null | head -1
}

# 杀 daemon（先 SIGTERM 5s 等待，再 SIGKILL）
kill_daemon() {
    local pid
    pid=$(get_daemon_pid)
    if [[ -z "$pid" ]]; then
        log INFO "no daemon to kill"
        return 0
    fi
    log INFO "killing daemon PID $pid (SIGTERM, then SIGKILL after 5s)"
    kill -TERM "$pid" 2>/dev/null || true
    local i=0
    while [[ $i -lt 10 ]] && pgrep -f "$DAEMON_NAME" >/dev/null 2>&1; do
        sleep 0.5
        i=$((i + 1))
    done
    if pgrep -f "$DAEMON_NAME" >/dev/null 2>&1; then
        log WARN "SIGTERM failed (daemon may be deadlocked), sending SIGKILL"
        pkill -9 -f "$DAEMON_NAME" 2>/dev/null || true
        sleep 1
    fi
    if pgrep -f "$DAEMON_NAME" >/dev/null 2>&1; then
        log ERROR "daemon still alive after SIGKILL — escalate"
        return 1
    fi
    log INFO "daemon killed cleanly"
    return 0
}

# 用 osascript 启 Terminal.app 跑 start-daemon.sh（拿 TCC 摄像头授权）
start_daemon() {
    if [[ ! -x "$START_DAEMON_SCRIPT" ]]; then
        log ERROR "start-daemon.sh 不存在或不可执行: $START_DAEMON_SCRIPT"
        return 1
    fi
    log INFO "launching new daemon via osascript + Terminal.app: $START_DAEMON_SCRIPT"

    # osascript 失败时（如 Automation 未授权）输出错误到 stderr，捕获并 log
    local osascript_err
    osascript_err=$(osascript <<EOF 2>&1 1>/dev/null
tell application "Terminal"
    activate
    do script "$START_DAEMON_SCRIPT"
end tell
EOF
)
    local osa_exit=$?
    if [[ $osa_exit -ne 0 ]]; then
        log ERROR "osascript failed (exit $osa_exit): $osascript_err"
        log ERROR "如首次运行，请在 System Settings → Privacy & Security → Automation 给"
        log ERROR "supervisor 或父进程添加 Terminal.app 控制权限"
        return 1
    fi
    log INFO "osascript launched Terminal.app"
    return 0
}

# 验证 daemon 起来（poll /health）
# NOTE: 响应 codex F5 — 此检查仅"进程可达"语义，不验证 frames_processed 增长。
# 真假死场景（HTTP 响应但 pipeline 卡死）由 watchdog 负责再次检测 → 重新写 flag。
# supervisor 职责边界：执行重启动作，不二次诊断 pipeline 健康。
verify_daemon_up() {
    local i=0
    local max_iter=$((DAEMON_WARMUP_S * 2))  # 0.5s 间隔
    while [[ $i -lt $max_iter ]]; do
        if curl -s --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
            log INFO "daemon /health responsive after $((i / 2))s"
            return 0
        fi
        sleep 0.5
        i=$((i + 1))
    done
    log ERROR "daemon /health unreachable after ${DAEMON_WARMUP_S}s"
    return 1
}

# ---------- 主重启流程 ----------

handle_restart_flag() {
    # 响应 codex F1（TOCTOU）：当前读取-处理-删除 flag 流程非原子，watchdog 可能在
    # 读取后到删除前再次写 flag，导致"第二次原因"被本次 rm 一并清掉。
    # 影响：仅丢审计痕迹（不会漏重启），因为本轮已经处理。如要更强：
    #   mv "$RESTART_FLAG" "$RESTART_FLAG.inflight.$$" 原子消费后再处理。
    # PoC 阶段：风险接受（watchdog 写频率 1/min，supervisor 处理 < 60s）。
    local reason
    reason=$(tail -1 "$RESTART_FLAG" 2>/dev/null || echo "<unknown>")
    log INFO "RESTART_NEEDED detected: $reason"

    # cooldown 检查（防 thrash）
    if [[ -f "$RESTART_FLAG.last_action_ts" ]]; then
        local last_action_epoch
        last_action_epoch=$(cat "$RESTART_FLAG.last_action_ts" 2>/dev/null || echo "0")
        local now_epoch=$(date +%s)
        local elapsed=$((now_epoch - last_action_epoch))
        if [[ $elapsed -lt $RESTART_COOLDOWN_S ]]; then
            log WARN "cooldown active (elapsed=${elapsed}s < ${RESTART_COOLDOWN_S}s), 暂不重启"
            return 0
        fi
    fi

    # 1. 杀老 daemon
    if ! kill_daemon; then
        log ERROR "kill_daemon 失败，写 SUPERVISOR_ERROR 标记，跳过本次重启"
        printf '%s\nkill_daemon failed\n' "$(date -u +%FT%TZ)" > "$SUPERVISOR_ERROR_FLAG"
        return 1
    fi

    # 2. 启新 daemon
    if ! start_daemon; then
        log ERROR "start_daemon 失败（osascript 失败），写 SUPERVISOR_ERROR 标记"
        printf '%s\nstart_daemon failed (osascript)\n' "$(date -u +%FT%TZ)" > "$SUPERVISOR_ERROR_FLAG"
        return 1
    fi

    # 3. 验证新 daemon 起来
    if ! verify_daemon_up; then
        log ERROR "新 daemon 未在 ${DAEMON_WARMUP_S}s 内启动成功"
        printf '%s\nverify_daemon_up failed\n' "$(date -u +%FT%TZ)" > "$SUPERVISOR_ERROR_FLAG"
        return 1
    fi

    # 4. 清 RESTART_NEEDED + SUPERVISOR_ERROR + 记录 last_action_ts
    rm -f "$RESTART_FLAG" "$SUPERVISOR_ERROR_FLAG"
    date +%s > "$RESTART_FLAG.last_action_ts"
    log INFO "✅ daemon 重启成功，清理 flag"
    return 0
}

# ---------- 主循环 ----------

main() {
    mkdir -p "$(dirname "$SUPERVISOR_LOG")" || {
        echo "ERROR: cannot create log dir" >&2
        exit 2
    }
    log INFO "supervisor start: flag=$RESTART_FLAG interval=${INTERVAL}s cooldown=${RESTART_COOLDOWN_S}s"
    log INFO "start_daemon=$START_DAEMON_SCRIPT"
    log INFO "backoff: max_fail=$MAX_CONSECUTIVE_FAILURES backoff_intv=${BACKOFF_INTERVAL}s hibernate_after=$HIBERNATE_AFTER_FAILURES"

    # 响应 codex F2 — consecutive failure counter + backoff/hibernate
    local consecutive_failures=0
    local current_interval=$INTERVAL

    while true; do
        if [[ -f "$RESTART_FLAG" ]]; then
            if handle_restart_flag; then
                consecutive_failures=0
                current_interval=$INTERVAL
            else
                consecutive_failures=$((consecutive_failures + 1))
                if [[ $consecutive_failures -ge $HIBERNATE_AFTER_FAILURES ]]; then
                    current_interval=$HIBERNATE_INTERVAL
                    log ERROR "consecutive_failures=$consecutive_failures ≥ $HIBERNATE_AFTER_FAILURES, 进入 hibernation (${HIBERNATE_INTERVAL}s 间隔)"
                elif [[ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]]; then
                    current_interval=$BACKOFF_INTERVAL
                    log WARN "consecutive_failures=$consecutive_failures ≥ $MAX_CONSECUTIVE_FAILURES, 进入 backoff (${BACKOFF_INTERVAL}s 间隔)"
                else
                    log WARN "重启流程失败 (consecutive=$consecutive_failures), 下次循环重试"
                fi
            fi
        fi
        sleep "$current_interval"
    done
}

# 单次运行模式（测试用 — 不进入循环）
if [[ "${1:-}" == "--once" ]]; then
    if [[ -f "$RESTART_FLAG" ]]; then
        handle_restart_flag
        exit $?
    else
        echo "no RESTART_NEEDED flag, nothing to do"
        exit 0
    fi
fi

main
