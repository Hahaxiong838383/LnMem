#!/usr/bin/env bash
# LnMem daemon watchdog (M0 W1 P0-4)
#
# 监控 daemon V0.2/V0.3 PID + RSS + last_frame_ts，异常时告警/重启。
#
# 用法：
#   前台运行： ./daemon_watchdog.sh
#   launchd ：  配 com.lnmem.daemon-watchdog.plist 后台
#
# 行为：每 INTERVAL 秒检查一次：
#   1. PID 存活（pgrep）
#   2. RSS：> RSS_WARN_GB → 日志 / > RSS_KILL_GB → SIGTERM + 重启 hook
#   3. /health.last_frame_ts：> STALE_SECONDS 无新帧 → daemon 假死，重启
#   4. signals jsonl > LOG_ROTATE_MB → 调用 log_rotation.sh
#
# 注：watchdog 不直接启 daemon（TCC 授权链限制 macOS），仅写 RESTART_NEEDED 标记
# 文件，川哥常用终端起的 supervisor 脚本检测到则重启。
#
# 退出码：
#   0 = 正常监控（守护 mode）
#   1 = 配置错误（DAEMON_NAME / HEALTH_URL 未设）
#   2 = log dir 不可写

set -uo pipefail

# ---------- 配置（环境变量覆盖）----------
DAEMON_NAME="${DAEMON_NAME:-poc.src.realtime.daemon_main}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:9876/health}"
INTERVAL="${INTERVAL:-60}"           # 秒
RSS_WARN_GB="${RSS_WARN_GB:-2.0}"    # > 此值发 warning 日志
RSS_KILL_GB="${RSS_KILL_GB:-3.0}"    # > 此值标记 RESTART_NEEDED
STALE_SECONDS="${STALE_SECONDS:-300}" # last_frame_ts 老于此值视为假死
LOG_ROTATE_MB="${LOG_ROTATE_MB:-500}" # signals jsonl > 此 MB 触发 rotate

WATCHDOG_LOG="${WATCHDOG_LOG:-$HOME/.lnmem/watchdog.log}"
RESTART_FLAG="${RESTART_FLAG:-$HOME/.lnmem/RESTART_NEEDED}"
SIGNALS_DIR="${SIGNALS_DIR:-$HOME/.lnmem/data}"

# ---------- 工具函数 ----------

log() {
    local level="$1"; shift
    local ts
    ts=$(date '+%Y-%m-%dT%H:%M:%S%z')
    printf '%s [%s] %s\n' "$ts" "$level" "$*" | tee -a "$WATCHDOG_LOG"
}

mark_restart_needed() {
    local reason="$1"
    log WARN "RESTART_NEEDED: $reason"
    printf '%s\n%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$reason" > "$RESTART_FLAG"
}

# 取 daemon PID（多个则取首个）
get_daemon_pid() {
    pgrep -f "$DAEMON_NAME" 2>/dev/null | head -1
}

# 取 PID 的 RSS（GB），失败返 -1
get_rss_gb() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
        echo "-1"
        return
    fi
    # ps RSS 是 KB
    local rss_kb
    rss_kb=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ')
    if [[ -z "$rss_kb" ]]; then
        echo "-1"
        return
    fi
    # 转 GB（保留 2 位小数）
    awk "BEGIN { printf \"%.2f\", $rss_kb / 1024 / 1024 }"
}

# 取 daemon /health.last_frame_ts（ISO8601 → epoch）
get_last_frame_epoch() {
    local resp
    resp=$(curl -s --max-time 3 "$HEALTH_URL" 2>/dev/null) || {
        echo "-1"
        return
    }
    local ts
    ts=$(printf '%s' "$resp" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(d.get('last_frame_ts', ''))
except Exception:
    print('')
" 2>/dev/null)
    if [[ -z "$ts" ]]; then
        echo "-1"
        return
    fi
    # ISO8601 → epoch（macOS date 不支持 -d，用 python3）
    python3 -c "
from datetime import datetime, timezone
ts = '$ts'
try:
    if ts.endswith('Z'):
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    else:
        dt = datetime.fromisoformat(ts)
    print(int(dt.timestamp()))
except Exception:
    print(-1)
"
}

# 检查 signals jsonl 是否需要 rotate
check_log_rotation() {
    local rotate_threshold_bytes=$((LOG_ROTATE_MB * 1024 * 1024))
    if [[ ! -d "$SIGNALS_DIR" ]]; then
        return
    fi
    while IFS= read -r path; do
        local size
        size=$(stat -f%z "$path" 2>/dev/null) || continue
        if (( size > rotate_threshold_bytes )); then
            log INFO "signals file > ${LOG_ROTATE_MB}MB triggers rotation: $path ($((size/1024/1024))MB)"
            local rotation_script="${0%/*}/log_rotation.sh"
            if [[ -x "$rotation_script" ]]; then
                "$rotation_script" "$path" || log ERROR "rotation failed: $path"
            else
                log WARN "log_rotation.sh not found at $rotation_script"
            fi
        fi
    done < <(find "$SIGNALS_DIR" -name 'signals_*.jsonl' -type f 2>/dev/null)
}

# ---------- 单次检查循环 ----------

check_once() {
    local pid rss_gb last_frame_epoch now stale
    pid=$(get_daemon_pid)
    if [[ -z "$pid" ]]; then
        mark_restart_needed "daemon process not found (PID lookup failed)"
        return
    fi

    rss_gb=$(get_rss_gb "$pid")
    if [[ "$rss_gb" == "-1" ]]; then
        log ERROR "cannot read RSS for PID $pid"
        return
    fi

    # RSS 阈值检查
    if awk "BEGIN { exit !($rss_gb > $RSS_KILL_GB) }"; then
        mark_restart_needed "RSS=${rss_gb}GB > ${RSS_KILL_GB}GB kill threshold (PID $pid)"
    elif awk "BEGIN { exit !($rss_gb > $RSS_WARN_GB) }"; then
        log WARN "RSS=${rss_gb}GB > ${RSS_WARN_GB}GB warn threshold (PID $pid)"
    else
        log DEBUG "RSS=${rss_gb}GB OK (PID $pid)"
    fi

    # last_frame_ts 假死检查
    last_frame_epoch=$(get_last_frame_epoch)
    if [[ "$last_frame_epoch" != "-1" ]]; then
        now=$(date +%s)
        stale=$(( now - last_frame_epoch ))
        if (( stale > STALE_SECONDS )); then
            mark_restart_needed "last_frame_ts ${stale}s ago > ${STALE_SECONDS}s staleness threshold"
        else
            log DEBUG "last_frame_ts ${stale}s ago OK"
        fi
    else
        log WARN "cannot read /health.last_frame_ts (daemon may not be serving HTTP)"
    fi

    # log rotation
    check_log_rotation
}

# ---------- 主循环 ----------

main() {
    mkdir -p "$(dirname "$WATCHDOG_LOG")" "$(dirname "$RESTART_FLAG")" || {
        echo "ERROR: cannot create log/flag dirs" >&2
        exit 2
    }

    log INFO "watchdog start: daemon=$DAEMON_NAME health=$HEALTH_URL interval=${INTERVAL}s"
    log INFO "thresholds: RSS_WARN=${RSS_WARN_GB}GB RSS_KILL=${RSS_KILL_GB}GB STALE=${STALE_SECONDS}s ROTATE=${LOG_ROTATE_MB}MB"

    while true; do
        check_once
        sleep "$INTERVAL"
    done
}

# 单次运行模式（测试用）
if [[ "${1:-}" == "--once" ]]; then
    check_once
    exit 0
fi

main
