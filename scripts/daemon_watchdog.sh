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
USER_IDLE_THRESHOLD="${USER_IDLE_THRESHOLD:-300}" # W1-P2-1: HIDIdleTime > 此值视为用户离开/显示器休眠，跳过 STALE
LOG_ROTATE_MB="${LOG_ROTATE_MB:-500}" # signals jsonl > 此 MB 触发 rotate
# W2-P0-bugfix（2026-05-16）：HTTP 假死独立信号
# 根因事故：5/15 20:43 daemon HTTP server 卡死 18h，watchdog 走"cannot read"WARN 分支不重启。
# 修复：连续 N 次 curl 失败视为"HTTP 假死"，与 PID/RSS/STALE 并列触发 RESTART_NEEDED。
CONSECUTIVE_HEALTH_FAIL_THRESHOLD="${CONSECUTIVE_HEALTH_FAIL_THRESHOLD:-3}"
HEALTH_CONNECT_TIMEOUT="${HEALTH_CONNECT_TIMEOUT:-5}"  # curl --connect-timeout
HEALTH_MAX_TIME="${HEALTH_MAX_TIME:-15}"               # curl --max-time（含 read）
# F-04 fix（codex 二审）：idle 豁免上限，防真故障期间用户长时 idle → 永不自愈
# 累计 idle-skip 超过此值（默认 1h）→ 仍计入失败计数，让阈值能触发重启
MAX_IDLE_SKIP_SECONDS="${MAX_IDLE_SKIP_SECONDS:-3600}"

WATCHDOG_LOG="${WATCHDOG_LOG:-$HOME/.lnmem/watchdog.log}"
RESTART_FLAG="${RESTART_FLAG:-$HOME/.lnmem/RESTART_NEEDED}"
SIGNALS_DIR="${SIGNALS_DIR:-$HOME/.lnmem/data}"
# 响应 codex F4（supervisor 评审）— watchdog 把 silent SUPERVISOR_ERROR 外显到 log
SUPERVISOR_ERROR_FLAG="${SUPERVISOR_ERROR_FLAG:-$HOME/.lnmem/SUPERVISOR_ERROR}"
# HTTP 失败计数持久化（跨 watchdog 重启保留计数）
# F-01 fix（codex 二审）：文件格式改为 "<pid>:<count>" 绑定 daemon 代际，
# PID 变化时视为新代际自动清零，防止旧 daemon 残留计数污染新 daemon 冷启动
HEALTH_FAIL_COUNT_FILE="${HEALTH_FAIL_COUNT_FILE:-$HOME/.lnmem/watchdog_health_fail_count}"
# F-04 fix：idle skip 累计计时持久化
IDLE_SKIP_ACCUM_FILE="${IDLE_SKIP_ACCUM_FILE:-$HOME/.lnmem/watchdog_idle_skip_accum}"
# 重启前 SIGUSR1 dump traceback 落盘位置（cc / 老板复盘用）
DAEMON_TRACEBACK_DUMP="${DAEMON_TRACEBACK_DUMP:-$HOME/.lnmem/daemon_traceback.log}"

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

# 取系统/用户输入空闲秒数（HIDIdleTime, 跨 Intel/Apple Silicon）
# 用途：W1-P2-1 watchdog 误报修复 — 川哥扣屏幕 → display sleep → 摄像头停帧
# 不算 daemon hung；HIDIdleTime > USER_IDLE_THRESHOLD 时跳过 STALE 检测。
# 失败返 -1（视为未知，按现行 STALE 逻辑兜底）。
get_user_idle_sec() {
    ioreg -c IOHIDSystem 2>/dev/null \
        | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}' \
        | grep -E '^[0-9]+$' || echo "-1"
}

# HTTP 响应 + last_frame_ts 解析。
# 返回值约定（独立信号化，2026-05-16 W2-P0-bugfix）：
#   echo epoch + return 0 → HTTP OK 且字段有效
#   echo "HTTP_FAIL:<curl_exit>" + return 1 → HTTP 不响应（connect/read timeout / refused 等）
#   echo "HTTP_OK_NO_TS"      + return 2 → HTTP 响应但缺 last_frame_ts 字段
# 调用方根据这三类分别走"连续失败计数"vs"STALE 比对"vs"字段缺失警告"。
get_last_frame_epoch() {
    local resp curl_exit
    # --connect-timeout 5 区分"连不上"vs"连上但 hang"（read timeout）
    # --max-time 15 总上限，含 read。事故根因是 read 阶段 hang，必须给足时间观测。
    resp=$(curl -sS --connect-timeout "$HEALTH_CONNECT_TIMEOUT" --max-time "$HEALTH_MAX_TIME" "$HEALTH_URL" 2>/dev/null)
    curl_exit=$?
    if (( curl_exit != 0 )); then
        echo "HTTP_FAIL:${curl_exit}"
        return 1
    fi
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
        echo "HTTP_OK_NO_TS"
        return 2
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
    return 0
}

# HTTP 失败计数器（持久化跨 watchdog 重启）
# F-01 fix（codex 二审）：counter 与 daemon "代际" 绑定（PID）
# 文件格式："<pid>:<count>"  例如 "67207:2"
# 读取时若当前活 daemon PID 与文件 PID 不一致 → 视为新代际，返 0（不污染）
read_health_fail_count() {
    local current_pid="$1"
    if [[ ! -f "$HEALTH_FAIL_COUNT_FILE" ]]; then
        echo 0
        return
    fi
    local raw
    raw=$(cat "$HEALTH_FAIL_COUNT_FILE" 2>/dev/null | tr -d ' \n')
    # F-06 fix（codex 二审）：旧格式（纯数字）首轮跨代污染窗口
    # 修：检测到旧格式 → 直接视为 "未绑代际"，强制返 0 + 让后续 write 改写为新格式
    # 代价：升级首轮丢失最多 N-1 个累计失败状态，但避免污染
    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        echo 0
        return
    fi
    if [[ ! "$raw" =~ ^[0-9]+:[0-9]+$ ]]; then
        # 损坏 → 0
        echo 0
        return
    fi
    local stored_pid="${raw%:*}"
    local stored_count="${raw#*:}"
    if [[ "$stored_pid" != "$current_pid" ]]; then
        # 代际切换 — 旧计数无效
        echo 0
        return
    fi
    echo "$stored_count"
}

write_health_fail_count() {
    local pid="$1"
    local count="$2"
    printf '%s:%s\n' "$pid" "$count" > "$HEALTH_FAIL_COUNT_FILE" 2>/dev/null || true
}

reset_health_fail_count() {
    local pid="$1"
    write_health_fail_count "$pid" 0
}

# F-04 fix（codex 二审）：idle skip 累计读写，超过 MAX_IDLE_SKIP_SECONDS 时强制计入失败
read_idle_skip_accum() {
    if [[ ! -f "$IDLE_SKIP_ACCUM_FILE" ]]; then
        echo 0
        return
    fi
    local v
    v=$(cat "$IDLE_SKIP_ACCUM_FILE" 2>/dev/null | tr -d ' \n')
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

write_idle_skip_accum() {
    printf '%s\n' "$1" > "$IDLE_SKIP_ACCUM_FILE" 2>/dev/null || true
}

# F-02 + F-03 fix（codex 二审）：发 SIGUSR1 前二次校验 PID 身份 + dump 成功性可观测
# 1) ps -p pid -o command 包含 DAEMON_NAME → 确认是目标 daemon（防 PID 复用打错进程）
# 2) 记录 dump 文件 pre_mtime，dump+wait 后检查 mtime 是否更新 → 失败明确 WARN
dump_daemon_traceback_before_restart() {
    local pid="$1"
    [[ -z "$pid" ]] && return

    # F-02: 二次校验 PID 仍是目标 daemon（不是 PID 复用后的其他进程）
    local current_cmd
    current_cmd=$(ps -p "$pid" -o command= 2>/dev/null || echo "")
    if [[ -z "$current_cmd" ]]; then
        log WARN "SIGUSR1 skip: PID $pid no longer alive (daemon may have already died)"
        return
    fi
    if ! printf '%s' "$current_cmd" | grep -qF "$DAEMON_NAME"; then
        log WARN "SIGUSR1 skip: PID $pid cmdline does not match DAEMON_NAME ($DAEMON_NAME) — possible PID reuse, refusing to send"
        return
    fi

    # F-03 + F-07 fix（codex 二审）：dump 观测用 (mtime, size) 双条件
    # 单 mtime 秒级精度有同秒假阴性；size 增长是更强信号（dump 必然写入新内容）
    local pre_mtime=0 pre_size=0
    if [[ -f "$DAEMON_TRACEBACK_DUMP" ]]; then
        pre_mtime=$(stat -f%m "$DAEMON_TRACEBACK_DUMP" 2>/dev/null || echo 0)
        pre_size=$(stat -f%z "$DAEMON_TRACEBACK_DUMP" 2>/dev/null || echo 0)
    fi

    if ! kill -USR1 "$pid" 2>/dev/null; then
        log WARN "SIGUSR1 send to PID $pid failed (kill returned non-zero)"
        return
    fi
    log INFO "sent SIGUSR1 to PID $pid (cmd verified) → expect dump at $DAEMON_TRACEBACK_DUMP"
    sleep 1

    # F-03 + F-07: 校验 dump 是否真写入（mtime OR size 任一变化 → 视为 OBSERVED）
    local post_mtime=0 post_size=0
    if [[ -f "$DAEMON_TRACEBACK_DUMP" ]]; then
        post_mtime=$(stat -f%m "$DAEMON_TRACEBACK_DUMP" 2>/dev/null || echo 0)
        post_size=$(stat -f%z "$DAEMON_TRACEBACK_DUMP" 2>/dev/null || echo 0)
    fi
    if (( post_mtime > pre_mtime )) || (( post_size > pre_size )); then
        log INFO "SIGUSR1 dump OBSERVED: mtime=$post_mtime size=${post_size}B (Δsize=$((post_size - pre_size))B)"
    else
        log WARN "SIGUSR1 dump NOT OBSERVED within 1s (mtime+size both unchanged — daemon may not handle SIGUSR1 or is fully blocked)"
    fi
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
    # 响应 codex F4（supervisor 评审）— 把 SUPERVISOR_ERROR silent 状态外显
    if [[ -f "$SUPERVISOR_ERROR_FLAG" ]]; then
        local sup_err_reason
        sup_err_reason=$(tail -1 "$SUPERVISOR_ERROR_FLAG" 2>/dev/null || echo "<unknown>")
        log WARN "SUPERVISOR_ERROR 标记存在: $sup_err_reason — supervisor 无法自动重启 daemon，需人工介入"
    fi
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

    # last_frame_ts 假死检查（W1-P2-1: 加 user idle 过滤防误报）
    # W2-P0-bugfix（2026-05-16）：HTTP 不响应改成独立 first-class 信号（连续 N 次 → RESTART）
    # codex 二审 fix（F-01/F-02/F-03/F-04）：代际绑定 + idle 上限 + dump 可观测
    last_frame_epoch=$(get_last_frame_epoch)
    local probe_rc=$?
    if (( probe_rc == 0 )); then
        # HTTP OK → 重置失败计数（按当前 PID）+ 清 idle skip 累计 + 走 STALE 检测
        reset_health_fail_count "$pid"
        write_idle_skip_accum 0
        now=$(date +%s)
        stale=$(( now - last_frame_epoch ))
        if (( stale > STALE_SECONDS )); then
            # 5/11 早上误报根因：川哥扣 MacBook 屏幕 → display sleep → 摄像头硬件关
            # → daemon 拿不到帧 → last_frame_ts 停更 ≠ daemon hung
            # 解：若用户输入空闲 > USER_IDLE_THRESHOLD，视为预期 idle，跳过 RESTART_NEEDED
            idle=$(get_user_idle_sec)
            if [[ "$idle" != "-1" ]] && (( idle > USER_IDLE_THRESHOLD )); then
                log INFO "last_frame_ts ${stale}s ago but user idle ${idle}s > ${USER_IDLE_THRESHOLD}s — likely display sleep, skip STALE"
            else
                mark_restart_needed "last_frame_ts ${stale}s ago > ${STALE_SECONDS}s staleness threshold (user idle ${idle}s)"
            fi
        else
            log DEBUG "last_frame_ts ${stale}s ago OK"
        fi
    elif (( probe_rc == 1 )); then
        # HTTP 完全不响应（curl 失败）— 连续 N 次触发 RESTART
        # F-04: idle 跳过有上限，累计 idle-skip > MAX_IDLE_SKIP_SECONDS → 仍计入失败
        idle=$(get_user_idle_sec)
        local idle_skip_accum
        idle_skip_accum=$(read_idle_skip_accum)
        local should_skip_for_idle=0
        if [[ "$idle" != "-1" ]] && (( idle > USER_IDLE_THRESHOLD )); then
            # 在 idle 豁免窗口内
            idle_skip_accum=$(( idle_skip_accum + INTERVAL ))
            write_idle_skip_accum "$idle_skip_accum"
            if (( idle_skip_accum <= MAX_IDLE_SKIP_SECONDS )); then
                should_skip_for_idle=1
                log INFO "HTTP unresponsive ($last_frame_epoch) but user idle ${idle}s — skip fail count (accum=${idle_skip_accum}s/${MAX_IDLE_SKIP_SECONDS}s)"
            else
                log WARN "HTTP unresponsive (accum idle-skip ${idle_skip_accum}s > ${MAX_IDLE_SKIP_SECONDS}s upper bound) — falling back to count this fail"
            fi
        else
            # 非 idle 期 → 计数清零累计
            write_idle_skip_accum 0
        fi

        if (( should_skip_for_idle == 0 )); then
            local fail_count
            fail_count=$(read_health_fail_count "$pid")
            fail_count=$(( fail_count + 1 ))
            write_health_fail_count "$pid" "$fail_count"
            if (( fail_count >= CONSECUTIVE_HEALTH_FAIL_THRESHOLD )); then
                # 重启前尝试 dump traceback（如果 daemon 注册了 SIGUSR1 handler）
                dump_daemon_traceback_before_restart "$pid"
                mark_restart_needed "HTTP unresponsive ${fail_count} consecutive checks ($last_frame_epoch); see ${DAEMON_TRACEBACK_DUMP} if exists"
                reset_health_fail_count "$pid"
            else
                log WARN "HTTP unresponsive (${fail_count}/${CONSECUTIVE_HEALTH_FAIL_THRESHOLD} consecutive) reason=$last_frame_epoch"
            fi
        fi
    else
        # probe_rc == 2: HTTP OK 但缺 last_frame_ts 字段 — schema 异常，记录但不计入假死
        reset_health_fail_count "$pid"
        write_idle_skip_accum 0
        log WARN "/health responded but missing last_frame_ts field (schema regression?)"
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
    log INFO "thresholds: RSS_WARN=${RSS_WARN_GB}GB RSS_KILL=${RSS_KILL_GB}GB STALE=${STALE_SECONDS}s ROTATE=${LOG_ROTATE_MB}MB HTTP_FAIL=${CONSECUTIVE_HEALTH_FAIL_THRESHOLD}x (connect=${HEALTH_CONNECT_TIMEOUT}s,max=${HEALTH_MAX_TIME}s)"

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
