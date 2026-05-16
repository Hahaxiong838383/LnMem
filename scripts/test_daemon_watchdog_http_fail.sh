#!/usr/bin/env bash
# W2-P0-bugfix（2026-05-16）：watchdog HTTP 假死处理回归测试。
#
# 覆盖场景：
#   T1. HTTP 完全不响应（curl 失败）3 次 → mark RESTART_NEEDED + 重置计数
#   T2. HTTP 失败 2 次后恢复 1 次 → 计数重置回 0，不重启
#   T3. HTTP 响应但缺 last_frame_ts → WARN，不计入失败
#   T4. 用户 idle 期间 HTTP 失败 → 不计数
#   T5. 持久化文件存在错误内容 → 安全回退到 0
#
# 不真起 daemon，靠 stub HEALTH_URL + 临时目录隔离。

set -uo pipefail

TEST_DIR="$(mktemp -d -t lnmem-watchdog-test-XXXXXX)"
WATCHDOG="$(cd "$(dirname "$0")" && pwd)/daemon_watchdog.sh"
# stub：404 + 跑得通的本地端口（用 python3 临时 server）
STUB_PORT=$((30000 + RANDOM % 5000))
STUB_PID=""
# codex F-05 fix：起 fake daemon（自带唯一标识 token），不依赖外部 bash 进程
FAKE_DAEMON_TOKEN="lnmem-test-fake-daemon-$$-$RANDOM"
FAKE_DAEMON_PID=""

start_fake_daemon() {
    # 用 python3 起一个 sleep 进程，cmdline 含 FAKE_DAEMON_TOKEN 唯一标识
    # 这样 pgrep -f "$FAKE_DAEMON_TOKEN" 能稳定找到，且不会误抓其他进程
    python3 -c "
import sys, time, signal as sig
# 解析 token 但不用，目的是让 cmdline 含 token
_token = sys.argv[1] if len(sys.argv) > 1 else ''
def _handle(*a): pass
# 装 SIGUSR1 handler 让测试能验证发信号成功（不真 dump）
sig.signal(sig.SIGUSR1, _handle)
time.sleep(300)
" "$FAKE_DAEMON_TOKEN" >/dev/null 2>&1 &
    FAKE_DAEMON_PID=$!
    sleep 0.3
    # 校验起来了
    if ! kill -0 "$FAKE_DAEMON_PID" 2>/dev/null; then
        echo "ERROR: fake daemon failed to start" >&2
        exit 1
    fi
}

stop_fake_daemon() {
    [[ -n "$FAKE_DAEMON_PID" ]] && kill "$FAKE_DAEMON_PID" 2>/dev/null && wait "$FAKE_DAEMON_PID" 2>/dev/null
    FAKE_DAEMON_PID=""
}

cleanup() {
    [[ -n "$STUB_PID" ]] && kill "$STUB_PID" 2>/dev/null || true
    stop_fake_daemon
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# 启动 fake daemon 作为测试的 daemon target
start_fake_daemon

start_stub() {
    local mode="$1"  # ok | no_ts | dead
    case "$mode" in
        dead)
            STUB_PID=""  # 不起 server，让 curl connect refused
            ;;
        ok)
            # 用当前时间作 last_frame_ts，避免 STALE 误触
            python3 -c "
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone
import json
class H(BaseHTTPRequestHandler):
    def do_GET(s):
        s.send_response(200); s.send_header('Content-Type','application/json'); s.end_headers()
        now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        s.wfile.write(json.dumps({'status':'ok','last_frame_ts': now_iso}).encode())
    def log_message(s,*a): pass
HTTPServer(('127.0.0.1', $STUB_PORT), H).serve_forever()
" &
            STUB_PID=$!
            ;;
        no_ts)
            python3 -c "
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
class H(BaseHTTPRequestHandler):
    def do_GET(s):
        s.send_response(200); s.send_header('Content-Type','application/json'); s.end_headers()
        s.wfile.write(json.dumps({'status':'ok'}).encode())
    def log_message(s,*a): pass
HTTPServer(('127.0.0.1', $STUB_PORT), H).serve_forever()
" &
            STUB_PID=$!
            ;;
    esac
    # F-08 fix（codex 二审）：start_stub 后必须显式校验存活 + 端口可连通
    if [[ "$1" != "dead" ]]; then
        local tries=20
        while (( tries > 0 )); do
            # PID 还活着 + 端口能连
            if kill -0 "$STUB_PID" 2>/dev/null && \
               python3 -c "import socket; s=socket.socket(); s.settimeout(0.2); s.connect(('127.0.0.1',$STUB_PORT)); s.close()" 2>/dev/null; then
                return
            fi
            sleep 0.1
            tries=$(( tries - 1 ))
        done
        echo "ERROR: stub failed to come up on port $STUB_PORT (mode=$1)" >&2
        exit 1
    fi
}

stop_stub() {
    [[ -n "$STUB_PID" ]] && kill "$STUB_PID" 2>/dev/null && wait "$STUB_PID" 2>/dev/null
    STUB_PID=""
    # 等端口真释放（macOS TIME_WAIT 不需要等，但保险）
    sleep 0.2
}

run_once() {
    HEALTH_URL="http://127.0.0.1:$STUB_PORT/health" \
    WATCHDOG_LOG="$TEST_DIR/watchdog.log" \
    RESTART_FLAG="$TEST_DIR/RESTART_NEEDED" \
    HEALTH_FAIL_COUNT_FILE="$TEST_DIR/health_fail_count" \
    IDLE_SKIP_ACCUM_FILE="$TEST_DIR/idle_skip_accum" \
    SUPERVISOR_ERROR_FLAG="$TEST_DIR/SUPERVISOR_ERROR" \
    SIGNALS_DIR="$TEST_DIR/data" \
    DAEMON_TRACEBACK_DUMP="$TEST_DIR/daemon_traceback.log" \
    DAEMON_NAME="$FAKE_DAEMON_TOKEN" \
    INTERVAL="${INTERVAL_OVERRIDE:-1}" \
    STALE_SECONDS="60" \
    USER_IDLE_THRESHOLD="${USER_IDLE_THRESHOLD_OVERRIDE:-99999}" \
    MAX_IDLE_SKIP_SECONDS="${MAX_IDLE_SKIP_SECONDS_OVERRIDE:-3600}" \
    HEALTH_CONNECT_TIMEOUT="2" \
    HEALTH_MAX_TIME="3" \
    "$WATCHDOG" --once
}

fail() { printf '❌ %s\n' "$*" >&2; exit 1; }
pass() { printf '✅ %s\n' "$*"; }

# 解析"<pid>:<count>"格式取 count（兼容旧纯数字格式）
get_count() {
    local raw
    raw=$(cat "$TEST_DIR/health_fail_count" 2>/dev/null | tr -d ' \n')
    [[ -z "$raw" ]] && { echo 0; return; }
    if [[ "$raw" =~ ^[0-9]+$ ]]; then echo "$raw"; return; fi
    if [[ "$raw" =~ ^[0-9]+:[0-9]+$ ]]; then echo "${raw#*:}"; return; fi
    echo 0
}

# ============ T1: HTTP dead 3 次 → RESTART ============
echo "== T1: 3 consecutive HTTP fail → mark RESTART =="
start_stub dead
rm -f "$TEST_DIR/RESTART_NEEDED" "$TEST_DIR/health_fail_count" "$TEST_DIR/idle_skip_accum"
run_once >/dev/null 2>&1
[[ "$(get_count)" == "1" ]] || fail "T1.1 count != 1 (got $(get_count))"
[[ ! -f "$TEST_DIR/RESTART_NEEDED" ]] || fail "T1.1 RESTART_NEEDED should NOT exist yet"
run_once >/dev/null 2>&1
[[ "$(get_count)" == "2" ]] || fail "T1.2 count != 2"
[[ ! -f "$TEST_DIR/RESTART_NEEDED" ]] || fail "T1.2 RESTART_NEEDED should NOT exist yet"
run_once >/dev/null 2>&1
[[ -f "$TEST_DIR/RESTART_NEEDED" ]] || fail "T1.3 RESTART_NEEDED missing after 3rd fail"
grep -q "HTTP unresponsive" "$TEST_DIR/RESTART_NEEDED" || fail "T1.3 RESTART_NEEDED missing 'HTTP unresponsive'"
[[ "$(get_count)" == "0" ]] || fail "T1.3 count not reset to 0 after RESTART trigger"
# F-03 验证：daemon SIGUSR1 应被 watchdog 发送（fake daemon 装了 handler 但不真 dump，故 mtime 不变 → WARN）
grep -qE "SIGUSR1.*PID.*verified" "$TEST_DIR/watchdog.log" || fail "T1.3 missing SIGUSR1 send log"
grep -qE "SIGUSR1 dump NOT OBSERVED" "$TEST_DIR/watchdog.log" || fail "T1.3 missing dump-not-observed warning (since fake daemon doesn't dump)"
pass "T1 PASS"

# ============ T2: 2 次失败后 1 次成功 → 计数重置 ============
echo "== T2: 2 fails then 1 OK → counter reset =="
rm -f "$TEST_DIR/RESTART_NEEDED" "$TEST_DIR/health_fail_count" "$TEST_DIR/idle_skip_accum"
start_stub dead
run_once >/dev/null 2>&1
run_once >/dev/null 2>&1
[[ "$(get_count)" == "2" ]] || fail "T2 setup count != 2"
stop_stub
start_stub ok
run_once >/dev/null 2>&1
[[ "$(get_count)" == "0" ]] || fail "T2 count not reset after HTTP OK"
[[ ! -f "$TEST_DIR/RESTART_NEEDED" ]] || fail "T2 RESTART_NEEDED should not exist"
stop_stub
pass "T2 PASS"

# ============ T3: HTTP OK 但缺 last_frame_ts ============
echo "== T3: HTTP 200 but missing last_frame_ts → WARN =="
rm -f "$TEST_DIR/RESTART_NEEDED" "$TEST_DIR/health_fail_count" "$TEST_DIR/idle_skip_accum"
start_stub no_ts
run_once >/dev/null 2>&1
[[ ! -f "$TEST_DIR/RESTART_NEEDED" ]] || fail "T3 RESTART_NEEDED should not exist"
grep -q "missing last_frame_ts" "$TEST_DIR/watchdog.log" || fail "T3 missing schema warn log"
[[ "$(get_count)" == "0" ]] || fail "T3 count != 0"
stop_stub
pass "T3 PASS"

# ============ T4: 用户 idle 期间 HTTP 失败不计数（窗口内）============
# THRESHOLD=-1 让 watchdog 把"任何非负 idle 值"都视为 idle（cc 在跑时 idle=0）
echo "== T4: HTTP fail during user idle (within MAX_IDLE_SKIP) → no count =="
rm -f "$TEST_DIR/RESTART_NEEDED" "$TEST_DIR/health_fail_count" "$TEST_DIR/idle_skip_accum"
start_stub dead
USER_IDLE_THRESHOLD_OVERRIDE=-1 run_once >/dev/null 2>&1
[[ "$(get_count)" == "0" ]] || fail "T4 count != 0 during idle (got $(get_count))"
[[ ! -f "$TEST_DIR/RESTART_NEEDED" ]] || fail "T4 RESTART_NEEDED should not exist during idle"
grep -qE "skip fail count" "$TEST_DIR/watchdog.log" || fail "T4 missing skip-on-idle log"
pass "T4 PASS"

# ============ T5: 计数文件损坏 → 安全回退到 0 ============
echo "== T5: corrupt counter file → safe reset to 0 =="
rm -f "$TEST_DIR/RESTART_NEEDED" "$TEST_DIR/idle_skip_accum"
echo "not-a-number-garbage" > "$TEST_DIR/health_fail_count"
start_stub dead
run_once >/dev/null 2>&1
val=$(get_count)
[[ "$val" == "1" ]] || fail "T5 expected count=1 after corrupt reset, got $val"
pass "T5 PASS"

# ============ T6 (codex F-01): daemon PID 代际切换清零 ============
echo "== T6: PID generation change wipes stale counter (F-01) =="
rm -f "$TEST_DIR/RESTART_NEEDED" "$TEST_DIR/idle_skip_accum"
# 模拟旧 daemon PID=99999 残留计数 2
echo "99999:2" > "$TEST_DIR/health_fail_count"
start_stub dead
# 当前 FAKE_DAEMON_PID != 99999 → 读取应回 0，本次失败为第 1 次
run_once >/dev/null 2>&1
val=$(get_count)
[[ "$val" == "1" ]] || fail "T6 expected new-gen count=1 (got $val), file=$(cat $TEST_DIR/health_fail_count)"
# 文件应被改写为 <FAKE_PID>:1
raw=$(cat "$TEST_DIR/health_fail_count")
[[ "$raw" == "$FAKE_DAEMON_PID:1" ]] || fail "T6 file should be $FAKE_DAEMON_PID:1 got $raw"
stop_stub
pass "T6 PASS"

# ============ T7 (codex F-04): idle 累计超上限强制计入 ============
echo "== T7: idle skip accum > MAX_IDLE_SKIP_SECONDS → falls back to count (F-04) =="
rm -f "$TEST_DIR/RESTART_NEEDED" "$TEST_DIR/health_fail_count"
# 预置 idle_skip_accum=10000（远超下面的 60s 上限）
echo 10000 > "$TEST_DIR/idle_skip_accum"
start_stub dead
USER_IDLE_THRESHOLD_OVERRIDE=-1 MAX_IDLE_SKIP_SECONDS_OVERRIDE=60 INTERVAL_OVERRIDE=1 run_once >/dev/null 2>&1
[[ "$(get_count)" == "1" ]] || fail "T7 expected count=1 after idle upper bound exceeded, got $(get_count)"
grep -qE "idle-skip.*upper bound.*falling back" "$TEST_DIR/watchdog.log" || fail "T7 missing upper-bound fallback log"
stop_stub
pass "T7 PASS"

echo ""
echo "🎉 ALL 7 TESTS PASS"
