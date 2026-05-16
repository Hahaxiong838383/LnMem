#!/usr/bin/env bash
# LnMem fit_step launchd cron 包装脚本（M0 W2 P-1，2026-05-16）
#
# 触发自然 fit：检查 trigger_if_ready 阈值 → 满足则跑一次 fit_step → 链式更新 ckpt
#
# 配套：scripts/com.lnmem.fit-step.plist（StartInterval=900 即 15min 轮询）
#
# 链式 ckpt 选择规则：
#   1. 如 m0w2_phase2/ 有 ckpt → 选最新一个作为 ckpt-in（增量学习）
#   2. 否则 fallback 到 day12_with_self/checkpoint_day12.pt（首次 cold start）
#
# 退出码（cron 友好）：
#   0 = triggered 或 below_threshold 等合理未触发（不让 launchd 误报失败）
#   2 = 真错误（no_ckpt / error）
#
# 可观测性：
#   - stdout: 结构化 JSON 报告（每次跑）→ /tmp/lnmem-fit-step.stdout.log（plist 配）
#   - stderr: 人读 progress → /tmp/lnmem-fit-step.stderr.log
#   - 历史 audit: ~/.lnmem/fit_step_history.jsonl（每次 append 一行结果，flock 串行写）
#
# codex 二审修复（2026-05-16）：
#   - F-02：加 flock 互斥锁，防 cron + 手动并发触发竞态
#   - F-03：传 --min-event-ts <ckpt_in 的 fired_at_utc> 给 run_fit_step，仅 fit 增量 events
#   - F-05：history 写入加 flock + fsync 确保并发安全（注释承诺与实现对齐）

set -uo pipefail

REPO_ROOT="${LNMEM_REPO_ROOT:-/Users/rayman.chen/mycc/2-Projects/LnMem-poc}"
VENV_PY="${LNMEM_VENV_PY:-${REPO_ROOT}/.venv-poc-fresh/bin/python3}"
PHASE2_CKPT_DIR="${REPO_ROOT}/poc/data/checkpoints/m0w2_phase2"
COLD_START_CKPT="${REPO_ROOT}/poc/data/checkpoints/day12_with_self/checkpoint_day12.pt"
HISTORY_LOG="${HISTORY_LOG:-$HOME/.lnmem/fit_step_history.jsonl}"
# F-02: 单实例互斥锁（防 launchd cron + 手动调用并发雪崩）
LOCK_FILE="${LOCK_FILE:-$HOME/.lnmem/fit_step.lock}"

cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT" >&2; exit 2; }

# F-02 fix（codex 二审）：flock 获取互斥锁，5s 内拿不到立即退出（不阻塞 launchd）
# macOS flock 通过 shlock(1) 等替代；这里用 mkdir 风格原子锁
#
# ⚠️ F-07 known limitation（codex 三审 2026-05-16）：
# stale lock 自愈仅靠 kill -0 PID 存活判定。若原持锁进程死亡且 PID 被复用为无关进程，
# 仍会误判"running" → 长期假活锁。SIGKILL 不走 trap 也加剧此风险。
# 现状接受：cron StartInterval=900 + 手工运行频次低，PoC 阶段概率极小。
# 长期方案（下轮 PR）：lock 文件存 pid+start_time+cmdline 三元组校验，或改用目录锁 + owner metadata。
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE" || { echo "ERROR: cannot open lock file $LOCK_FILE" >&2; exit 2; }
if command -v flock >/dev/null 2>&1; then
    if ! flock -n 9; then
        echo "$(date '+%Y-%m-%dT%H:%M:%S%z') another fit_step instance running, skip" >&2
        exit 0
    fi
else
    # macOS 默认无 flock 命令；用 fcntl 风格的 perl 单行兜底
    perl -e '
        use Fcntl qw(:flock);
        open(my $fh, ">>", $ENV{"LOCK_FILE"}) or exit 2;
        flock($fh, LOCK_EX | LOCK_NB) or exit 1;
        # 持有锁直到 perl 退出（这里 perl 立即 exec 父 shell 的剩余）
        # 但 perl 退出后锁就释放，无法跨 perl/bash 边界 — 改方案：
        exit 0;
    ' 2>/dev/null
    # 用 ln 软链原子性当兜底（POSIX-portable）
    if ! ln -s "$$" "${LOCK_FILE}.symlink" 2>/dev/null; then
        # 检查 stale lock：链接指向的 PID 不存在 → 清理重建
        existing_pid=$(readlink "${LOCK_FILE}.symlink" 2>/dev/null || echo "")
        if [[ -n "$existing_pid" ]] && ! kill -0 "$existing_pid" 2>/dev/null; then
            rm -f "${LOCK_FILE}.symlink"
            ln -s "$$" "${LOCK_FILE}.symlink" || { echo "lock contention, skip" >&2; exit 0; }
        else
            echo "$(date '+%Y-%m-%dT%H:%M:%S%z') another fit_step instance running (PID $existing_pid), skip" >&2
            exit 0
        fi
    fi
    trap 'rm -f "${LOCK_FILE}.symlink"' EXIT
fi

# venv 存在性校验
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found: $VENV_PY" >&2
    exit 2
fi

# ckpt 选择（链式）
mkdir -p "$PHASE2_CKPT_DIR" 2>/dev/null
CKPT_IN=""
if ls "$PHASE2_CKPT_DIR"/*.pt >/dev/null 2>&1; then
    # 选最新 mtime（按时间排序）
    CKPT_IN=$(ls -t "$PHASE2_CKPT_DIR"/*.pt | head -1)
elif [[ -f "$COLD_START_CKPT" ]]; then
    CKPT_IN="$COLD_START_CKPT"
else
    echo "ERROR: no ckpt available (neither phase2 nor cold-start)" >&2
    exit 2
fi

# F-03 fix（codex 二审）：从 ckpt_in 读 fired_at_utc 作为水位线
# 仅 fit ts > fired_at_utc 的增量 events，避免重复训练同一批数据导致漂移
# cold-start ckpt（day12_with_self）无 phase2_metadata，水位线置空（首轮 fit 全量）
# 兼容旧 ckpt：fired_at_utc 可能是紧凑格式 "20260516T081409Z"，转标准 ISO8601 字符串
MIN_EVENT_TS=""
if [[ "$CKPT_IN" == *m0w2_phase2* ]]; then
    MIN_EVENT_TS=$("$VENV_PY" -c "
import torch, sys, re
try:
    ckpt = torch.load('$CKPT_IN', map_location='cpu', weights_only=False)
    meta = ckpt.get('phase2_metadata') or {}
    ts = meta.get('fired_at_utc', '') or ''
    # F-03 fix: 兼容旧 ckpt 的紧凑格式 \"20260516T081409Z\"
    # 转标准 ISO8601 与 events.ts (\"2026-05-16T...Z\") 字典序对齐
    m = re.match(r'^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z\$', ts)
    if m:
        ts = f'{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}Z'
    print(ts)
except Exception as e:
    print('', end='')
    print(f'WARN: cannot read fired_at_utc from ckpt: {e}', file=sys.stderr)
" 2>>/tmp/lnmem-fit-step.stderr.log)
fi

# 时间戳作业 ID
JOB_TS=$(date '+%Y-%m-%dT%H:%M:%S%z')
JOB_ID="fit-step-$(date '+%Y%m%dT%H%M%SZ')"

echo "[$JOB_TS] $JOB_ID start: ckpt_in=$CKPT_IN min_event_ts=${MIN_EVENT_TS:-<none>}" >&2

# 跑 fit_step（含 F-03 水位线参数）
REPORT_FILE="/tmp/lnmem-fit-step-report-${JOB_ID}.json"
EXTRA_ARGS=()
if [[ -n "$MIN_EVENT_TS" ]]; then
    EXTRA_ARGS+=(--min-event-ts "$MIN_EVENT_TS")
fi
"$VENV_PY" scripts/run_fit_step.py \
    --ckpt-in "$CKPT_IN" \
    --ckpt-out-dir "$PHASE2_CKPT_DIR" \
    "${EXTRA_ARGS[@]}" \
    > "$REPORT_FILE" 2>>/tmp/lnmem-fit-step.stderr.log
EXIT_CODE=$?

# 解析 JSON 结果（不抛错）+ 追加 history
PARSED=$("$VENV_PY" -c "
import json, sys
try:
    with open('$REPORT_FILE') as f:
        r = json.load(f)
    out = {
        'job_id': '$JOB_ID',
        'ts': '$JOB_TS',
        'exit': $EXIT_CODE,
        'ckpt_in': '$CKPT_IN',
        'triggered': r.get('triggered'),
        'reason': r.get('reason'),
        'counts': r.get('counts'),
        'metrics': r.get('metrics'),
        'ckpt_out_path': r.get('ckpt_out_path'),
        'error': r.get('error'),
    }
    print(json.dumps(out, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'job_id':'$JOB_ID','ts':'$JOB_TS','exit':$EXIT_CODE,'parse_error':str(e)}, ensure_ascii=False))
" 2>/dev/null)

# F-05 fix（codex 二审）：history append 单行原子（一次 syscall）+ fsync 防截断
# 此前注释承诺"原子 append"但实际是简单 >>。改正：
# - 外层 LOCK_FILE 已保证单实例（顶部 flock/symlink），不需要文件级 lock
# - 单 printf 调用一次性写完一行（短行单 write syscall 是原子的）
# - fsync 确保落盘（防 daemon kill 时 buffer 丢失）
mkdir -p "$(dirname "$HISTORY_LOG")"
printf '%s\n' "$PARSED" >> "$HISTORY_LOG"
"$VENV_PY" -c "
import os, sys
try:
    fd = os.open('$HISTORY_LOG', os.O_WRONLY | os.O_APPEND)
    os.fsync(fd)
    os.close(fd)
except Exception as _e:
    print(f'WARN: fsync history failed: {_e}', file=sys.stderr)
" 2>>/tmp/lnmem-fit-step.stderr.log

# stdout: 把报告 cat 出来供 launchd 日志看
cat "$REPORT_FILE"
echo "[$JOB_TS] $JOB_ID end: exit=$EXIT_CODE" >&2

# 清 report tmp（避免 /tmp 堆积，history 已留）
rm -f "$REPORT_FILE"

exit "$EXIT_CODE"
