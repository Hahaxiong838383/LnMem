#!/usr/bin/env bash
# sync_daemon_to_studio.sh — Air → Studio 单向 rsync（Q1=A 钦定 ADR-028 v0.1 Sprint 2）
#
# 范围：~/.lnmem/data/ 下的 signal jsonl + intent_feedback + au_cache（训练所需）
# 排除：raw_video（14GB，训练不用且 Studio 空间紧张 76GB）
# 频率：建议 cron @hourly（用 LaunchAgent 见下方）
# 锁：单实例 flock，防 daemon 写入时 rsync 撕半
#
# 5/22 教训 (feedback_rsync_exclude_delete_pitfall)：
#   --exclude='dir/' 配 --delete 会删 destination 的 dir/！
#   本脚本不用 --delete（增量追加为主，删历史走单独脚本）
#
# 用法：
#   ./scripts/sync_daemon_to_studio.sh           # 默认 incremental
#   ./scripts/sync_daemon_to_studio.sh --dry     # dry-run 看会传啥
#   ./scripts/sync_daemon_to_studio.sh --verbose # 详细日志

set -euo pipefail

# ---- 配置 ----
SRC="${HOME}/.lnmem/data/"
DST_HOST="${LNMEM_STUDIO_HOST:-studio-pub}"  # 默认走公网（cli4.hahaxiong.cc 跳板），家里 iBRO 可 export LNMEM_STUDIO_HOST=studio-ibro
DST_PATH="~/lnmem-daemon-sync/"   # 新建目录避免与 Studio 现有 ~/lnmem-data（训练集 cache）冲突
LOCK_FILE="/tmp/lnmem_sync_daemon_to_studio.lock"
LOG_DIR="${HOME}/.lnmem/logs/sync"
LOG_FILE="${LOG_DIR}/sync-$(date +%Y-%m-%d).log"

# ---- 参数 ----
DRY_RUN=""
VERBOSE=""
for arg in "$@"; do
    case "$arg" in
        --dry|--dry-run) DRY_RUN="--dry-run" ;;
        --verbose|-v) VERBOSE="-v" ;;
        --help|-h)
            grep '^#' "$0" | head -25
            exit 0
            ;;
    esac
done

# ---- 锁（macOS 无 flock，用 PID 文件 + ps 检查） ----
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" >/dev/null 2>&1; then
        echo "[SKIP] 上一次 sync 仍在跑 (PID=$OLD_PID)，退出"
        exit 0
    else
        echo "[WARN] 发现陈旧锁 (PID=$OLD_PID 已死)，清理"
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM

mkdir -p "$LOG_DIR"

# ---- 日志头 ----
{
    echo ""
    echo "===== sync_daemon_to_studio.sh $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "SRC: $SRC"
    echo "DST: ${DST_HOST}:${DST_PATH}"
    [ -n "$DRY_RUN" ] && echo "MODE: DRY-RUN"
} | tee -a "$LOG_FILE"

# ---- Studio 侧目录存在性 ----
if ! ssh -o ConnectTimeout=10 "$DST_HOST" "mkdir -p ${DST_PATH}" 2>>"$LOG_FILE"; then
    echo "[ERR] Studio 不可达或建目录失败" | tee -a "$LOG_FILE"
    exit 2
fi

# ---- rsync 主命令 ----
# 排除清单 (raw_video 太大且训练不需要)
EXCLUDES=(
    --exclude='raw_video/'
    --exclude='*.tmp'
    --exclude='.DS_Store'
    --exclude='*.pyc'
    --exclude='__pycache__/'
)

# 不用 --delete（增量为主，防误删）
# -a: archive (perms+times+links)
# -h: human readable
# --partial: 断网续传
# --info=stats2: 只看统计（macOS openrsync 不支持 progress2，feedback 已记）
RSYNC_OPTS=(
    -a -h --partial
    --timeout=300
    "${EXCLUDES[@]}"
    $DRY_RUN
    $VERBOSE
)

# 显式用 brew rsync (homebrew 装的 rsync 3.x，比 macOS 自带 openrsync 稳)
RSYNC_BIN=$(command -v /opt/homebrew/bin/rsync || command -v /usr/local/bin/rsync || echo rsync)
echo "rsync binary: $RSYNC_BIN" | tee -a "$LOG_FILE"

START_TS=$(date +%s)
if "$RSYNC_BIN" "${RSYNC_OPTS[@]}" "$SRC" "${DST_HOST}:${DST_PATH}" >>"$LOG_FILE" 2>&1; then
    END_TS=$(date +%s)
    DUR=$((END_TS - START_TS))
    echo "[OK] rsync done in ${DUR}s" | tee -a "$LOG_FILE"
else
    EXIT_CODE=$?
    echo "[ERR] rsync exit $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

# ---- Studio 侧校验：列文件数 + 总大小 ----
SUMMARY=$(ssh -o ConnectTimeout=10 "$DST_HOST" "du -sh ${DST_PATH} 2>/dev/null && ls ${DST_PATH} | wc -l" 2>/dev/null || echo "校验失败")
echo "[Studio 侧] $SUMMARY" | tee -a "$LOG_FILE"

# ---- 日志滚动：留近 7 天 ----
find "$LOG_DIR" -name 'sync-*.log' -mtime +7 -delete 2>/dev/null || true

echo "===== sync ok $(date '+%H:%M:%S') =====" | tee -a "$LOG_FILE"
