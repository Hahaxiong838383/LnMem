#!/usr/bin/env bash
# pull_ckpt_from_studio.sh — Air 主动 pull Studio 训完的 ckpt（Q2=A v0.1 Sprint 2）
#
# 配套：scripts/studio_nightly_train.sh（Studio 夜间训练后维护 intent_head_latest.pt symlink）
# 频率：Air cron 每 10 分钟跑（凌晨训完很快就拉过来）
# 触发热重载：daemon 看 ~/.lnmem/ckpt/intent_head_latest.pt mtime 变 → 软重启 LNN inferencer

set -euo pipefail

SRC_HOST="${LNMEM_STUDIO_HOST:-studio-pub}"  # 默认走公网
SRC_PATH="~/lnmem-train/ckpt/"   # Studio 已有 lnmem-train 目录
DST_PATH="${HOME}/.lnmem/ckpt/"
LOCK_FILE="/tmp/lnmem_pull_ckpt.lock"
LOG_DIR="${HOME}/.lnmem/logs/ckpt_pull"
LOG_FILE="${LOG_DIR}/pull-$(date +%Y-%m-%d).log"

mkdir -p "$DST_PATH" "$LOG_DIR"

# ---- 锁 ----
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" >/dev/null 2>&1; then
        exit 0   # 静默跳过（cron 频繁不需要噪音）
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM

# ---- 探测 Studio 可达性（轻量，5s 超时） ----
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$SRC_HOST" "test -f $SRC_PATH/intent_head_latest.pt" 2>/dev/null; then
    # Studio 不可达或没 ckpt，静默退出（晨起脚本不要 spam）
    exit 0
fi

# ---- 检查是否有新 ckpt（用 sha 比 latest） ----
REMOTE_TARGET=$(ssh "$SRC_HOST" "readlink ${SRC_PATH}intent_head_latest.pt" 2>/dev/null || echo "")
if [ -z "$REMOTE_TARGET" ]; then
    exit 0
fi

LOCAL_LATEST="${DST_PATH}intent_head_latest.pt"
LOCAL_TARGET=""
if [ -L "$LOCAL_LATEST" ]; then
    LOCAL_TARGET=$(readlink "$LOCAL_LATEST")
fi

if [ "$REMOTE_TARGET" = "$LOCAL_TARGET" ] && [ -f "$LOCAL_LATEST" ]; then
    # 已是最新，静默退出
    exit 0
fi

# ---- 拉 ----
echo "===== pull ckpt $(date '+%H:%M:%S') =====" >> "$LOG_FILE"
echo "remote latest → $REMOTE_TARGET" >> "$LOG_FILE"
echo "local latest  → ${LOCAL_TARGET:-<none>}" >> "$LOG_FILE"

RSYNC_BIN=$(command -v /opt/homebrew/bin/rsync || echo rsync)

# 拉新 ckpt 文件本身
if "$RSYNC_BIN" -a --partial --timeout=60 \
    "${SRC_HOST}:${SRC_PATH}${REMOTE_TARGET}" "${DST_PATH}${REMOTE_TARGET}" \
    >> "$LOG_FILE" 2>&1; then
    # 原子更新 symlink（先写临时再 mv，防 daemon 读到半截）
    ln -sf "$REMOTE_TARGET" "${LOCAL_LATEST}.tmp"
    mv -f "${LOCAL_LATEST}.tmp" "$LOCAL_LATEST"
    echo "[OK] ckpt updated → $REMOTE_TARGET" >> "$LOG_FILE"
    # 留近 7 个，删旧
    cd "$DST_PATH" && ls -t intent_head_*.pt 2>/dev/null | tail -n +8 | xargs rm -f 2>/dev/null || true
else
    echo "[ERR] rsync failed" >> "$LOG_FILE"
    exit 1
fi
