#!/usr/bin/env bash
# studio_nightly_train.sh — Studio 夜间 03:00 跑 LNN intent classifier 训练 + ckpt 推回 Air
#
# Q2=A 钦定方案（ADR-028 v0.1 Sprint 2）：
#   1. 看新增反馈条数 ≥ 50（FEEDBACK_MIN）才训，否则跳过
#   2. 训 intent_head（codex Sprint 2-B 实现）
#   3. ckpt rsync 回 Air，daemon 看 ckpt mtime 软重启 inferencer
#
# 部署：Studio 上跑（这个脚本通过 git 同步过去）
# Cron: 03:00 daily（用 LaunchAgent 配，见 docs/operations/studio-cron.md）
#
# 用法（Studio 上）：
#   /Users/fu-121/lnmem-poc/scripts/studio_nightly_train.sh           # 正常跑
#   /Users/fu-121/lnmem-poc/scripts/studio_nightly_train.sh --force   # 跳过 feedback 阈值强制训
#   /Users/fu-121/lnmem-poc/scripts/studio_nightly_train.sh --dry     # dry-run

set -euo pipefail

# ---- 配置（Studio 视角） ----
REPO_DIR="${LNMEM_REPO:-${HOME}/lnmem-poc}"           # Studio 已有 ~/lnmem-poc clone
DATA_DIR="${LNMEM_DATA_DIR:-${HOME}/lnmem-daemon-sync}"  # Air rsync 过来的 daemon 数据（避开 Studio 训练集 ~/lnmem-data）
CKPT_DIR="${LNMEM_CKPT_DIR:-${HOME}/lnmem-train/ckpt}" # 训练 ckpt 输出（Studio 已有 ~/lnmem-train）
LOG_DIR="${HOME}/lnmem-train/logs/train"
LOCK_FILE="/tmp/lnmem_nightly_train.lock"
FEEDBACK_MIN=50                                        # 新增反馈条数阈值
# 注：Air 无 SSH 别名（DHCP 动态 IP），改用 Air 主动 pull 模式
# Studio 这里只负责把 ckpt 落到本地 + 维护 intent_head_latest.pt 符号链接
# Air 端的 cron 跑 scripts/pull_ckpt_from_studio.sh 拉过去（每 10min）

# ---- 参数 ----
FORCE_TRAIN=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE_TRAIN=1 ;;
        --dry|--dry-run) DRY_RUN=1 ;;
        --help|-h) grep '^#' "$0" | head -20; exit 0 ;;
    esac
done

# ---- 锁 ----
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" >/dev/null 2>&1; then
        echo "[SKIP] 训练仍在跑 (PID=$OLD_PID)"; exit 0
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM

mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="${LOG_DIR}/nightly-$(date +%Y-%m-%d).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "===== nightly train start ====="
log "REPO: $REPO_DIR  DATA: $DATA_DIR  CKPT: $CKPT_DIR"

# ---- 1. 拉最新代码 ----
if [ "$DRY_RUN" -eq 0 ]; then
    log "git pull..."
    (cd "$REPO_DIR" && git pull --rebase --autostash 2>&1 | tail -3) | tee -a "$LOG_FILE"
fi

# ---- 2. 数据健康度检查 ----
SIGNAL_COUNT=$(find "$DATA_DIR" -name 'signals_*.jsonl' -mtime -1 2>/dev/null | wc -l | tr -d ' ')
FEEDBACK_COUNT=0
if [ -d "$DATA_DIR/intent_feedback" ]; then
    # 今天 + 昨天的 intent_feedback 总条数
    FEEDBACK_COUNT=$(cat "$DATA_DIR/intent_feedback/"*.jsonl 2>/dev/null | wc -l | tr -d ' ' || echo 0)
fi
log "data: signal_files_today=$SIGNAL_COUNT  total_feedback=$FEEDBACK_COUNT"

if [ "$FORCE_TRAIN" -eq 0 ] && [ "$FEEDBACK_COUNT" -lt "$FEEDBACK_MIN" ]; then
    log "[SKIP] 反馈不足 ($FEEDBACK_COUNT < $FEEDBACK_MIN)，跳过训练（--force 可强制）"
    exit 0
fi

# ---- 3. 训练（调用 poc/training/train_intent_classifier.py） ----
TRAIN_SCRIPT="${REPO_DIR}/poc/training/train_intent_classifier.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    log "[ERR] 训练脚本不存在: $TRAIN_SCRIPT（codex Sprint 2-B 待出）"
    exit 3
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_CKPT="${CKPT_DIR}/intent_head_${TIMESTAMP}.pt"
TRAIN_LOG="${LOG_DIR}/train-${TIMESTAMP}.log"

log "training → $OUT_CKPT"
if [ "$DRY_RUN" -eq 0 ]; then
    cd "$REPO_DIR"
    if .venv/bin/python "$TRAIN_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --out-ckpt "$OUT_CKPT" \
        --epochs 20 \
        --batch-size 32 \
        --device mps \
        > "$TRAIN_LOG" 2>&1; then
        log "[OK] 训练完成"
    else
        log "[ERR] 训练失败，看 $TRAIN_LOG"
        tail -20 "$TRAIN_LOG" | tee -a "$LOG_FILE"
        exit 4
    fi
else
    log "[DRY] 跳过训练"
fi

# ---- 4. 维护 latest symlink（Air 端 pull 看这个） ----
if [ "$DRY_RUN" -eq 0 ] && [ -f "$OUT_CKPT" ]; then
    ln -sf "$(basename $OUT_CKPT)" "${CKPT_DIR}/intent_head_latest.pt"
    log "[OK] latest symlink → $(basename $OUT_CKPT)"
    log "    Air 端 cron 跑 pull_ckpt_from_studio.sh 自动拉过去"
fi

# ---- 5. ckpt 清理（保留近 7 个） ----
cd "$CKPT_DIR" && ls -t intent_head_*.pt 2>/dev/null | tail -n +8 | xargs rm -f 2>/dev/null || true

log "===== nightly train ok ====="
