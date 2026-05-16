#!/usr/bin/env bash
# LnMem signal_path 切换（ADR-005 v0.3.2，2026-05-16）
#
# 设计：
# - calibrated（默认）= 历史路径，trigger 用 expression.brow_furrow > 0.95
# - raw = ADR-005 推荐，trigger 用 raw_au.AU04 > 0.55（P91 数据驱动）
#
# 用法：
#   ./scripts/signal-path.sh raw         # 切到 raw + 自动重启 daemon
#   ./scripts/signal-path.sh calibrated  # 切回 calibrated
#   ./scripts/signal-path.sh status      # 查当前
#
# 与 ./scripts/trigger-mode.sh 同范式（env > file > default）

set -uo pipefail

PATH_FILE="$HOME/.lnmem/signal_path"
RESTART_FLAG="$HOME/.lnmem/RESTART_NEEDED"
HEALTH_URL="http://127.0.0.1:9876/health"

show_thresholds() {
    local sp="$1"
    if [[ "$sp" == "raw" ]]; then
        cat <<EOF
raw signal_path（ADR-005，推荐）：
  brow_burst:  raw_au.AU04 > 0.55（P91 数据驱动，川哥 4 天 235K 帧 sweep）
  pose_held:   不变（pose 信号本来就健康）
  策略：       trigger 应用层用 raw；LNN 演化层让 tau_drift 吸收 baseline drift
  预期触发：    ~5-6 次/天 brow（work 节奏）
EOF
    else
        cat <<EOF
calibrated signal_path（向后兼容默认）：
  brow_burst:  expression.brow_furrow > 0.95（calibration 后字段）
  pose_held:   不变
  ⚠️  注意：    D-1 数据表明此路径在川哥脸上有 22% 信息熵损失（polarized 双峰）
EOF
    fi
}

status() {
    local current="calibrated"
    [[ -f "$PATH_FILE" ]] && current=$(cat "$PATH_FILE" 2>/dev/null | tr -d ' \n')
    echo "current signal_path: $current"
    echo ""
    show_thresholds "$current"
}

switch_path() {
    local target="$1"
    local current="calibrated"
    [[ -f "$PATH_FILE" ]] && current=$(cat "$PATH_FILE" 2>/dev/null | tr -d ' \n')

    if [[ "$current" == "$target" ]]; then
        echo "已经是 $target signal_path，无需切换"
        echo ""
        show_thresholds "$target"
        return 0
    fi

    mkdir -p "$(dirname "$PATH_FILE")"
    printf '%s\n' "$target" > "$PATH_FILE"
    echo "✅ signal_path 已写入 $PATH_FILE → $target"
    echo ""

    date '+%Y-%m-%dT%H:%M:%S%z' > "$RESTART_FLAG"
    echo "signal-path-switch-to-$target" >> "$RESTART_FLAG"
    echo "📨 已写 RESTART_NEEDED，等 supervisor 重启 daemon..."

    local i=0
    while [[ -f "$RESTART_FLAG" && $i -lt 60 ]]; do
        sleep 2
        i=$((i + 2))
    done
    if [[ -f "$RESTART_FLAG" ]]; then
        echo "⚠️  60s 内未处理 RESTART_NEEDED，请手动重启 daemon"
        return 1
    fi

    sleep 2
    if curl -sS -m 5 "$HEALTH_URL" >/dev/null 2>&1; then
        echo "✅ daemon 已重启并响应 /health（新 $target signal_path 生效）"
        echo ""
        show_thresholds "$target"
    else
        echo "⚠️  daemon 已重启但 /health 不响应，需排查"
        return 1
    fi
}

case "${1:-status}" in
    raw)         switch_path raw ;;
    calibrated)  switch_path calibrated ;;
    status)      status ;;
    *)
        echo "Usage: $0 {raw|calibrated|status}"
        exit 2
        ;;
esac
