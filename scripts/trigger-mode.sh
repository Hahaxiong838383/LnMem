#!/usr/bin/env bash
# LnMem trigger 模式切换（V0.3.1，2026-05-16 川哥钦定 dual-mode）
#
# 设计：
# - work（默认）= 生产严阈值，"不打扰为先"（产品定义 §10.1）
# - demo = 测试/体验阈值，方便快速验证 LNN 预测能力
#
# 用法：
#   ./scripts/trigger-mode.sh demo    # 切到 demo 模式 + 自动重启 daemon
#   ./scripts/trigger-mode.sh work    # 切回 work 模式 + 自动重启
#   ./scripts/trigger-mode.sh status  # 查当前模式
#
# 实现：
# - 写 ~/.lnmem/trigger_mode 文件（daemon 启动时读）
# - 写 ~/.lnmem/RESTART_NEEDED 让 supervisor 重启 daemon
# - 等 supervisor 完成重启确认

set -uo pipefail

MODE_FILE="$HOME/.lnmem/trigger_mode"
RESTART_FLAG="$HOME/.lnmem/RESTART_NEEDED"
HEALTH_URL="http://127.0.0.1:9876/health"

show_thresholds() {
    local mode="$1"
    if [[ "$mode" == "demo" ]]; then
        cat <<EOF
demo 模式阈值（频繁触发，方便体验）：
  brow_burst:     window=5 帧 / threshold=0.80 / cooldown=30s
  pose_held:      window=20 帧 / stddev=3.5° / cooldown=30s
  low_vlm_conf:   cooldown=30s
  rate-limit:     60 询问/小时（1 分钟一次）
EOF
    else
        cat <<EOF
work 模式阈值（不打扰为先，生产值）：
  brow_burst:     window=15 帧 / threshold=0.95 / cooldown=300s
  pose_held:      window=60 帧 / stddev=2.5° / cooldown=300s
  low_vlm_conf:   cooldown=120s
  rate-limit:     3 询问/小时（20min 最小间隔）
EOF
    fi
}

status() {
    local current
    if [[ -f "$MODE_FILE" ]]; then
        current=$(cat "$MODE_FILE" 2>/dev/null | tr -d ' \n')
    else
        current="work"
    fi
    echo "current mode: $current"
    echo ""
    show_thresholds "$current"
}

switch_mode() {
    local target="$1"
    local current="work"
    [[ -f "$MODE_FILE" ]] && current=$(cat "$MODE_FILE" 2>/dev/null | tr -d ' \n')

    if [[ "$current" == "$target" ]]; then
        echo "已经是 $target 模式，无需切换"
        echo ""
        show_thresholds "$target"
        return 0
    fi

    mkdir -p "$(dirname "$MODE_FILE")"
    printf '%s\n' "$target" > "$MODE_FILE"
    echo "✅ 模式已写入 $MODE_FILE → $target"
    echo ""

    # 触发 daemon 重启
    date '+%Y-%m-%dT%H:%M:%S%z' > "$RESTART_FLAG"
    echo "trigger-mode-switch-to-$target" >> "$RESTART_FLAG"
    echo "📨 已写 RESTART_NEEDED，等 supervisor 重启 daemon..."

    # 等 supervisor 处理（最多 60s）
    local i=0
    while [[ -f "$RESTART_FLAG" && $i -lt 60 ]]; do
        sleep 2
        i=$((i + 2))
    done
    if [[ -f "$RESTART_FLAG" ]]; then
        echo "⚠️  60s 内未处理 RESTART_NEEDED，supervisor 可能未跑。请手动重启 daemon。"
        return 1
    fi

    # 验证新 daemon 健康
    sleep 2
    if curl -sS -m 5 "$HEALTH_URL" >/dev/null 2>&1; then
        echo "✅ daemon 已重启并响应 /health（新 $target 模式生效）"
        echo ""
        show_thresholds "$target"
    else
        echo "⚠️  daemon 已重启但 /health 不响应，需排查"
        return 1
    fi
}

case "${1:-status}" in
    demo)  switch_mode demo ;;
    work)  switch_mode work ;;
    status) status ;;
    *)
        echo "Usage: $0 {demo|work|status}"
        exit 2
        ;;
esac
