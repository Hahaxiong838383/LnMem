#!/usr/bin/env bash
# LnMem trigger 模式切换（V0.3.1，2026-05-16 川哥钦定 dual-mode）
#
# 设计：
# - work（默认）= 生产严阈值，"不打扰为先"（产品定义 §10.1）
# - demo = 测试/体验阈值，方便快速验证 LNN 预测能力
#
# 用法：
#   ./scripts/trigger-mode.sh demo            # 切到 demo 模式 + 自动重启 daemon
#   ./scripts/trigger-mode.sh work            # 切回 work 模式 + 自动重启
#   ./scripts/trigger-mode.sh demo --force    # 绕过 supervisor 5min cooldown（倒推 last_action_ts）
#   ./scripts/trigger-mode.sh status          # 查当前模式
#
# 实现：
# - 写 ~/.lnmem/trigger_mode 文件（daemon 启动时读）
# - 写 ~/.lnmem/RESTART_NEEDED 让 supervisor 重启 daemon
# - 等 supervisor 完成重启确认
#
# Cooldown 协作（V0.3.2，2026-05-16）：
# - supervisor 有 5min thrash cooldown（RESTART_COOLDOWN_S=300，记 RESTART_NEEDED.last_action_ts）
# - 本脚本在写 RESTART_NEEDED 前主动读 last_action_ts 检测剩余冷却时间，等过期再写
# - --force 用于绕过（仅 dev/debug；倒推 last_action_ts 让 supervisor 立刻处理）

set -uo pipefail

MODE_FILE="$HOME/.lnmem/trigger_mode"
RESTART_FLAG="$HOME/.lnmem/RESTART_NEEDED"
LAST_ACTION_TS_FILE="$RESTART_FLAG.last_action_ts"
SUPERVISOR_LOG="$HOME/.lnmem/supervisor.log"
HEALTH_URL="http://127.0.0.1:9876/health"
COOLDOWN_S="${COOLDOWN_S:-300}"  # 与 supervisor RESTART_COOLDOWN_S 对齐；测试可 env override

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

wait_for_cooldown() {
    # 与 supervisor 的 5min thrash cooldown 协作：在写 RESTART_NEEDED 前等冷却过期。
    # --force 时调用方会先倒推 last_action_ts，本函数读到的 elapsed 已 >= COOLDOWN_S，立刻返回。
    [[ -f "$LAST_ACTION_TS_FILE" ]] || return 0

    local last_epoch now_epoch elapsed remaining
    last_epoch=$(cat "$LAST_ACTION_TS_FILE" 2>/dev/null || echo "0")
    [[ "$last_epoch" =~ ^[0-9]+$ ]] || return 0  # 文件损坏当作无 cooldown

    now_epoch=$(date +%s)
    elapsed=$((now_epoch - last_epoch))
    if [[ $elapsed -ge $COOLDOWN_S ]]; then
        return 0
    fi

    remaining=$((COOLDOWN_S - elapsed))
    echo "⏳ supervisor cooldown 剩余 ${remaining} 秒（防 thrash 设计，RESTART_COOLDOWN_S=${COOLDOWN_S}），等待中..."
    echo "   （如需立刻切换，重跑加 --force）"
    sleep "$remaining"
    # 多加 1s 确保 supervisor 下一轮扫描时 elapsed >= COOLDOWN_S
    sleep 1
    echo "✅ cooldown 已过期，继续"
}

bypass_cooldown() {
    # 倒推 last_action_ts 到 COOLDOWN_S+1 秒前，让 supervisor 下一轮认为冷却已过。
    # 注意：仅绕过 trigger 触发的等待，不修改 supervisor 本身 5min cooldown 设计。
    local backdated
    backdated=$(( $(date +%s) - COOLDOWN_S - 1 ))
    mkdir -p "$(dirname "$LAST_ACTION_TS_FILE")"
    printf '%s\n' "$backdated" > "$LAST_ACTION_TS_FILE"
    echo "⚡ --force 已倒推 last_action_ts → $backdated（绕过 cooldown）"
}

switch_mode() {
    local target="$1"
    local force="${2:-0}"
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

    if [[ "$force" == "1" ]]; then
        bypass_cooldown
    else
        wait_for_cooldown
    fi

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
        echo "⚠️  60s 内 RESTART_NEEDED 未被消费。supervisor 已在冷却或异常，请查 ${SUPERVISOR_LOG}"
        echo "   （cooldown 检查见 daemon_supervisor.sh handle_restart_flag；如确需立刻切换可加 --force）"
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

# 解析参数：支持 `demo --force` / `--force demo` / `demo` / `status`
ACTION=""
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        demo|work|status)
            if [[ -z "$ACTION" ]]; then
                ACTION="$arg"
            else
                echo "Usage: $0 {demo|work|status} [--force]" >&2
                exit 2
            fi
            ;;
        *)
            echo "Usage: $0 {demo|work|status} [--force]" >&2
            exit 2
            ;;
    esac
done
ACTION="${ACTION:-status}"

case "$ACTION" in
    demo)  switch_mode demo "$FORCE" ;;
    work)  switch_mode work "$FORCE" ;;
    status) status ;;
esac
