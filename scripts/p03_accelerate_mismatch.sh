#!/usr/bin/env bash
# P0-3 加速 mismatch 累积辅助（M0 W2，2026-05-12）
#
# 用法：摄像头前坐着，跑此脚本，按提示做对应表情。
# 每 65s 自动发 1 次 lnmem-tell（避开 60s rate-limit）。
# 6 次循环 6 个 true_state，覆盖 STATE_LABELS 中 8 个的核心 6 个。
#
# 期望：mismatch 4 → 10（满足 fit_step 自然触发条件）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "scripts/lnmem-tell" ]]; then
    echo "❌ scripts/lnmem-tell 不可执行" >&2
    exit 1
fi

# 6 次循环：(true_state, 表情提示, note)
declare -a STATES=(
    "distracted|做走神/看向别处的表情|加速 1/6 走神"
    "emotional_negative|做皱眉/不悦的表情|加速 2/6 皱眉"
    "focused|做专注/凝视屏幕的表情|加速 3/6 专注"
    "emotional_positive|做微笑/愉悦的表情|加速 4/6 微笑"
    "fatigued|做疲惫/打哈欠的表情|加速 5/6 疲惫"
    "neutral|做放松/中性的表情|加速 6/6 中性"
)

echo "======================================================"
echo "P0-3 加速 mismatch 累积 (6 次 × 65s ≈ 7min)"
echo "======================================================"
echo "目标：mismatch 4 → 10 触发 LNN 真自学习首次"
echo ""
echo "操作流程："
echo "  1) 摄像头前坐着别走开"
echo "  2) 跑此脚本，按提示做表情（每次约 5s 准备 + 60s 等待）"
echo "  3) 6 次后 cc 立即跑 scripts/run_fit_step.py 非 force 验证"
echo ""
read -p "准备好了按回车开始..." _

for i in {0..5}; do
    IFS='|' read -r STATE PROMPT NOTE <<< "${STATES[$i]}"
    echo ""
    echo "======================================================"
    echo "[$(($i + 1))/6] true_state=$STATE"
    echo "👉 现在请你 ${PROMPT}（保持 ~5s 让摄像头捕捉）"
    echo "======================================================"
    # 给 5s 让川哥摆好表情 + daemon 多采几帧
    sleep 5
    echo "发送 lnmem-tell..."
    ./scripts/lnmem-tell "$STATE" "$NOTE" || {
        echo "❌ lnmem-tell 失败，可能 daemon 没起来或 rate-limit"
        exit 2
    }
    # 最后一次不等
    if [[ $i -lt 5 ]]; then
        echo "⏳ 等 65s 后下一次..."
        sleep 65
    fi
done

echo ""
echo "======================================================"
echo "✅ 6 次完成。当前反馈统计："
curl -s http://127.0.0.1:9876/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
fb = d.get('feedback', {})
trig = fb.get('trigger_types', {})
print(f'  total feedback observed: {fb.get(\"observed\", 0)}')
print(f'  user_triggered: {fb.get(\"user_triggered_metrics\", {})}')
print(f'  trigger_types: {trig}')
"
echo ""
echo "下一步："
echo "  cd $REPO_ROOT"
echo "  .venv-poc-fresh/bin/python3 scripts/run_fit_step.py    # 非 force，期望自然触发"
