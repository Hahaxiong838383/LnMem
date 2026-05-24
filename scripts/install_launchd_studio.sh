#!/usr/bin/env bash
# install_launchd_studio.sh — Studio 端一键装 LnMem nightly train LaunchAgent
#
# 装：com.lnmem.studio-nightly-train  每天 03:00 跑 LNN intent classifier 训练
#
# 用法（在 Studio 上）：./scripts/install_launchd_studio.sh [--uninstall]

set -euo pipefail

REPO_DIR="${HOME}/lnmem-poc"
LAUNCHD_DIR="${HOME}/Library/LaunchAgents"
LABEL="com.lnmem.studio-nightly-train"

UNINSTALL=0
[ "${1:-}" = "--uninstall" ] && UNINSTALL=1

mkdir -p "$LAUNCHD_DIR" "${HOME}/lnmem-train/logs/train" "${HOME}/lnmem-train/ckpt"

src="${REPO_DIR}/scripts/launchd/${LABEL}.plist"
dst="${LAUNCHD_DIR}/${LABEL}.plist"

if [ "$UNINSTALL" -eq 1 ]; then
    launchctl unload "$dst" 2>/dev/null || true
    rm -f "$dst"
    echo "[uninstalled] $LABEL"
    exit 0
fi

launchctl unload "$dst" 2>/dev/null || true
cp "$src" "$dst"
launchctl load "$dst"
echo "[installed] $LABEL"
echo ""
echo "===== Studio 夜训已配 ====="
launchctl list | grep "com.lnmem" || echo "(未在 launchctl list 显示，可能 unload 状态)"
echo ""
echo "查看日志："
echo "  tail -f ~/lnmem-train/logs/train/launchd.out.log"
echo ""
echo "手动触发一次（不等 03:00）："
echo "  launchctl start $LABEL"
echo "  或 ~/lnmem-poc/scripts/studio_nightly_train.sh --force"
