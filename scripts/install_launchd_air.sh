#!/usr/bin/env bash
# install_launchd_air.sh — Air 端一键装 LnMem LaunchAgent
#
# 装两个：
#   1. com.lnmem.sync-daemon-to-studio   每小时同步 daemon jsonl → Studio
#   2. com.lnmem.pull-ckpt-from-studio   每 10min 拉 Studio 训完的 ckpt 回 Air
#
# 用法：./scripts/install_launchd_air.sh [--uninstall]

set -euo pipefail

REPO_DIR="/Users/rayman.chen/mycc/2-Projects/LnMem-poc"
LAUNCHD_DIR="${HOME}/Library/LaunchAgents"
PLISTS=(
  "com.lnmem.sync-daemon-to-studio"
  "com.lnmem.pull-ckpt-from-studio"
)

UNINSTALL=0
[ "${1:-}" = "--uninstall" ] && UNINSTALL=1

mkdir -p "$LAUNCHD_DIR" "${HOME}/.lnmem/logs/sync" "${HOME}/.lnmem/logs/ckpt_pull"

for label in "${PLISTS[@]}"; do
    src="${REPO_DIR}/scripts/launchd/${label}.plist"
    dst="${LAUNCHD_DIR}/${label}.plist"

    if [ "$UNINSTALL" -eq 1 ]; then
        if [ -f "$dst" ]; then
            launchctl unload "$dst" 2>/dev/null || true
            rm -f "$dst"
            echo "[uninstalled] $label"
        else
            echo "[skip] $label not installed"
        fi
        continue
    fi

    # 先 unload（如果已存在）
    launchctl unload "$dst" 2>/dev/null || true
    # 复制 plist
    cp "$src" "$dst"
    # 装载
    if launchctl load "$dst"; then
        echo "[installed] $label"
    else
        echo "[ERR] launchctl load failed for $label"
        exit 1
    fi
done

echo ""
echo "===== 安装完成 ====="
launchctl list | grep "com.lnmem" || echo "(无 LnMem LaunchAgent 在跑)"
echo ""
echo "查看日志："
echo "  tail -f ~/.lnmem/logs/sync/launchd.out.log"
echo "  tail -f ~/.lnmem/logs/ckpt_pull/launchd.out.log"
echo ""
echo "手动触发一次："
echo "  launchctl start com.lnmem.sync-daemon-to-studio"
echo "  launchctl start com.lnmem.pull-ckpt-from-studio"
echo ""
echo "卸载：./scripts/install_launchd_air.sh --uninstall"
