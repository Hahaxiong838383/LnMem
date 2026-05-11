#!/usr/bin/env bash
# LnMem daemon V0.3 一键启动脚本 (M0 W1 P3-1)
#
# 用途：从 worktree 路径用 worktree 内的 venv 启 daemon，消除对 /tmp 的依赖。
#
# 设计要点（参考 daemon V0.2 启动方式）：
# 1. 从 worktree 当前目录启（cwd=worktree root），确保 import poc.* 路径解析正确
# 2. 使用 .venv-poc-fresh/bin/python（worktree 内本地 venv）
# 3. macOS TCC 摄像头授权依赖**当前终端**有授权
#    → 在川哥常用 Terminal.app 跑此脚本即可拿权限
#    → 不推荐 nohup（脱离 Terminal 上下文丢 TCC）
# 4. SIGINT/SIGTERM 由 daemon_main.py 内部 handler 优雅退出
#
# 用法（你常用 Terminal.app）：
#   cd ~/mycc/2-Projects/LnMem-poc
#   ./scripts/start-daemon.sh             # 默认参数
#   ./scripts/start-daemon.sh --device cpu --camera 0
#
# 自动加载：
# - 当前 worktree 的 V0.3 代码（包括 user_triggered channel）
# - 当前 worktree 的 .venv-poc-fresh（如不存在会提示安装步骤）
# - poc/.env（GEMINI_PROXY_KEY / MINIMAX_API_KEY 等 VLM 凭据，daemon 自动 load）
#
# 退出码：
# - 0    正常退出（Ctrl+C / SIGTERM）
# - 2    venv 不存在
# - 3    Python module 不存在
# - 4    端口 9876 已被占用（已有 daemon 在跑）

set -euo pipefail

# 全局错误 trap（响应 codex F3）— 区分前置检查失败（exit 2/3/4/5）与 exec 后失败
trap 'EC=$?; [[ $EC -ne 0 ]] && echo -e "\033[0;31m❌ start-daemon.sh failed (exit $EC) at line $LINENO\033[0m" >&2; exit $EC' ERR

# REPO_ROOT 推导（响应 codex F2）— 用 BASH_SOURCE + realpath 解析 symlink
# realpath -m 允许部分路径不存在；macOS coreutils 默认有 realpath
SCRIPT_SRC="${BASH_SOURCE[0]:-$0}"
if command -v realpath >/dev/null 2>&1; then
    SCRIPT_REAL="$(realpath "$SCRIPT_SRC" 2>/dev/null || echo "$SCRIPT_SRC")"
else
    # macOS 上 realpath 来自 coreutils；如缺则降级用 readlink
    SCRIPT_REAL="$(cd "$(dirname "$SCRIPT_SRC")" && pwd)/$(basename "$SCRIPT_SRC")"
fi
REPO_ROOT="$(cd "$(dirname "$SCRIPT_REAL")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv-poc-fresh/bin/python"
DAEMON_MODULE="poc.src.realtime.daemon_main"

# 启动模式（响应 codex F1）— FULL / DEGRADED 区分让 banner 不误导用户
MODE="FULL"
REQUIRE_TOKEN=0
# 解析 wrapper 自定义 flag（剩余的透传给 daemon）
DAEMON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --require-token)
            REQUIRE_TOKEN=1
            shift
            ;;
        *)
            DAEMON_ARGS+=("$1")
            shift
            ;;
    esac
done

# ANSI 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# ---------- 前置检查 ----------

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo -e "${RED}❌ venv 不存在: $VENV_PYTHON${NC}"
    echo "   安装："
    echo "     cd $REPO_ROOT"
    echo "     python3 -m venv .venv-poc-fresh"
    echo "     .venv-poc-fresh/bin/pip install -r requirements.txt"
    exit 2
fi

# 验证模块存在（PYTHONPATH=REPO_ROOT 让 import 不依赖调用方 cwd）
if ! PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" -c "import poc.src.realtime.daemon_main" 2>/dev/null; then
    echo -e "${RED}❌ Python module 不存在: $DAEMON_MODULE${NC}"
    echo "   检查 cwd: $REPO_ROOT"
    echo "   检查 import: PYTHONPATH=$REPO_ROOT $VENV_PYTHON -c 'import $DAEMON_MODULE'"
    exit 3
fi

# 端口检查（响应 codex F4: TOCTOU 竞态非原子，仅友好提示，应用层 bind 兜底）
# 若需更强保护，可用 flock 覆盖检查+exec 窗口；当前 PoC 阶段不必要
if lsof -i :9876 -t >/dev/null 2>&1; then
    EXISTING_PID=$(lsof -i :9876 -t | head -1)
    echo -e "${YELLOW}⚠️  端口 9876 已被占用（PID $EXISTING_PID）${NC}"
    echo "   如确认要重启："
    echo "     kill -TERM $EXISTING_PID; sleep 3; $0"
    exit 4
fi

# secret_token 检查（响应 codex F1: 缺失时进入 DEGRADED 模式或硬 fail）
TOKEN_PATH="$HOME/.lnmem/secret_token"
if [[ ! -f "$TOKEN_PATH" ]]; then
    if [[ "$REQUIRE_TOKEN" -eq 1 ]]; then
        echo -e "${RED}❌ secret_token 未生成且 --require-token 模式开启: $TOKEN_PATH${NC}"
        echo "   解决："
        echo "     ./scripts/lnmem-init-token.sh"
        exit 5
    fi
    MODE="DEGRADED(no secret_token)"
    echo -e "${YELLOW}⚠️  secret_token 未生成: $TOKEN_PATH${NC}"
    echo "   /trigger_mismatch 端点将禁用（弹窗反馈仍可用，仅主动通道不可用）"
    echo "   生成 token: ./scripts/lnmem-init-token.sh"
    echo "   或要求 token 模式启动: $0 --require-token"
    echo ""
fi

# ---------- 启动 ----------

# 启动 banner（响应 codex F1: 显式标 MODE 防误导）
if [[ "$MODE" == "FULL" ]]; then
    echo -e "${GREEN}🚀 启动 LnMem daemon V0.3 [MODE=FULL]${NC}"
else
    echo -e "${YELLOW}🚀 启动 LnMem daemon V0.3 [MODE=$MODE]${NC}"
fi
echo "   cwd:    $REPO_ROOT"
echo "   python: $VENV_PYTHON"
echo "   module: $DAEMON_MODULE"
echo "   git:    $(cd "$REPO_ROOT" && git log --oneline -1 2>/dev/null || echo '<not a git repo>')"
echo "   args:   ${DAEMON_ARGS[*]:-<none>}"
echo ""
echo -e "${YELLOW}注意：摄像头 TCC 授权依赖当前 Terminal.app 上下文，不要 Ctrl+Z 后台${NC}"
echo ""

cd "$REPO_ROOT"
# 取消 ERR trap 让 daemon_main 的退出码原样回传（响应 codex F3 — exec 后不再 trap）
trap - ERR
# exec 替换 shell 自身，确保 SIGINT 直达 daemon_main
# exec 失败极少见（venv python 已在前置检查中验过），但仍打印明确错误
exec "$VENV_PYTHON" -m "$DAEMON_MODULE" "${DAEMON_ARGS[@]}" || {
    echo -e "${RED}❌ exec failed: python module 启动失败（前置检查过但 runtime 失败）${NC}" >&2
    exit 6
}
