#!/usr/bin/env bash
# LnMem V0.3 secret_token 一键生成 (M0 W2 P0-1)
#
# 生成 32 字节随机 token 写入 ~/.lnmem/secret_token (chmod 600)，
# daemon 启动时加载用于 /trigger_mismatch 端点 X-LnMem-Token Header 校验。
#
# 用法：
#   ./scripts/lnmem-init-token.sh         # 生成（如已存在则提示）
#   ./scripts/lnmem-init-token.sh --force  # 覆盖现有 token

set -euo pipefail

TOKEN_PATH="${HOME}/.lnmem/secret_token"
FORCE="${1:-}"

mkdir -p "$(dirname "$TOKEN_PATH")"

if [[ -f "$TOKEN_PATH" && "$FORCE" != "--force" ]]; then
    echo "❌ token 已存在: $TOKEN_PATH"
    echo "   如要覆盖请加 --force"
    exit 1
fi

# 32 字节随机（256 bit）→ base64url，约 43 字符
TOKEN=$(openssl rand 32 | base64 | tr '+/' '-_' | tr -d '=' | tr -d '\n')

# 原子写
TMP=$(mktemp "${TOKEN_PATH}.XXXXXX")
printf '%s\n' "$TOKEN" > "$TMP"
chmod 600 "$TMP"
mv -f "$TMP" "$TOKEN_PATH"

echo "✅ token 生成成功: $TOKEN_PATH"
echo "   权限: $(stat -f%Lp "$TOKEN_PATH") (期望 600)"
echo "   长度: ${#TOKEN} chars"
echo ""
echo "📌 重启 daemon 让其加载新 token："
echo "   pkill -f poc.src.realtime.daemon_main; 川哥常用终端重启 daemon"
echo ""
echo "📌 使用 lnmem-tell CLI 主动反馈："
echo "   scripts/lnmem-tell focused \"我在写代码\""
