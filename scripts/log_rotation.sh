#!/usr/bin/env bash
# LnMem log rotation helper (called by daemon_watchdog.sh)
#
# 用法：log_rotation.sh <jsonl_path>
# 行为：把目标文件 archive 到同目录 archived/<filename>.<utc_ts>.gz，新建空文件
#
# 注：archive 不删除原文件，仅 mv + gzip + 创建新空文件，保证 daemon 写入不
# 中断（rename 是原子操作 + 新建空文件后 daemon 下次 open 走新文件）。

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <jsonl_path>" >&2
    exit 1
fi

src="$1"
if [[ ! -f "$src" ]]; then
    echo "ERROR: $src not found" >&2
    exit 1
fi

dir="$(dirname "$src")"
basename="$(basename "$src")"
archive_dir="$dir/archived"
mkdir -p "$archive_dir"

ts=$(date -u '+%Y%m%dT%H%M%SZ')
archived_path="$archive_dir/${basename}.${ts}"

mv "$src" "$archived_path"
touch "$src"
gzip "$archived_path" &

echo "rotated: $src -> ${archived_path}.gz (background gzip)"
