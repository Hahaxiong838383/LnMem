#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT}/.venv-poc-fresh/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"

BASE_V2_CKPT="${ROOT}/poc/data/checkpoints/base_v2/checkpoint_base_v2_best.pt"
BASE_V1_CKPT="${ROOT}/poc/data/checkpoints/day12_with_self/checkpoint_day12.pt"
OUTPUT_CKPT="${ROOT}/poc/data/checkpoints/base_v2/checkpoint_base_v2_1_${TS}.pt"
TRAIN_MD="/tmp/base_v2_1_train_${TS}.md"
REPLAY_MD="/tmp/base_v2_1_replay_${TS}.md"
FINAL_MD="${ROOT}/docs/reports/2026-05-21-base-v2_1-finetune.md"

cd "${ROOT}"

"${PYTHON_BIN}" scripts/finetune_base_v2_1.py \
  --base-v2-ckpt "${BASE_V2_CKPT}" \
  --base-v1-ckpt "${BASE_V1_CKPT}" \
  --unlabeled-signals-glob "~/.lnmem/data/signals_2026-05-*.jsonl" \
  --labeled-queries-glob "~/.lnmem/feedback/queries_*.jsonl" \
  --output-ckpt "${OUTPUT_CKPT}" \
  --output-report "${TRAIN_MD}" \
  --ssl-epochs 5 \
  --sup-epochs 30 \
  --sup-patience 8 \
  --persona-idx 30 \
  --device cpu \
  --seed 42

"${PYTHON_BIN}" scripts/ab_replay_ckpt_v1_v2.py \
  --ckpt-v1 "${BASE_V1_CKPT}" \
  --ckpt-v2 "${BASE_V2_CKPT}" \
  --ckpt-extra "base_v2_1=${OUTPUT_CKPT}" \
  --output-report "${REPLAY_MD}" \
  --queries-glob "~/.lnmem/feedback/queries_2026-05-*.jsonl" \
  --signals-dir "~/.lnmem/data" \
  --default-persona-idx 30 \
  --device cpu

mkdir -p "$(dirname "${FINAL_MD}")"
{
  echo "# 2026-05-21 base_v2.1 finetune"
  echo
  echo "- output_ckpt: \`${OUTPUT_CKPT}\`"
  echo "- generated_at: \`${TS}\`"
  echo
  cat "${TRAIN_MD}"
  echo
  cat "${REPLAY_MD}"
} > "${FINAL_MD}"

echo "checkpoint=${OUTPUT_CKPT}"
echo "report=${FINAL_MD}"
