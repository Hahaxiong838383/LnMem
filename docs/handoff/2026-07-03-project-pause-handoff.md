---
title: 7/3 项目暂停交接 · Phase 2 裁判终局 + 服务全停 + 重启 checklist
date: 2026-07-03
status: paused
previous_session: 5/29 ADR-038/039/040 + Phase1 (handoff 2026-05-29-adr038-040-phase1-handoff.md)
related_adrs: ADR-039 / ADR-040 / ADR-041
---

# 2026-07-03 项目暂停交接

## 一句话

Phase 2 中立裁判用 462 条干净数据重跑完毕（grok+codex 两发对抗审签收）：**AU-only 产品级到顶、τ 非当前瓶颈、首要方向是接活 context/audio**；川哥决策项目暂停，本机服务全停、产物落盘，重启入口 = ADR-042（多模态输入面升级）。

## 当前状态（暂停时刻快照）

| 项 | 状态 |
|---|---|
| 本机 LaunchAgents | **全停全移除**（6 个 plist 快照在 `scripts/launchd-deployed-snapshot-20260703/`） |
| daemon / notify_watcher / fit-step / sync / pull-ckpt | 无进程、无调度 |
| ⚠️ Studio 侧 nightly-train | **未停**（8h 调度还在 Studio 机器 launchd 里；increment=0 不会真训练，重启或彻底封存时处理） |
| 数据资产 | 543 条标注冻结在 `~/.lnmem/data/intent_feedback/`（462 条 ts_ns 干净，460 条带长程窗） |
| 最后 ckpt | `~/.lnmem/ckpt/intent_head_20260703-000006.pt` |
| 裁判 harness | `scripts/diag_mutual_information.py` / `diag_within_session.py` 可复用（复用时必读终局报告 §3 三修正） |

## 裁决要点（详见 [终局报告](../reports/2026-07-03-phase2-verdict-final-and-pause.md)）

1. 域内信号显著但极弱：0s 单帧 I=+0.086 bits（p=0.0033），F1 0.39 vs 基线 0.24
2. 10s 时序零增量 → LTC τ 不是当前瓶颈，τ 演化之争在 AU-only 下接近伪命题
3. 长程判决欠功率挂起（需 1000+ 长程标注）
4. 「转 FiLM」被两发评审推翻（单人数据撑不起 per-user 论证）
5. probe 实证：粗 context 代理即有 F1 +18% 互补增益；预估 P0+ 升级档 F1 0.39→0.50~0.58
6. accuracy 口径当前跑不赢全猜 none（0.548），北极星指标建议改 Macro-F1

## 重启 checklist（按序）

1. **恢复服务**：`cp scripts/launchd-deployed-snapshot-20260703/*.plist ~/Library/LaunchAgents/` + 逐个 `launchctl bootstrap gui/$UID/ ~/Library/LaunchAgents/<name>.plist`
2. **立 ADR-042 多模态输入面升级**（需川哥签字，动 EventTensor schema 封版件）：
   - P0+ 特征：active_app 类别化 / AI action 类型+距今时长 / 键鼠活动 / audio RMS 能量+韵律（不走 ASR）
   - 采集元数据补齐（codex 清单）：AI action timestamp / 用户点击 timestamp / 日期与光照 session 元信息
   - 评估协议：blocked-by-day / rolling-origin 替代随机 5-fold（堵时间泄露）
   - 北极星：Macro-F1 ≥0.55 + 非none 二分类 F1 ≥0.65
3. **接模态 + 重启采集**（目标 800-1000 条，2-3 周）
4. **重跑裁判**（带终局报告 §3 三修正读结果）
5. τ 演化 / 长程判决 / FiLM 维持挂起，gate 在新裁判结果

## 下个 session 接手 prompt（可粘贴）

```
恢复 LnMem 项目（2026-07-03 暂停）。

先读两份：
1. docs/handoff/2026-07-03-project-pause-handoff.md（本交接）
2. docs/reports/2026-07-03-phase2-verdict-final-and-pause.md（裁判终局，重点 §3 方法学修正 + §5 预估 + §7 重启入口）

关键结论（已双引擎对抗审签收，不要重新争论）：
- AU-only 产品级到顶（0.086 bits），降级辅助模态；τ 非当前瓶颈；FiLM 论证已被推翻
- 首要方向 = ADR-042 多模态输入面升级（context 因果上游主攻 none 闸门 + audio 辅攻 pos/neg）

第一件事：按本交接「重启 checklist」§1 恢复服务，然后起草 ADR-042 给川哥签字。
环境备忘：venv = PYTHONPATH=. .venv-poc-fresh/bin/python；数据在 ~/.lnmem/data/intent_feedback/；
gemini 链路已取消，对抗审用 team 两发（grok+codex，review-3send.sh 自动两发）。
⚠️ Studio 侧 nightly-train 未停，重启采集前先确认其状态（ssh mini → ssh fu-121@192.168.100.2）。
```

**Why 写这份交接**：项目暂停是计划内 HANDOFF 场景——不写，一个月后重启时「为什么停 / 停在哪 / 从哪继续」要重新考古一遍。
