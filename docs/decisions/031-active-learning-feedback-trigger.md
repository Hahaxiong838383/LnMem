# ADR-031: 主动反馈从阈值触发改 active learning 范式

**状态**：accepted
**日期**：2026-05-25
**决策者**：川哥（5/25 12:25 钦定）
**触发**：川哥指示"对于弹窗，我希望不是按照设定阈值，而是 LNN 判断需要我反馈时，弹窗让我反馈并附上理由，为什么这次需要我反馈，我的反馈起到什么作用"

---

## 背景

v1 `notify_watcher.sh` 用单一阈值触发：`intent_confidence ≥ 0.7 AND elapsed ≥ 60s AND state ≠ 上次`。

问题：
- **高置信不等于"需要学习"**：模型对一个状态很确信时反而不需要更多反馈
- **没有理由**：川哥不知道为什么这次问他，反馈的边际价值无感
- **盲目均匀**：不会在 LNN 真正"困惑"或"知识缺失"的时刻优先问

川哥要求：把"模型预测过 X 阈值"反过来，做成"模型自己判断需要学习时主动求反馈"——这是机器学习经典的 **active learning** 范式。

## 决策

v2 用 6 个 active learning 触发器（按优先级，同时段只取最高优先级一个）：

| 触发器 | 信号来源 | 触发条件 |
|--------|---------|---------|
| **F. 冷启动校准** | ckpt_sha8 变化 | ckpt 5 分钟内换过 + 有任意不确定信号 |
| **A. 意图不确定** | intent_probs | top-2 差 < 0.10 持续 ≥ 3 帧 |
| **B. 表情不确定** | phase2 margin/entropy | margin < 0.30 或 entropy > 0.60 持续 ≥ 3 帧 |
| **C. 状态突变** | predicted_state 变化 | 两次都 top_prob ≥ 0.80 但 state 不同 |
| **D. 稀疏类样本** | feedback jsonl 历史 | 此 state 历史 < 5 条 |
| **E. 历史标注分歧** | feedback jsonl 历史 | 同 state 历史 ≥ 10 条 且主类占比在 40-60% |

每次弹窗附 **3 行诚实文案**（codex 对抗审钦定）：
1. **【为什么这次问你】** 触发器描述
2. **【依据】** 触发器命中的关键字段+数值
3. **【你的反馈如何帮我】** 这次反馈对模型的具体作用

## 选项与对比

| 选项 | 优点 | 缺点 |
|------|------|------|
| **v1: 单一阈值** | 实现简单 | 高置信≠需要学；无理由 |
| **v2: 6 触发器 active learning** ⭐ | 触发时机和反馈价值正相关；川哥能看到理由 | 实现复杂；规则多 |
| v3: 数据驱动训练 acquisition function | 理论最优 | 需要 meta-learning，工程量太大 |

## 为什么选 v2（codex 对抗审 + 整改）

codex 5 个对抗审都采纳：

1. **F 缺时间戳问题** → 加 `~/.lnmem/watcher_state/last_ckpt_change_ts` 持久化（修了 watcher 首次启动误把 last_sha 当"刚换"的 bug）
2. **C "高置信"未定义** → 显式 `top_prob ≥ 0.80` 阈值
3. **E "50/50 翻转"小样本噪声** → 加 ≥ 10 条样本量门
4. **D 长尾稀疏可能尖刺** → 加 600s 全局 cooldown
5. **A+B OR 重复触发** → 改优先级单选（codex 钦定不再 OR）
6. **遗漏：信号质量异常** → 加 lnn.error / 字段缺失 SKIP（不弹窗）
7. **遗漏：时间连续性** → A/B 必须持续 ≥ 3 帧才触发

## 防抖机制

- `GLOBAL_COOLDOWN_S=30`：任意两次弹窗最小间隔
- `PER_REASON_COOLDOWN_S=120`：同触发器+同 state 最小间隔
- `DAILY_MAX=150`：日上限保护
- A/B 持续 3 帧防瞬时噪声
- D/E 单独 600s cooldown 防长尾抖动

## 文案诚实性（codex 第 4 点钦定）

**禁止**：心理化拟人表达 "我觉得 / 我感到困惑"
**强制**：可追溯到具体字段 + 数值

示例：
```
🤔 LNN 主动求反馈

【为什么这次问你】意图不确定

【依据】连续 3 帧三类概率差 < 0.10
        intent_top2_diff=0.038 phase2_margin=0.486

【你的反馈如何帮我】你告诉我到底是 positive/negative/none，
                    我下次同情境能学会
```

多触发器并发只展示最高优先级 1 个主因（不拼接"编故事"）。

## 部署

**LaunchAgent**：`com.lnmem.notify-watcher.plist`
- `KeepAlive=true` 崩了自动重启
- `RunAtLoad=true` 装上立刻起
- `ThrottleInterval=60s` 防 crash loop
- env 写死全部阈值，重启行为一致

**状态目录**：`~/.lnmem/watcher_state/`
- `last_ckpt_sha8` / `last_ckpt_change_ts`
- `last_global_ts` / `last_reason_ts` / `last_high_conf_state`
- `persist_counter` (JSON A/B 帧计数)
- `daily_count_YYYY-MM-DD`

**反馈 jsonl note 格式（v2 标识）**：
```
v2|reason=A|phase2=emotional_negative|intent_top2_diff=0.038 phase2_margin=0.486
```

## 端到端验收（5/25 12:31）

- LaunchAgent 装上 PID 27849 + KeepAlive ✅
- 实测触发 F → C，evidence 完整写入 jsonl ✅
- bug 修复：watcher 首次启动不再误报 F 冷启动 ✅
- daily_count 计数正常 ✅

## 后续监测

- 1 周后 retro：每个触发器实际触发次数分布（应该 A+B 占大头，F 偶发，D/E 长尾）
- 川哥主观打分：弹窗时机是否"问得有道理"
- 反馈质量：v1 全 corrected=none 占 79%，v2 应该 corrected 分布更均匀（因为 A/B 触发的真不确定帧川哥会真分类）

---

**关联 ADR**：ADR-028（LNN 角色重定义）/ ADR-029（Sprint 2）/ ADR-030（Studio 8h 训练）
**关联文件**：`scripts/notify_watcher.sh` / `scripts/launchd/com.lnmem.notify-watcher.plist`
