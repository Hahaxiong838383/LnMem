---
version: v1.0
audience: 川哥 / cc / codex / 未来接手的 AI 同事
purpose: Phase 2 中立裁判终局裁决（干净数据重跑）+ 模态效果预估 + 项目暂停记录
status: 稳定（裁决已双引擎对抗审签收）
last_updated: 2026-07-03
related_adrs: ADR-039 / ADR-040 / ADR-041
---

# Phase 2 中立裁判终局 + 项目暂停（2026-07-03）

> **一句话裁决**：AU→意图信号真实存在（域内 p=0.0033）但产品级不够用（0.086 bits，占 H(Y) 5.7%）；10s 时序零增量——τ 演化之争在 AU-only 输入下接近伪命题；两发对抗审（grok+codex 均 REVISE）一致：首要行动是接活 context/audio 死模态，AU 降级为辅助信号。**川哥决策：项目暂停，落盘归档，重启入口 = ADR-042（多模态输入面升级）。**

---

## 1. 背景

- 5/29 ADR-041 修复标注-窗口 100% 错位后，川哥 5/30-6/14 攒了 543 条新标注（462 条 anchor_mode=ts_ns 干净 + 81 条 fallback_now 兜底）。
- 6/14 起项目停滞：daemon/notify_watcher 已停，但 fit-step（15min）/sync（1h）/pull-ckpt（10min）三个 LaunchAgent 空转月余（7/3 凌晨仍在产 ckpt）。
- 7/3 cc 复跑 ADR-040 Phase 2 中立裁判（harness 原样，双轮对照），并首次跑 within-session 判别实验。

**Why 重跑**：5/29 裁判用的是全错位废数据，结论作废；干净数据攒够后一直没人重跑，所有下游决策（τ/FiLM/重训）都 gated 在这上面。

## 2. 四轮裁判结果

| 轮 | 数据 | 检验 | 0s 单帧 | 10s 窗口 | 结论 |
|---|---|---|---|---|---|
| 1 | 全量 543 | 跨session GroupKFold+置换200 | p=1.00, F1=0.334 | p=1.00, F1=0.343 | 检验退化（§3.1） |
| 2 | 纯ts_ns 462 | 同上 | p=1.00, F1=0.357 | p=1.00, F1=0.344 | 同上 |
| 3 | 全量最大session n=502 | 域内 StratifiedKFold5+置换300 | **p=0.0033**, I=+0.069 bits, F1=0.383（基线0.234） | p=0.010, F1=0.371 | 域内显著 |
| 4 | ts_ns最大session n=422 | 同上 | **p=0.0033**, I=+0.086 bits, F1=0.388（基线0.236） | p=0.020, F1=0.350 | 域内显著更强 |

- 长程（10-120s）条件增益：两轮均不显著（p=0.756 / p=0.488），n_long<1000 欠功率，判决按脚本 EPV 规则挂起。
- 轮3→轮4 对比：**剔除 81 条 fallback_now 后信号 +23%（0.069→0.086 bits）**——ADR-041 anchor 修复方向再获反向验证。

## 3. 方法学三发现（对抗审确认）

### 3.1 GroupKFold 在新数据上退化
91% 数据集中在单一 ui_session（notify_watcher 常驻进程固定 id 跨 15 天），跨组 CV 某些 fold train 仅 121 条 → CE 爆炸 → I_lower 大负数。**轮1/2 的「p=1.0」是划分伪影，只能解读为「跨渠道泛化评估失败」，不能解读为「无信号」**。

### 3.2 kNN MI 噪声地板
恒 0.5 的死维 dim16 测出 MI=0.039 nats（sklearn `mutual_info_classif` 对加噪恒定特征的正偏伪影）。per-AU MI 绝对值全部不可信；top1 AU[8]=0.067 仅比噪声地板高 72%。

### 3.3 脚本判别文案语义错误
`diag_within_session` 输出「DOMAIN SHIFT（意图 per-person）→ 支持转 FiLM」在单人数据上不成立——跨 session 崩只能解释为跨**渠道/日期**漂移（弹窗 vs 控制台、光照、标注标准漂移）。**FiLM 结论被两发评审一致推翻**（单用户数据撑不起 per-user 调制的论证）。

**Why 记录这三条**：下轮任何人复用这套 harness，必须带着这三个修正读结果。

## 4. 两发对抗审（gemini 链路已下线，grok+codex 两发）

| | grok（xAI grok-4.3） | codex（OpenAI gpt-5.5） |
|---|---|---|
| verdict | REVISE | REVISE |
| AU-only 路线 | 「证伪」（更强硬） | **科学上未证伪 / 产品上到顶，降级辅助模态**（分层更准，采纳） |
| 时序结论 | 现有数据无被掩盖证据 | 成立但限定当前特征+样本量；特征工程弱/窗口对齐/PCA 压稀疏动态都可能掩盖 |
| per-person 文案 | 确认错误 | 确认错误，给出 7 种替代解释 |
| 决策排序 | (a)接模态 > (b)攒长程 > (d)转向 > (c)FiLM | (a)接模态 > (d)AU 降辅助 > (b)长程随新管线自然积累 > (c)FiLM |
| 独有增量 | — | ① within 随机 5-fold 有时间泄露风险，下轮必须 blocked-by-day / rolling-origin；② 采集元数据清单（AI action ts / 点击 ts / active app / 音频能量韵律） |

评审全文：[`2026-07-03-verdict-review-grok.md`](2026-07-03-verdict-review-grok.md) / [`2026-07-03-verdict-review-codex.md`](2026-07-03-verdict-review-codex.md)

## 5. 接活模态的效果预估（probe 实测锚点）

### 5.1 原 spec 4 维远比想象弱（fusion.py 实读）
- context 2 维 = context_known（app 名**非空**布尔——接活后恒 1 零信息）+ focus_stability（切换频率）
- audio 2 维 = voice_present + density（依赖 ASR transcript，桌面场景稀疏）
- **照字面接活 ≠ context 信号的真正潜力**。

### 5.2 代理 probe（用 note 里的触发器 reason + 时刻，n=423 域内同协议）
| 特征组 | I_lower | Macro-F1 | accuracy |
|---|---|---|---|
| AU-only（baseline） | +0.0855 | 0.354 | 0.532 |
| AU + context 代理（7 维） | +0.0715（维度税） | **0.418（+18%）** | 0.537 |
| 全猜 none 基线 | 0 | 0.236 | **0.548** |

- 触发器与标签强关联：none 占比按 reason 从 39% 摆到 75%；控制台渠道 none 仅 10%——「交互模式」几乎决定有无反馈。
- **accuracy 口径残酷事实：当前一切配置跑不赢全猜 none**。v0.1 的「65% 准确率」承诺在 none≈55% 分布下有水分，建议北极星改 **Macro-F1 ≥0.55 + 非none 二分类 F1 ≥0.65（blocked-by-day 口径）**。

### 5.3 分档预估
| 档 | 内容 | Macro-F1（现 0.39） | 判断 |
|---|---|---|---|
| P0 字面 | 原 spec 4 维照接 | 0.41~0.44 | **不值得单独做** |
| P0+ 升级 | active_app 类别化 + AI action 类型/距今 + 键鼠活动 + audio 能量韵律（+8~12 维，动 schema 需 ADR） | **0.50~0.58** | 值得投，none 闸门结构性解决 |
| 上界 | — | 受标注跨天漂移封顶（标注可复现性估 70-80%） | — |

机制：H(Y)≈1.44 bits 分解为「有无反馈」（≈1.0 bit，context 因果上游主攻）×「正 or 负」（AU+audio 主攻）。当前 AU-only 只咬到第二级一小口，第一级全靠猜。

**Why 采纳 P0+ 而非 P0**：probe 证明连 7 维粗代理都有 +18% F1 互补增益，而字面 4 维的信息编码比代理还弱。

## 6. 项目暂停执行记录（2026-07-03）

- `launchctl bootout` 停止 3 个运行中服务：fit-step / pull-ckpt-from-studio / sync-daemon-to-studio。
- **6 个部署 plist 全部从 `~/Library/LaunchAgents/` 移除**（含已停的 supervisor/watchdog/notify-watcher——四个 RunAtLoad=true，只停运行中的 3 个则下次登录 daemon 全家桶自动复活）。
- 移除前保真快照至 [`scripts/launchd-deployed-snapshot-20260703/`](../../scripts/launchd-deployed-snapshot-20260703/)（与 repo 模板 diff 仅为占位符实例化，无手改逻辑）。
- 三重验证通过：launchctl 无 lnmem / LaunchAgents 无 plist / 无相关进程。
- ⚠️ **Studio 侧 `com.lnmem.studio-nightly-train`（8h 调度）未停**——在 Studio 机器上，本次未操作（川哥指令范围为本机 3 服务）。数据侧 increment=0 不会真训练，但 launchd 仍在调度，重启项目前或彻底封存时应处理。
- 数据资产冻结：543 条标注（462 干净）在 `~/.lnmem/data/intent_feedback/`；最后 ckpt `intent_head_20260703-000006.pt`。

## 7. 重启入口（ADR-042 草案要点）

1. 恢复 LaunchAgents（从快照 cp 回 + bootstrap）
2. 立 ADR-042「多模态输入面升级」：P0+ 特征清单 / EventTensor schema 变更（封版件，需川哥签字）/ 评估协议 blocked-by-day / 北极星指标修订
3. 接活 context（active_app 类别 + AI action 事件桥）+ audio（RMS 能量+韵律，不走 ASR）
4. 采集 2-3 周（目标 800-1000 条）→ 重跑本套裁判（带 §3 三修正）
5. τ 演化/长程判决/FiLM 维持挂起

---

**产出**：cc 主线（裁判复跑 4 轮 + probe 3 组 + 死模态/口径验证）；对抗审 grok + codex；川哥决策暂停。
