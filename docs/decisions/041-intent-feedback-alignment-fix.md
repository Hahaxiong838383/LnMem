# ADR-041: intent_feedback 标注-窗口对齐管线修复 + 模态半死审计

**状态**:accepted
**日期**:2026-05-29
**决策者**:川哥(钦定先确认机制再修)/ cc(诊断 + 编排 + 验收)/ gemini-3.5-high(两轮对抗审,推翻 v1、确认审拦截 v2 过度外推)/ codex-gpt-5.3-codex(派工修 patch)
**关联**:[`ADR-039`](039-deterministic-retrain-proves-data-bottleneck.md)(数据瓶颈)、[`ADR-040`](040-action3-tau-evolution-gated-horizon-diagnosis.md)(动作3门控 + 中立裁判 Phase 2a/2c)

---

## 背景

ADR-040 把 τ 演化设为 gated 任务,要求先用信息论中立裁判(I(X;Y))判定"intent 任务到底需不需要 >10s 历史",绕开 lag ablation 被推翻的循环论证。本 ADR 是 Phase 2 中立裁判跑完后的**意外结论 + 修复决策**。

## Phase 2 中立裁判结果(三重门控 + 两轮对抗审)

直接算原始 AU 特征 ↔ `corrected` 真标签互信息(不碰短 τ 模型,zero-model):

| 验证层 | 结果 |
|---|---|
| v1 随机 KFold | 10s "p=0.005" → 被 gemini 推翻为 session-block 泄露伪影 |
| v2 GroupKFold by ui_session | 0s p=0.353 / 10s p=0.214,Macro-F1 ≤ 平凡基线 |
| 域内 CV(gemini 钦定判别器) | 单 session 内 p=0.70/0.73,Macro-F1 ≤ 基线 |
| per-AU 模型无关 MI | 最高 AU[8]=0.0345 nats(~3% of H(Y)) |

**结论**:三重门控下,AU 特征对 `corrected` 标签**无信号**——任何时程、跨 session 或域内都无。

## 真根因(两条数据管线 bug,完整可验证)

### 根因 A · 标注-窗口时间错位 100%(致命)

| 测量 | 结果 |
|---|---|
| 481 条样本中,事件 `ts_ns` 落在其 `au_window` 时间范围内 | **0 / 481** |
| 事件 → 存储窗口 时间差 | median **305 秒**(p10=302,p90=325) |

**机制链**(每环代码行号锚定):
1. 控制台 [`streamlit_ui/intent_console.py:38`](../../streamlit_ui/intent_console.py) `SIGNALS_URL = "...?limit=1"` —— 传 `limit` 参数
2. daemon [`poc/src/realtime/daemon_main.py:746-748`](../../poc/src/realtime/daemon_main.py) handler 只认 `seconds`(默认 300),**完全忽略 limit** → 永远返回 ~5min 列表
3. 控制台 [`streamlit_ui/intent_console.py:119`](../../streamlit_ui/intent_console.py) `rec = data[0]` 取列表第一条
4. 实测 `/signals/recent` 返回 oldest→newest 排序,跨度 ~3 分钟,`data[0]` = **最旧那条(~5min 前)**
5. 该 `data[0].ts` 被当作 `ts_ns`([intent_console.py:137-138](../../streamlit_ui/intent_console.py))
6. POST 到 [`intent_feedback_server.py:104`](../../poc/src/realtime/intent_feedback_server.py) `_handle_intent_feedback` → `_collect_temporal_window(seconds=10)` 取 inferencer 当前缓冲最近 10s(锚定**POST 时刻**)
7. 存盘:`ts_ns`(5min前事件) + `au_window`(现在最近10s) → **从构造上不可能对齐**

模型被训练去学"川哥现在的脸 → 5 分钟前某事的标签"。这从源头让任何下游算法/τ/历史长度的优化都白搭。

### 根因 B · 多模态融合 4/8 维死(次要)

`fusion.py:217` 18 维布局 `face4 + derived4 + expression6 + audio2 + context2`,`missing_default=0.5`。实测 4788 帧:

| dim | 身份 | 状态 |
|---|---|---|
| 14, 15 | voice_present / utterance_density | **AUDIO 整类未接**,恒 0.5 |
| 16, 17 | context_known / focus_stability | **CONTEXT 整类未接**,恒 0.5 |
| 3 | gaze_stability | 检测器不提供 |

仅 13/18 维活,且 audio + context 这两类(很可能是意图最强的信号源)整类断开。

## 决策:Z 方案(双层修复 + 历史数据归档)

### 上游 · Y(streamlit 控制台,修源头)
- `streamlit_ui/intent_console.py:38` `SIGNALS_URL = "...?seconds=1"` 替代 `?limit=1`(daemon 不认 limit)
- `streamlit_ui/intent_console.py:119` `rec = data[0]` → `rec = data[-1]`(取最新)

### 下游 · X(server 端兜底防御,锚定 ts_ns)
- `poc/src/realtime/lnn_inferencer.py` 新增 `get_window_around_ts(ts_ns: int, recent_s: int = 10) -> list[dict]`:从 `_recent_temporal` 切出**以 ts_ns 为右端**的 recent_s 秒窗口(`[ts_ns - recent_s*1e9, ts_ns]`);若 ts_ns 超出缓冲范围(过早/过晚 > tolerance),fail-soft 返回 [] 让 server 降级。同样新增 `get_long_context_around_ts(ts_ns, recent_s, long_s, long_hz)`,把长程窗也锚定到 ts_ns 而非"当下"
- `poc/src/realtime/intent_feedback_server.py:104,107` `_collect_temporal_window` / `_collect_long_context` 接受可选 `anchor_ts_ns` 参数,优先按 ts_ns 切窗;缺失/超出降级回当前最近 10s(并在 record 元数据加 `anchor_mode: "ts_ns" | "fallback_now"` 标记,方便后续审计)

### 历史数据归档
- 481 条旧数据已 `mv ~/.lnmem/data/intent_feedback → intent_feedback.pre_alignment_fix_20260529`(保留不删,作根因证据)
- 新空目录 `~/.lnmem/data/intent_feedback/` 等修复完攒新数据
- 不计入未来训练/裁判直至重攒 ≥200 条对齐样本

### 模态半死(根因 B)处理
- 本 ADR **不修复**(超出当前 PR 范围,且需要产品决策"audio/context 真接还是划出 MVP")
- 单独立 issue 由川哥决定:
  - 选项 a:接入 audio + context 采集
  - 选项 b:明确"v0.3 仅做纯脸",从 fusion 移除两类死维,降到 14 维
- 当前修复完先看"对齐后单纯靠脸+表情有没有信号",再决定 B 怎么走

## 验收门(双门 + 端到端冒烟)

### Gate 1 · Verify(目标达成)
- 单测覆盖 `get_window_around_ts`:命中正常 ts_ns / ts_ns 在缓冲范围外(早) / ts_ns 在缓冲范围外(晚) / 容差边界 / 空缓冲 fail-soft → 5 个用例
- 单测覆盖 `_collect_temporal_window(anchor_ts_ns=...)` 与 `_collect_long_context(anchor_ts_ns=...)`:命中/降级路径
- 端到端:启 daemon + 控制台,新点 3-5 条标注,验证 `event ts_ns ∈ [au_window 首帧 ts, au_window 末帧 ts]`,落入率应 ~100%

### Gate 2 · Guard(无回归)
- `pytest poc/` 全跑无失败
- 现有 281 单测保持通过
- daemon 现有 endpoint(`/signals/recent` / `/predict_future` / `/health`)行为不变(只新增可选 anchor,不破坏现有调用)

### codex 对抗审(CLAUDE.md §3.1 PR 三道门)
- patch 完成后,派 codex 做 5 个最容易翻车的点审查
- 至少覆盖:ts_ns 越界处理 / 容差选取 / 降级标记 / 时间字段类型(int vs ns vs ISO) / 并发安全(_recent_temporal 是 deque 无锁)

## 关键教训(沉淀)

1. **中立裁判优先于模型**:11 轮 lag ablation 测不出 τ 之争,中立 MI 一次性绕开循环论证、暴露真根因。详见 [[feedback_neutral_judge_groupkfold_discriminator]]。
2. **数据管线 bug 会伪装成算法瓶颈**:从 2026-05 起的所有"τ 训不动 / responses=0 / 数据瓶颈"现象,根因都是这条对齐 bug + audio/context 死维,跟 LTC 内核无关。
3. **cc 单轮自评仍系统乐观**:v1 修完估计器 bug 后,cc 一度判"10s 有信号"——被 gemini 推翻为泄露;v2 后又想直接转 FiLM——被 gemini 拦下,逼出域内判别才钉死。重要结论必双引擎对抗审。
4. **对 gemini 也要 no-guessing**:gemini 第一轮说"窗口重叠 90%",cc 实测相邻标注 median 间隔 204s 只 4.2% 对 <10s,推翻这个具体前提。但更本质的"session-block 非独立性"成立,采纳。剔除幻觉、采纳真洞察、各 1/2 比例。

## 回退方案

- `git tag before-alignment-fix-20260529` 锁定修复前状态
- 旧数据 `intent_feedback.pre_alignment_fix_20260529` 保留,可一键 `mv` 恢复
- daemon 不重启,只控制台 + server 改完即生效;若新行为出问题,`git revert` + 重启 daemon 即可

---

**本 ADR 版本:v1.0 · 2026-05-29**
**诊断脚本(保留供未来回归测试)**:
- [`scripts/diag_mutual_information.py`](../../scripts/diag_mutual_information.py)(v2 GroupKFold + PCA 同维)
- [`scripts/diag_within_session.py`](../../scripts/diag_within_session.py)(域内判别)
- [`scripts/diag_horizon_ablation.py`](../../scripts/diag_horizon_ablation.py)(ADR-040 历史保留)

**两轮 gemini 对抗审产物**:`/tmp/gemini_mi_review.md`(v1 推翻)、`/tmp/gemini_mi_review2.md`(v2 确认审)
