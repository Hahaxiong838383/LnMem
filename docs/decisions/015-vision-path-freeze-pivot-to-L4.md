---
adr-id: 015
title: 视觉感知路径阶段性冻结 + 战略转向 L4 自传记忆主线（W+ 方案）
status: accepted（2026-05-21 12:05，川哥钦定 + cc + gemini 3.5-flash 战略审查共识）
date: 2026-05-21
deciders: 川哥 + cc (互校 gemini) + gemini 3.5-flash (战略审查)
supersedes: ADR-013 v2.1 微调路径不再继续 Step 3/4
related: ADR-007 / ADR-012 / ADR-013 / ADR-014 / docs/product-definition.md §10.1
---

## 上下文

### Day 1-2 投入复盘（5/9-5/21 共 13 天）

```
5/9-5/16  daemon 真实运行 + 428 反馈累积
5/16     spike LNN realtime 在川哥真人 0% acc → ADR-007 立项
5/17     ADR-007 Phase 1 plan + gemini 调研
5/19-20  Phase 1.1-1.4：data + spike + sweep + train base v2（VEATIC 12400帧）
5/20晚  base v2 ckpt 落盘（val_acc 0.48）
5/21早  ADR-013 实测 base v2 川哥真人数据 64/64 全坍缩 neutral
5/21中  ADR-013 v2.1 DFAT 微调（gemini 85% 置信度方案）→ 部分改善 distracted 15.91% 但 collapse 仍 WARN
5/21正午 gemini 3.5-flash 战略审查 + cc 互校 → W+ 方案
```

### A/B 实测三模型（66 真实 mismatch 样本）

| model | mismatch | focused | distracted | mean_russell | mean_entropy | top1_neutral | 状态 |
|-------|----------|---------|------------|--------------|--------------|--------------|------|
| base_v1 | 100% | 0% | 0% | 2.32 | 0.63 | 0 | 合成数据训不识真人 |
| base_v2 | 100% | 0% | 0% | 1.09 | 1.68 | 66/66 | 🔴 全坍缩 |
| v2.1 (DFAT) | 89.4% | 0% | 15.91% | 1.07 | 1.72 | 66/66 | 🟡 fallback 改善但坍缩未解决 |

## Gemini 战略审查（颠覆性结论）

> **"cc 主线已陷入算法完美主义陷阱 + 认知逃避。用 10 天去解决一个对 MVP 而言 P3 级问题（视觉状态分类），导致 P0 级产品承诺（L4 记忆 / CLS / 双端 / 记忆 UI）完全停滞。"**

### 关键洞察
1. 视觉感知在 LnMem 5 项产品承诺中权重 **2/10**（极低）
   - "上周三和谁吃饭" 0% 依赖视觉
   - "它会让你看见" 0% 依赖视觉
2. 67 events / 12 天 → LNN 在如此稀疏数据上**只能过拟合或坍缩**（实测）
3. 真正方案是改 **UX**：Cmd+Shift+F 一键状态锚点，标注密度暴涨

### Gemini 4 方案对比

| 方案 | 推荐度 | 理由 |
|------|--------|------|
| Z (mask neutral) | P2 | 自欺欺人 |
| Y (warm-start v1 readout) | P1 | 唯一值得 ML 路径 |
| X (base v3 from scratch DANN) | ❌ 战略自杀 | 4-6h 大概率变 1-2 天 |
| **W (暂停视觉直奔 L4)** | **🏆 P0** | 唯一战略推荐 |

## cc 主线互校（5% 修正）

### 同意 gemini 95%
- 视觉 MVP 阶段权重低 ✅
- 沉没成本陷阱 ✅
- UX 主动锚点是天才级建议 ✅
- W 方向正确 ✅
- "选 dopamine 不是承诺" 直击痛点 — cc 被 humbled ✅

### 修正 gemini 5%

1. **视觉感知长期权重被低估**：
   - MVP 不阻塞（W 方案对）
   - M3+ per-user 个性化栈依赖（必重启）
   - cc 修订：阶段性冻结，不是永久放弃

2. **「键盘+前台 App」不是视觉替代**：
   - gemini 说"比 webcam 准/便宜/稳定"
   - cc 反驳：前台 App 只能区分**工作类型**，无法区分**情绪**
   - LnMem 差异化是情绪轨迹（vs 普通时间追踪工具）
   - cc 修订：UX 主动锚点是**新增 ground truth**，不是替代信号；视觉模块继续以 base v1 + ADR-010/011 guard 跑

3. **Z 方案不值得做（30min 都不值）**：
   - Z 是伪工程，输出 fallback 不是真识别
   - daemon 真跑时强制非 neutral 制造噪声
   - cc 直接跳过 Z

4. **base v2.1 ckpt 不删**：
   - 留作 M3+ 数据增多时的 starting point
   - finetune_base_v2_1.py 脚本保留 repo

## 决策（W+ 方案）

### 立即执行（今天 12:00-17:30）

```
12:05  写本 ADR-015 + commit v2.1 Step1 成果（含 ckpt 落盘但不上线）
12:30-14:00  L4 砖 1：端到端 demo 跑通
             - parser → graphiti 自动写入 (代码已 ready)
             - Streamlit 加实时 KG 节点 table 显示
             - 5 句中文 demo + 浏览器看到完整闭环
14:00-16:00  L4 砖 2：CLS v0 最小可行
             - cron + LLM prompt 提炼当天 KG → 周/日总结
             - 写入 SQLite 长期存储（不引 LNN 复杂度，留 M1）
16:00-17:30  L4 砖 3：记忆 UI v0.1
             - Streamlit 加 KG 卡片【删除】【修改】按钮
             - 同步 Graphiti adapter.invalidate()
             - 川哥能看到/能改/能删（产品承诺第 5 项达成）
17:30  川哥浏览器看完整 L4 闭环
```

### 本周（5/21-5/26）

- M0 验收硬门：麦克风 → ASR → Parser → KG → 30min 后查询返回 < 5s
- 双端架构设计图（居家+车载共享 KG/τ-bias）
- UX 主动状态锚点 Cmd+Shift+F 实施（gemini bonus 建议）

### 视觉路径冻结条款

| 条款 | 内容 |
|------|------|
| daemon 当前状态 | 继续跑 base v1 + ADR-010/011 guard（已稳定 1 周）|
| base v2.1 ckpt | 保留 repo `poc/data/checkpoints/base_v2/`，不删不上线 |
| finetune 脚本 | 保留 `scripts/finetune_base_v2_1.py` + `ab_replay_ckpt_v1_v2.py` |
| 重启条件 | M3+ per-user 个性化栈启动时 + 川哥真实 labeled events ≥ 500 |
| 长期路径 | DANN 域对抗 + VEATIC + 川哥真人混合（base v3）|

## 关键风险提示（gemini 钦定 + cc 加强）

1. **研发陷阱**（gemini 最重要警告）：
   > "用'好解决的算法挑战'逃避'复杂的系统集成'。不要因为哪个好玩就做哪个，要因为哪个是承诺就做哪个。"

   cc 主线必须**警惕自己**对 ML 调参的多巴胺偏好，强迫执行 L4 系统集成（哪怕枯燥）。

2. **时间窗口风险**（修正版）：
   - M4 = 2026-09 月底（4 个月，非 3 个月）
   - 没 L4 闭环 = MVP 失败 = 项目失去 AI Companion 赛道立足点

3. **数据 UX 改设计**：
   - gemini 建议 Cmd+Shift+F 主动锚点
   - cc 加：daemon HTTP `/feedback/state` POST 端点接收（3 行代码）
   - 标注密度从 5-6/天 → 估 20-30/天（4-5x）

## 沉默成本不浪费

base v2.1 失败的 13 天投入沉淀为：
- ADR-007 Phase 1.2 公共底座 `extract_au_to_parquet.py`（M3+ 重启时还用）
- ADR-012 V-A→state Russell 9 state 设计（可复用）
- ADR-011 假脸 guard（daemon 一直在用）
- ADR-013 失败教训（避免 base v3 重蹈覆辙）
- gemini DFAT 评审 + 实测验证（M3+ 时改进基础）

## Summary

视觉感知路径阶段性冻结，全力转 L4 自传记忆主线（W+ 方案）。
ADR-007 v2.1 fallback 改善但 collapse 未解决（66/66 top-1 neutral），数据稀缺 + 跨域是根因。
今天下午 12:30-17:30 cc 主线打通 L4 三砖（端到端 demo + CLS v0 + 记忆 UI v0.1），川哥 17:30 浏览器看完整闭环。
base v2.1 ckpt + finetune 脚本保留，M3+ 重启视觉路径。

---

**模板版本：v0.1 · 2026-05-03**
