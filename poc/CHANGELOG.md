# PoC CHANGELOG

> PoC 内部变更日志 — 每日 retro / 重大决策 / 阶段验收都记入此文件。
> 主分支 LnMem 的整体变更不在此处记录。

---

## v0.1 — 2026-05-05（Day 0）

**起草**

- 创建 `poc/lnn-validation` 长期分支
- 起草 PoC 规范 `docs/poc-spec.md` v0.1
- 起草虚拟主人画像 brief `docs/data-generation-todo.md` v0.1
- 起草 `docs/limitations.md` v0.1 骨架
- 起草 `docs/migration-guide.md` v0.1 骨架
- 起草 `docs/lnn-training-log.md` v0.1 schema
- 创建目录结构：data/{personas,synthetic,real} + src/ + demos/
- 归因 grep 完成（CLAUDE.md §2.1 / product-definition.md §10.1 §1.3 §13 / cognitive.md / development-harness.md）
- **归因纠正**：原 prompt 引用 "CLAUDE.md §11.2 PR 三道门" → 实际位置 §2.1（line 35）

**Day 0 任务分配**：
- 老板：30 画像（按 brief 的 schema + 3 个范例为起点）
- cc：上述 5 份文档 + `src/validate_personas.py` 校验脚本

**未完成（待 Day 0 末或 Day 1 启动时补）**：
- `src/validate_personas.py` 校验脚本
- `requirements-poc.txt` 依赖清单
- 多模态信号采集模块骨架（Day 1 启动）

---

## v0.2 — 2026-05-05（Day 0 — 老板提供 3 个范例后升级 schema）

**触发**：老板基于 HEXACO 七维度 + 依恋类型大型样本分布数据 + Schwartz 价值观真实分布，提供了 3 个完整范例画像（陈墨 / 林嘉怡 / 苏野），覆盖 22% / 60% / 17.5% 三大依恋人群。

**升级**：

- 画像 schema 从「demographics + traits 简化版」升级为「七层结构」：
  - L1 HEXACO 6 维 + 关键 facets
  - L2 依恋（焦虑 + 回避两维 + 类型 + 人群占比）
  - L3 动机（4 个 need + Schwartz top3/bottom）
  - L4 if-then 情境签名（≥ 5 情境，含行为 / 情绪 / 多模态三段）
  - L5 身份与情境 + 当前生活章节 + 核心张力
  - L6 多模态特征签名（视觉 / 疲劳 / 压力 / 专注 + 语音 + 行为）
  - L7 AI 交互产品设计（信任建立路径 + 沟通风格 + 避免话题）
- 文件格式从 `.json` 改为 `.md`（YAML frontmatter 机读 + body 自然书写）
- 评估标准从「3 个 traits 显著不同」升级为「HEXACO 6 维余弦距离 ≥ 0.4」+ 7 条评估门
- 真实分布锚点：依恋安全型 ≤ 65% / 含 ≥ 2 回避 + ≥ 1 焦虑 + ≥ 1 恐惧-回避

**新增 3 个画像本体**：

- [`vboss_001.md`](data/personas/vboss_001.md) 陈墨 · 内向高自律的技术管理者（HEXACO 高 C/O/H 低 E，回避型，22% 人群）
- [`vboss_002.md`](data/personas/vboss_002.md) 林嘉怡 · 高能量高同理心的产品负责人（高 E/A，偏安全略焦虑，60% 人群）
- [`vboss_003.md`](data/personas/vboss_003.md) 苏野 · 高敏感的独立创作者（极高 O 高 H 高 N 极低 E，恐惧-回避型，17.5% 人群）

**老板已点出的待补缺口**：极度焦虑型（5-19%） / 戏剧性高 E 高 N（~5%） / 老年画像（55+，可暂缓）

**变更文件**：
- M `poc/docs/data-generation-todo.md` v0.1 → v0.2（schema 七层升级）
- A `poc/data/personas/vboss_001.md`
- A `poc/data/personas/vboss_002.md`
- A `poc/data/personas/vboss_003.md`

---

<!-- Day 1 / 2 / 3 ... 每日 retro 在此追加 -->
