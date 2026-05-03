# ADR 002 — M0 骨架对抗审查 findings（gemini-3-flash 替代 codex）

**日期**：2026-05-03
**状态**：已记录，整改 deferred 到 M0 W2 实现阶段
**审查者**：gemini-3-flash-preview（codex 配额到期/gemini-3-pro 429 双重备路）
**审查范围**：commits `16673c6` (M0 contracts + 4-way skeleton) + `4300f89` (lnn-math.md)

## 背景

[`CLAUDE.md`](../../CLAUDE.md) §3 三道门要求每份代码必经 codex 对抗审查。本次 session 收尾时 codex 当日配额满（恢复 2026-05-04 01:33）+ gemini-3-pro-preview 429 限流。改用 gemini-3-flash-preview 替代审查跑通，得到 5 个 findings。

flash 比 pro 弱一档，但本次审查任务是结构化扫描（5 翻车点 + 严重度），flash 表现尚可，作为"明天 codex 复跑前的临时记录"。

## 5 个 findings（gemini-3-flash 输出 + cc 整理）

### F1 · SlotValue.value: Any 语义真空（严重度 H）

**证据**：[`contracts/event_tensor.py`](../../contracts/event_tensor.py) `SlotValue.value: Any`

**问题**：4 路 codex 各自往 `value` 里塞不同类型（A 可能塞 dict，B 塞 str，C 塞 list[int]）。运行期合并时类型不兼容，schema_v1 槽位级语义随 backend 漂移。

**修复方向**（M0 W2）：
- 按 SlotMeta.value_type 收紧（entity → str / span → tuple[str,int,int] / scalar → float / link → str(event_id) / categorical → str-enum / enum → str）
- 在 SlotValue.__post_init__ 加运行时类型校验（schema 驱动）

### F2 · LNN τ 偏置缺正性约束 → 状态爆炸（严重度 H）

**证据**：[`docs/lnn-math.md`](../lnn-math.md) §4.1 公式 `1/(τ + τ_offset[u])`

**问题**：若 `τ_offset[u]` 学到负值且大于 `τ_base`，分母变零或负 → 衰减率发散 → ODE 状态爆炸。文档 §4.1 没明说"τ + τ_offset > 0 是硬约束"。

**修复方向**：
- spike Day 2-3 训练用 `softplus(τ_offset)` 或 `τ + ε + softplus(...)` 强制正性
- 文档 §4.1 加一条"参数化保正"注解
- 触发 spike 黄/红区时优先怀疑这条

### F3 · Protocol Gate 验证只看名字不看签名（严重度 M）

**证据**：[`/tmp/lnmem-codex/cc_verify.py:32`](#) `proto_methods.issubset(set(dir(cand)))`

**问题**：cc 契约对齐 Gate 用 `dir()` 比对方法名集合，不查参数签名。codex 实现可能 `parse(text: str)` 而 Protocol 是 `parse(req: ParseRequest)` — Gate 仍判通过，运行期 TypeError。

**修复方向**：
- cc_verify 加 `inspect.signature` 比对参数名 + 类型注解
- 或直接用 `typing.runtime_checkable + isinstance(impl, Proto)` —— 但 Protocol 默认不查签名细节，需要加 mypy/pyright 配合静态检查

### F4 · LLM 配置碎片化（严重度 M）

**证据**：[`synthetic_data_generator/config.py`](../../synthetic_data_generator/config.py) 和 [`universal_grammar_parser/config.py`](../../universal_grammar_parser/config.py) 两个 LLM config 不一致

**问题**：
- codex A 的 `SyntheticGeneratorConfig` 用 `fetch_impl` 注入 LLM client
- codex B 的 `ParserConfig` 用 `model_name + temperature + max_retries` 直配
- M1 集成时这两个 module 共用 LLM 网关会扯不清楚

**修复方向**（M0 W3）：
- 在 contracts/ 加 `LLMClientProtocol`（chat / embed / count_tokens 三方法）
- 两个模块都改成接受 `LLMClientProtocol` 实例，不各自管配置

### F5 · pytest.skip 占位 "false green" 风险（严重度 M）

**证据**：4 模块 tests/ 全部用 `pytest.skip("M0 W2 implementation pending: ...")`

**问题**：M0 W2 真填实现时，开发者忘记取消 skip → 测试看似 GREEN 实际从未跑过 → 假阳性"全部通过"。

**修复方向**：
- 改 skip 为 `pytest.mark.xfail(strict=True, reason="M0 W2 todo")` —— 一旦实现完成立刻变 XPASS（红），强制开发者面对
- 或在 conftest.py 加 sentinel：M0 W2 截止日期到了 skip 的 case 数量 ≥ X → CI fail

## 综合评级

**YELLOW**（gemini 自评）— 不阻塞 push（cc 已 push 4300f89），但 M0 W2 实现前必须先处理 F1 + F2（H 级）。

## cc 主线判断

- 同意 5 个 findings 全部进 backlog
- F1 / F2 是 H 级 → 列入 M0 W2 P0 任务
- F3 / F4 / F5 是 M 级 → 列入 M0 W3 P1 任务
- 明日 codex 配额恢复后再跑一次完整对抗审查（包括 LNN 数学的 Hadamard 维度 / CfC 推导、跨模块依赖深扫等 flash 没覆盖的维度）

## 验证复盘

3 个月后回看：
- F1（SlotValue.value 类型）实际是否引发兼容问题？
- F2（τ 偏置正性）spike Day 2-3 / Day 4-5 实测是否触发？
- gemini-flash 替代 pro/codex 的判断质量差距？

---

**v0.3.1 · cc 整理 gemini-3-flash-preview 输出 · 2026-05-03**
