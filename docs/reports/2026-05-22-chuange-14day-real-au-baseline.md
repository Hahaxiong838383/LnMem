# 川哥 14 天 daemon 真人 AU baseline（ADR-025 V0.2 校准决策）

> **作者**：cc 主线 + codex 适配器 | 2026-05-22 22:15
> **关联**：ADR-024 数据 effective_dim 物理上限 · ADR-025 罕见 AU 监控 V0.2 · gemini round 14 N=124 预测

## 1. 数据来源

- daemon 跑了 14 天（2026-05-09 → 2026-05-22）@ Air 摄像头 1 Hz 1280×720
- `~/.lnmem/data/signals_*.jsonl`，py-feat AU 字段已与 LnMem PYFEAT_AU_COLUMNS 完全对齐
- 已做 per-user calibration（`calibrated_au` 字段）

## 2. 数据规模

| 指标 | 数字 |
|---|---|
| 总信号行数 | 670,358 |
| `face_detected=true` 比例 | 79.0% |
| **AU 完整 + 非 NaN 有效帧** | **378,361** |
| session 数 | 55 |
| 对比 VEATIC 10 vid（ADR-024） | 2,690 → **141× 川哥数据更大** |
| 对比 VEATIC N=124 估算 | 33,500 → **11.3× 川哥数据更大** |

## 3. effective_dim 实测

| 数据集 | effective_dim | top-1 能量 | top-4 能量 | top-8 能量 |
|---|---|---|---|---|
| VEATIC 10 vid (ADR-024) | 5.07 | 44.6% | 76.9% | 90.5% |
| **川哥 14 天 calibrated** | **7.35** | **13.6%** | **41.7%** | **65.7%** |
| 川哥 14 天 raw | 7.02 | 14.2% | 41.7% | 65.2% |
| gemini round 14 N=124 预测 | 7.5 – 9.0 | — | — | — |

### 洞察

1. **川哥 effective_dim 7.35 已落入 gemini 14 N=124 预测区间 [7.5, 9.0] 下界附近**
   - 验证 gemini 14 的预测方向正确（÷2.5 sanity check 之后预测的 7.5-9 是靠谱的）
   - VEATIC 是 emotion clip 表演 → AU 模式集中；川哥日常 14 天 → AU 模式多样
   - 川哥 top-1 只占 13.6% vs VEATIC 44.6%，能量分布扁平得多

2. **ADR-024 5/27 启动门 #3 (effective_dim ≥ 0.8 × input_effective_dim) 在川哥数据上变难**
   - VEATIC 数据上：阈值 = 0.8 × 5.07 = 4.06
   - 川哥数据上：阈值 = 0.8 × 7.35 = **5.88**
   - 这是好事：模型上限被打高，能力上限可证明性更强

## 4. AU 长尾分布（calibrated, threshold=0.1）

按激活频率排序（高 → 低）：

| 排名 | AU | freq | FACS 名称 | 川哥日常含义 |
|---|---|---|---|---|
| 1 | AU04 | **0.848** | brow lowerer | 皱眉（思考/盯屏幕）|
| 2 | **AU28** | **0.589** | lip suck | **抿嘴**（专注）|
| 3 | AU15 | 0.573 | lip corner depressor | 嘴角下压 |
| 4 | AU06 | 0.512 | cheek raiser | 颧骨抬升 |
| 5 | AU01 | 0.478 | inner brow raiser | 内眉抬 |
| 6 | AU11 | 0.448 | nasolabial deepener | 鼻唇沟加深 |
| 7 | AU26 | 0.420 | jaw drop | 张嘴 |
| 8 | AU14 | 0.412 | dimpler | 酒窝 |
| 9 | AU24 | 0.405 | lip presser | 嘴唇紧压 |
| 10 | **AU43** | **0.394** | eye closure | **闭眼/眨眼** |
| 11 | AU17 | 0.387 | chin raiser | 下巴抬 |
| 12 | AU09 | 0.367 | nose wrinkler | 鼻翼皱 |
| 13 | AU23 | 0.347 | lip tightener | 嘴唇收紧 |
| 14 | AU02 | 0.336 | outer brow raiser | 外眉抬 |
| 15 | AU25 | 0.301 | lips part | 唇分开 |
| 16 | AU05 | 0.280 | upper lid raiser | 睁大眼（惊讶）|
| 17 | AU12 | 0.263 | lip corner puller | **微笑** (川哥很少😄) |
| 18 | AU20 | 0.135 | lip stretcher | 嘴抿宽 |
| 19 | AU10 | 0.112 | upper lip raiser | 上唇抬 |
| 20 | **AU07** | **0.087** | lid tightener | 眼睑收紧（最罕见）|

## 5. 🔥 颠覆性发现：ADR-025 fixed_bottom 钦定的 AU28/AU43 错了

| 钦定 AU | 川哥实测排名 | freq | 钦定假设 vs 实测 |
|---|---|---|---|
| AU28 (lip suck) | **第 2 名（高频）** | 0.589 | ❌ "罕见微表情" 假设破产 |
| AU43 (eye closure) | 第 10 名（中频） | 0.394 | ❌ "罕见微表情" 假设也破产 |

### 根因

gemini round 14 sign-off 钦定 AU28/AU43 是基于 **FACS 临床文献中的"罕见微表情"理论**——但**实际罕见性取决于个体使用模式 + 数据集场景**：

- 川哥日常 14 天 = 专注思考 + 盯屏幕 + 偶尔讲话 → 抿嘴 AU28 极频繁
- 川哥很少做 "惊讶"AU05 + "大笑"AU12 + "嫌弃"AU09 等表情
- VEATIC = emotion clip 表演 → 长尾分布跟川哥日常**不一样**（VEATIC 数据未直接验，需 5/23 Studio 上算一次对照）

### 川哥真实罕见 AU top-7（应当死守 bottom 桶）

`AU07 (0.087)`, `AU10 (0.112)`, `AU20 (0.135)`, `AU12 (0.263)`, `AU05 (0.280)`, `AU25 (0.301)`, `AU02 (0.336)`

## 6. ADR-025 V0.2 → V0.3 校准建议

### 不推荐：保留硬编码 `DEFAULT_FIXED_BOTTOM_INDICES = (18, 19)`
- 问题：N=124 训练用 VEATIC 数据 → bucket 用 VEATIC AU 分布；Phase δ 微调用川哥数据 → bucket 用川哥 AU 分布。**fixed_bottom 应当 dataset-specific**

### 推荐方案 A：移除硬编码默认，强制传入
```python
def partition_au_by_frequency(
    activation_freq: np.ndarray,
    top_k: int = 7,
    bottom_k: int = 7,
    fixed_bottom_indices: tuple[int, ...] = (),  # 改默认空
) -> AuBuckets:
```
- 优点：让调用方按数据集自主决定
- 缺点：失去 gemini 14 "AU43/AU28 必须监控" 的安全网

### 推荐方案 B：保留 FACS 钦定 + 加 "dataset-adapted" 旁路
- 默认 `DEFAULT_FIXED_BOTTOM_INDICES = (18, 19)` 不变（FACS 理论锚点）
- 加 `auto_fixed_bottom_by_freq(activation_freq, percentile=10) -> tuple[int, ...]`：自动选实际数据上频率最低的几个 AU
- 调用方明确选 `fixed_bottom_indices=DEFAULT_FIXED_BOTTOM_INDICES`（VEATIC 评估）或 `auto_fixed_bottom_by_freq(freq)`（per-dataset 自适应）

### 推荐方案 C（cc 倾向）：**双锚 hybrid**
- `fixed_bottom_indices = DEFAULT_FIXED_BOTTOM_INDICES + auto_lowest_k_by_freq(freq, k=2)`
- 既保 gemini 14 钦定的 AU28/AU43（理论安全网），又自动锁定实际数据上最罕见的 2 个
- 总 fixed = 4 个（≤ bottom_k=7），剩余 3 名额按频率排序填

## 7. 5/27 启动门校准

| 门 | ADR-024 原值 | 川哥数据校准后 | 备注 |
|---|---|---|---|
| #1 τ.std > 0.5 | 0.5 | 0.5 | 不变 |
| #2 vs_persistence < 0.7 | 0.7 | 0.7 | 不变 |
| #3 effective_dim ≥ 0.8 × input | 0.8 × 5.07 = 4.06 | **0.8 × 7.35 = 5.88** | **按真实数据集动态算** |
| #4 Shuffle ratio ≥ 1.5 | 1.5 | 1.5 | 不变 |
| #5 rare_au_verdict ∈ {green, yellow} | V0.2 | V0.3（看 §6 决策）| 必改 |

## 8. 真人数据使用边界（伦理）

| 用途 | 评估 |
|---|---|
| 加入 N=124 主训练集 | ❌ 标签维度不同（match/mismatch vs V-A）/ FPS 不同 / 单人 ≠ 124 不同人 |
| **替代 P0-1 β-self-test** | ✅✅ 现有数据远超原计划 10 句对话，直接收割即可 |
| **作为 ADR-025 V0.3 阈值校准 baseline** | ✅✅ 本报告即产出 |
| Phase δ 真人微调 6/10 | ✅ per-user τ-bias 训练数据 |
| commit / PR 出仓 | ❌ 生物特征敏感（mycc P1 隐私边界协议尚未写）|

`~/.lnmem/` 已天然在仓外，无泄漏风险。但任何衍生分析报告涉及具体 AU 数值时，**只能写入本文件（已 gitignore? 待确认）+ 不进 PR**。

## 9. 下一步建议

按优先级：

1. **5/23 早**：去 Studio 跑 VEATIC 10 vid 的 AU activation_freq 看 VEATIC 数据上 AU28/AU43 真的罕见吗 → 决定方案 A/B/C
2. **5/23 早**：v3.0 200ep ckpt 在 Studio，在川哥 daemon cache 上跑 evaluate → 看全 6 道启动门表现（一锅端 ROADMAP P0-1）
3. **5/23 中**：按方案 C 落地 ADR-025 V0.3 + ADR-026（5/27 启动门 #3 改为按 dataset 动态算 input_effective_dim）
4. **5/27 启动前**：确认 N=124 训练集本身的 AU 长尾分布，固化 fixed_bottom_indices

---

**📊 数据文件**：
- `/tmp/daemon_au_cache.npz`（55 sessions / 378,361 frames，本地不入仓）
- `/tmp/analyze_chuange_real_au.py`（cc 分析脚本）

**📝 维护者**：cc + codex 适配器 | **签发**：2026-05-22 22:15

---

# 附录：5/22 22:30 Y2 真人 inference 结果 + V0.4 补丁

## Y2 v3.0 200ep ckpt 真人推理（poc/training/inference_driver.py）

`/tmp/v30_chuange_eval_v04.json` (后 V0.4 补丁版)：

```json
{
  "n_sessions": 55,
  "total_frames": 378361,
  "gate_1_tau_std": "N/A (需 trainer 内部状态)",
  "gate_2_vs_persistence": 2.266,
  "gate_3_effective_dim_ratio": 0.707,
  "gate_4_shuffle_ratio": 0.008,
  "gate_5_trivial_recon_check": false,
  "gate_6_rare_au_verdict": "n/a"
}
```

### 两数据集 V3.0 表现完整对比

| 门 | 阈值 | VEATIC 10 vid | **川哥 14 天 daemon** | 状态反转 |
|------|------|--------------|---------------------|---------|
| 1 τ.std > 0.5 | > 0.5 | 0.577 ✅ | N/A | — |
| 2 vs_persistence | < 0.7 | 0.50 ✅ | **2.27 ❌** | 比 trivial 差 2.3x |
| 3 effective_dim / input | ≥ 0.8 | 0.97 ✅ | **0.71 ❌** | 川哥 input 高 → 覆盖率掉 |
| 4 Shuffle ratio | ≥ 1.5 | 1.73 ✅ | **0.008 ❌** | 时序信息几乎归零 |
| 5 trivial_recon_check | True | True ✅ | **False ❌** | naive 强过模型 |
| 6 rare_au_verdict | green/yellow | (未跑) | **n/a** (V0.4 整体崩盘) | — |

## V0.4 cc Y2 新增 finding：整体崩盘门

旧 V0.3 D' 在川哥数据上误判 verdict=green（因为 ratio_bottom=0.73 < gate=0.5 fallback 触发 green 分支），但实际 ratio_top=2.17（模型在高频桶也比 trivial 差）→ 模型整体崩盘，"长尾被牺牲"语义不适用。

### 实现
- 新常量 `TOP_COLLAPSE_RATIO = 1.0`
- evaluate_rare_au_drop 首先检查 `r_top >= TOP_COLLAPSE_RATIO` → verdict=`n/a`
- 防止整体不 work 的模型被 green 误判

## 项目级 LnMem 重大启示

1. **5/22 23:30 的 5/27 GREEN LIGHT 是 distribution-internal 乐观判断**
   - 4 道门全过都基于 VEATIC 自评，没做 cross-distribution 验证
   - 真人 baseline 之后才发现"VEATIC 训练 → 真人推理"几乎不 work
2. **Phase δ 真人微调不是可选，是必需**
   - 训练分布 (VEATIC 30fps emotion clips) ≠ 部署分布 (daemon 1fps 日常监控)
3. **根因猜测**（5/23 团队验证）
   - 采样率差 30x（30fps vs 1Hz）
   - 领域偏移（演员表演 vs 日常微动）
   - calibrated_au 分布偏移（训练用 raw_au）
4. **mycc 教训再次验证**
   - `feedback_threshold_must_have_data_baseline_first`（V0.2/V0.3 阈值都靠假数据起步）
   - `feedback_dynamic_assessment_over_checklist`（gate 设计需 dataset-aware 而非硬阈值）

## 5/23 优先级修订

| 原 P0 | 新 P0 |
|------|------|
| β-self-test webcam 10 句 | ⚠️ **改为：调查 30fps vs 1Hz 差距能否 sub-sample 缓解** |
| daemon 长跑后台 | 仍然 P1 |
| long training 500/1000ep | 降 P2（先解决 cross-dist 偏移） |

**新 P0**：5/23 早试验在 v3.0 ckpt 上做 fps 适配（用 daemon 数据但模拟 30fps stride=1）看 gate_4 shuffle ratio 能否提升。如果可以 → Phase δ 微调方向明确；如果不行 → ADR-027 提出新架构。

---

**📝 V0.4 + Y2 补丁维护者**：cc Y2 实证 + V0.4 patch | **签发**：2026-05-22 22:45
