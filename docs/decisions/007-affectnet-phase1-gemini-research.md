# ADR-007 AffectNet base ckpt 真人化 Phase 1 实施方案

## 第 1 章：AffectNet 数据获取方案
- **学术 License 申请途径**：官方通道（[mohammadmahoor.com/affectnet](http://mohammadmahoor.com/affectnet/)）需 PI/导师名义填表，耗时可能数天至数周；建议首选 Kaggle 上现存的非官方精简版（96x96 或 224x224 裁剪版）用于 M0 W3 快速跑通 pipeline。
- **数据规模**：全量约 45 万张手动标注图（8 Emotion + Valence/Arousal 连续标注），全量近 122GB。精简版约几 GB，含核心手动标注。
- **替代选项（降级预案）**：
  - **Aff-Wild2**：真实自然场景视频集，带逐帧 V-A 标注，比静态图更契合 LNN 演化，但申请流程类似。
  - **DEAP**：32 人多模态集（含视频），V-A 连续质量高但场景与人员丰富度偏小。
  - **RAVDESS**：仅 8 类离散情感，缺乏连续 V-A，且演员表演痕迹过重，不契合 LTC 平滑方程需求。

## 第 2 章：py-feat AU 提取 pipeline
- **工时与速度估算**：在 M2 Max 24GB 上使用 `py-feat`，由于 MPS 支持处于实验期，建议设 `device='cpu'` 配合 `batch_size=32` 或 `64` 以发挥统一内存多核优势。估算处理速度可达 50-100 FPS，45 万张图预估提取耗时约 1.5-2.5 小时。
- **Pipeline 优化**：推荐选用 `'retinaface'` 轻量模型。避免逐帧 `detect()`，须将数据组装为图片路径 list 后批量传入。
- **中间产物存储**：强烈建议落盘为 **Parquet** 格式（而非 CSV 或 Numpy）。Parquet 压缩率高、I/O 极快，且天然与 Pandas/Polars 的大 Batch 过滤操作高度契合，便于后续与 V-A 标签联合 join。

## 第 3 章：V-A → 8 state 离散映射设计
- **映射方案（Russell Circumplex 启发）**：
  - `focused` (1): V ∈ [0, 0.3], A ∈ [0.3, 0.7] (微正向、中高唤醒)
  - `emotional_positive` (4): V > 0.3, A > 0.3
  - `emotional_negative` (5): V < -0.2, A > 0.3
  - `distracted` (2): V ∈ [-0.2, 0.2], A ∈ [0.2, 0.5] (边缘状态)
  - `fatigued` (3): V < 0, A < 0.2 (低能量、负向)
  - `low_energy` (0): V ∈ [-0.2, 0.2], A < 0.2
  - `neutral` (6): V ∈ [-0.1, 0.1], A ∈ [0.1, 0.3]
  - `transitioning` (7): 基于相邻帧的 ΔV 和 ΔA 动态计算，不设绝对阈值。
- **Sweep 设计**：针对川哥 `focused` 被误判为 `negative` 的痛点，需对 `focused` 边界进行 Sweep。扫 `V_lower` ∈ [-0.1, 0, 0.1] × `A_lower` ∈ [0.2, 0.3, 0.4]。每组超参前向过滤与评估耗时在分钟级。

## 第 4 章：训练 pipeline 集成 fit_orchestrator
- **集成设计**：Pretrain 阶段与 `fit_orchestrator.py` 物理隔离。新增 `train_base_v2.py` 生成 base ckpt v2 并存入 `poc/data/checkpoints/`，随后 `fit_orchestrator` 的 `try_fit_step` 将无缝加载 V2 进行 OnlineLoRAAdapter 微调。
- **损失函数选择**：推荐**多任务联合 Loss**。主 Loss 使用 CCC（Concordance Correlation Coefficient）回归拟合 V-A 连续变量（完美契合 LNN 连续空间），辅 Loss 使用 CE (Cross Entropy) 进行 8 state 离散分类。
- **资源估算**：LNN 参数极少。45 万条 AU 特征序列，M2 Max 纯 CPU 训练 100 Epoch 预计在 1-2 小时内完成。内存占用峰值不超过 4GB。

## 第 5 章：验收 + 风险
- **关键验收指标**：在川哥 402 次真实反馈中，`brow_burst` 引发的 mismatch 率从 30.7% 降至 <10%；`user_triggered` 误判降至 <20%。
- **A/B 对比方法**：运行 `scripts/ab_replay_predict.py`，分别挂载 base v1（合成）与 base v2（AffectNet），输入 402 次历史 raw_au，对比两者混淆矩阵，验证 `focused` 的召回率绝对提升。
- **风险清单 (Top 5)**：
  1. **动力学缺失 (Critical)**：AffectNet 是静态图，而 LNN 需要时序序列。简单复制静态图伪造序列，会污染 `tau` 的时间常数学习，导致动态特性退化。
  2. **商用版权暴雷**：AffectNet 严禁商用，M3 阶段若不替换或购买授权，将阻断产品化。
  3. **数据域偏移**：欧美脸型比例过高，对川哥面部特征的泛化仍有折扣。
  4. **硬截断阶跃**：V-A 转 8 state 的边界依然存在人为阶跃，削弱微分方程平滑优势。
  5. **低分辨率 AU 劣化**：Kaggle 精简版的低分辨率可能导致 `py-feat` 对微妙眉毛动作（brow_furrow）提取失效。
- **工时估算（AI 同事 + cc 主线）**：
  数据提取 (0.5d) + 规则 Sweep (0.5d) + Pretrain/Eval (1.5d) = **共计 2.5 天**（无阻塞依赖）。

## Summary 推荐路径
立刻从 Kaggle 抓取 AffectNet 精简版，用 `py-feat` (batch=32, Parquet 落盘) 提 AU。采用 V-A 连续回归 (CCC Loss) 与 State 离散分类 (CE Loss) 联合预训练 base ckpt v2。为规避静态图破坏 LNN 动力学参数 `tau` 的致命风险，预训练时应冻结动力学参数，仅训练空间映射层，随后交由 `fit_orchestrator` 进行时序微调。通过 `ab_replay_predict.py` 验证 402 条 feedback 改善后即验收。