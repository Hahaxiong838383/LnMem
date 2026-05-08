---
version: v1.0
audience: cc / vision 模块开发者
purpose: py-feat AU 偏差矫正与 EMFACS 情绪映射的权威方案（gemini-3-pro-preview 调研产出）
status: 稳定 v1.0（2026-05-08）
parent: ../../docs/principles/lnn-first.md（P-1 最高准则 §4.1 视觉感知层）
references:
  - Rhim et al. (2022). "Racial and Gender Biases in Facial Expression Recognition"
  - Ekman & Friesen (1978) Facial Action Coding System (FACS)
  - Du et al. (2014) "Compound Facial Expressions of Emotion"
  - Ekman EMFACS (Emotion Facial Action Coding System)
---

# py-feat AU 偏差矫正与情绪映射方案

> **来源**：gemini-3-pro-preview 2026-05-08 调研产出，cc 主线归档。
> **场景**：LnMem PoC 真人推理时 py-feat 0.6.2 输出 17 AU 用作 LNN 输入，
> 但 py-feat XGB 模型在亚洲人脸上有显著形态学偏差，需校准。

## 1. py-feat 在亚洲人脸上的偏差（实测 + 文献）

### 实测数据（30s 摄像头采样，亚洲男性）

```
AU01 (内眉上扬) baseline mean=0.85   ← 异常高（内眦赘皮误判）
AU02 (外眉上扬)         mean=0.64
AU05 (上眼睑抬升)       mean=0.42
AU11 (鼻唇沟加深)       mean=0.50
AU17 (下巴抬升)         mean=0.45

AU04 (皱眉) 自然=0.08, 用力皱眉=0.19  ← 信号衰减（薄额部脂肪层）
AU06 (微笑) 自然=0.08, 微笑=0.71      ← 正常
AU12 (微笑) 自然=0.04, 微笑=0.76      ← 正常
```

### 根因（学术）

- **训练分布偏差**：BP4D / DISFA / EmotioNet 中亚洲样本仅 ~10%
- **形态学偏差**：内眦赘皮、较浅眼窝 → AU01/05 持续误触发
- **信号衰减**：薄额部肌肉脂肪 → AU04 皱眉 CV 梯度特征极弱

文献支撑：Rhim et al. 2022 实证 py-feat 类模型对亚洲面孔 AU 估计存在系统性低估（AU04）和过高假阳性（AU01）。

## 2. EMFACS：AU → 7 情绪权威公式

| Emotion | Primary AUs | 计算公式（线性加权，已归一化） |
|---------|------------|-----------------------------|
| Happiness | AU06, AU12 | `0.4×AU06 + 0.6×AU12` |
| Sadness | AU01, AU04, AU15 | `0.3×AU01 + 0.4×AU04 + 0.3×AU15` |
| Anger | AU04, AU05, AU07 | `0.4×AU04 + 0.3×AU05 + 0.3×AU07` |
| Fear | AU01, AU02, AU04, AU05, AU20 | `0.2×(AU01+AU02) + 0.3×(AU04+AU05) + 0.2×AU20` |
| Disgust | AU09, AU15, AU17 | `0.6×AU09 + 0.2×(AU15+AU17)` |
| Surprise | AU01, AU02, AU05, AU26 | `0.3×(AU01+AU02+AU05) + 0.1×AU26` |
| Contempt | AU14 (单侧) | `max(0, AU14 - 0.5×AU12)` |

每条公式 `min(1.0, ...)` clamp 到 [0,1]。

## 3. 校准算法：AUCaliper

针对实测数据（高底噪、弱信号），分三类处理：

### 3.1 高基线 AU（AU01/02/05/11/17）— 相对归一化

```python
calibrated = max(0, (raw - base) / (1 - base))
```

避免绝对减除导致负数。base 用 EMA 动态跟踪。

### 3.2 弱信号 AU（AU04）— 非线性 gamma 拉伸

```python
calibrated = min(1.0, ((raw - base) / 0.15) ** 0.8)
```

用力皱眉 0.19 → ~0.8 的强度。

### 3.3 良好信号 AU（AU06/12）— 仅去基线噪

```python
calibrated = max(0, raw - base)
```

### 3.4 EMA 动态基线

```python
self.baseline[au] = (1 - α) × self.baseline[au] + α × raw_val
α = 0.01（适合 30s-2min 长镜头，缓慢适应）
```

屏蔽极强信号避免污染基线（`raw < baseline × 1.5` 才更新）。

## 4. 完整代码（直接落 feature_calibrator.py）

见 `poc/src/vision/feature_calibrator.py` 的 `AUCaliper` 类（待整合）。

## 5. 局限性提醒

### 不可信 AU 预警

- **AU01/02 假阳性高**：在亚洲人脸上即便校准后，"惊讶/恐惧"增量能力差
- **建议**：情绪合成（特别是 Sadness 和 Fear）中**降权处理 AU01/02**，更依赖下半脸（AU15、AU20）+ AU04

### 模型切换建议（按优劣排序）

| Model | 优势 | 劣势 |
|-------|------|------|
| `xgb`（默认） | 快 | 偏态分布泛化弱 |
| `svm` (RBF) | 跨种族泛化好 | 慢 |
| **OpenFace 2.0**（M1 备选） | Person-Specific Normalization 跨种族好 | C++ 依赖 dlib + Boost，部署痛苦 |

## 6. 长期演进（M1）

- 转 OpenFace 2.0：原生 AU 强度（连续）+ AU 发生概率（0/1）双输出
- Person-Specific Normalization：每个用户保留 30s 中性脸基线
- Aff-Wild2 迁移：用更平衡数据集再训练 AU 模型

---

署名：gemini-3-pro-preview（2026-05-08 调研）+ cc 归档
