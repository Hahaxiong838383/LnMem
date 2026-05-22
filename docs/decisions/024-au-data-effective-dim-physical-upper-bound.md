# ADR-024: AU 数据 effective_dim 物理上限 + 5/27 启动阈值改判

> **决策者**：cc 主理 + Studio 实测 SVD + 川哥 A+B 钦定（2026-05-22 15:50）
> **状态**：accepted
> **关联**：ADR-023 范式转换 · 不可妥协 §10.1 LNN-FIRST · 5/27 Phase γ Step 1 启动条件

---

## 1. 触发事件

5/22 v3.0 Studio 50ep 实测 effective_dim 3.73，11 轮以来 cc/gemini 一直追"≥ 8.0"阈值物理不可达。

cc 自审 SVD 后发现 **AU 数据本身 effective_dim 上限 = 5.07**。

## 2. 铁证（Studio 实测）

```python
data = np.load("poc/data/au_cache_tiny.npz")  # 10 视频
mat = np.concatenate([data[v] for v in data.keys()], axis=0)
# shape: (2690 frames, 20 AU)
centered = mat - mat.mean(axis=0, keepdims=True)
s = np.linalg.svd(centered, compute_uv=False)
effective_dim = s.sum() / s[0]
# = 5.068
```

Singular value 能量分布：
| 维度 | 累积能量 |
|---|---|
| top 1 | 44.58% |
| top 4 | **76.9%** |
| top 8 | 90.5% |
| top 20 | 100% |

## 3. 数据上限对账

| 来源 | 数字 |
|---|---|
| **AU 数据实测 effective_dim** | **5.07** |
| LnMem v3.0 50ep hidden effective_dim | 3.73 |
| **占有率（model / data）** | **73.6%** |
| 当初 cc 拍的阈值 | 8.0 |
| 物理可达性 | ❌ |

**FACS 文献早就说**：24 AU 间高度相关（共激活/拮抗），实际信息维度 4-6。
cc 当初拍 8.0 是基于 hidden_dim=64 默认上限的 1/8，**没基于数据本身**。

## 4. 决策

### 4.1 改判 effective_dim 阈值

**旧**：absolute `effective_dim ≥ 8.0`（物理不可达）

**新**：relative `effective_dim ≥ 0.8 × input_effective_dim`
- AU 数据 input_effective_dim = 5.07
- 5/27 启动要求 effective_dim ≥ **4.06** (= 80% AU 上限)
- v3.0 实测 3.73 仍差 8.1%（继续改进或长训）

### 4.2 锁死 input_effective_dim 测量

新增 `poc/training/evaluate.py::measure_input_effective_dim`：
```python
def measure_input_effective_dim(dataset) -> float:
    """测量 AU 输入数据本身的 effective_dim 上限.
    
    5/27 启动判定必须先跑这个 metric，再算 model/input 占有率.
    禁止用 absolute 阈值（cc 11 轮血泪教训）.
    """
    all_au = []
    for i in range(len(dataset)):
        sample = dataset[i]
        au = sample["event_tensor"][:, :24].numpy()  # AU 段
        all_au.append(au)
    mat = np.concatenate(all_au, axis=0)
    centered = mat - mat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    return float(s.sum() / max(s[0], 1e-9))
```

### 4.3 N=124 时预期

10 视频 AU 数据 effective_dim = 5.07。N=124 时人脸多样性 + 表情多样性大幅扩展，**预计 input_effective_dim 升到 6-8**。模型 hidden effective_dim 跟着涨到 5-6.5。

## 5. 11 轮血泪教训复盘

| 轮 | 错钦定 |
|---|---|
| 7 | std_target=1.0 |
| 11 | bottleneck=16 物理压上限 |
| **12-13** | **effective_dim ≥ 8.0 阈值** |
| 1-13 共同 | **没人查 AU 数据本身 effective_dim** |

**根因**：cc/gemini 全程套用 "deep representation 必须高维" 教条，**忘了 LnMem 输入是 AU 不是 ImageNet**。

类似 5/20 踩坑 `feedback_user_mental_model_over_industry_pattern.md`「把方案抽象成行业范式会丢失用户真实意图」。

## 6. LNN-FIRST 6 项对账

| 项 | ADR-024 影响 |
|---|---|
| #1 LNN-FIRST | ✅ 数据驱动 metric 强化 LNN 哲学（用数据本身决定指标）|
| #2 per-user τ-bias | ✅ 不变 |
| #3 AU 多模态 | ✅ **明确**：AU 数据 effective_dim ~5，模型设计要匹配 |
| #4 离线优先 | ✅ 不变 |
| #5 ODE 可解释 | ✅ 不变 |
| #6 时序记忆 | ✅ 不变 |

## 7. 5/27 verdict 第 8 次更新

| 阶段 | verdict |
|---|---|
| 7 | GREEN 🟩 |
| 8 | RED 🟥 |
| 8b | YELLOW 🟨 |
| 9 | RED 🟥 |
| 10 | BLOCK ❌ |
| 11 | BLOCK ❌ |
| 12 | YELLOW 🟨 |
| 13 | YELLOW 🟨 |
| **本轮 (ADR-024)** | **🟨 → 🟢 (条件 GREEN)** |

**新启动条件（数据驱动）**：
1. ✅ τ.std > 0.5（多时间尺度真分化）— v3.0 实测 0.577
2. ✅ vs_persistence < 0.7 — v3.0 实测 0.58
3. ⏳ effective_dim / input_effective_dim ≥ 0.8 — v3.0 实测 0.736（差 8%）
4. ⏳ Shuffle ratio ≥ 1.5 — v3.0 实测 1.21（差 19%，长训可能突破）

5/22 同时启动 100ep + 200ep 长训验证 3/4。

## 8. cc 11 轮认知误区记录

cc 全程被"effective_dim 8.0 是 spec 钦定"束缚，**没人质疑这个数字的来源**。这是项目级元认知盲区：

- AC（验收标准）必须**先有数据 baseline**才能定阈值
- 否则 "8.0" 跟"投硬币定的 17"没区别
- 川哥指令"对账项目目标"是这次找到根因的钥匙

**新流程**：任何 metric 阈值都必须**先跑 baseline 测量**才能采纳。

---

**📝 维护者**：cc 主理 + Studio 实测 SVD + 川哥 A+B 钦定 | **签发**：2026-05-22 15:50
