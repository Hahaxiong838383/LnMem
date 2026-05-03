"""
spike 内自包含的最简对照集 — 5 个手写 prompt template，各 100 句

不依赖 synthetic_data_generator/ 模块（M0 W2-W4 才建），让 spike 是独立验证单元。

用于 Day 4-5 表达能力测试：看 τ_offset 能否学到区分 5 种行为模式。
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Persona:
    """5 维 mock 人格画像 — 用最简单的特征代替真实 EventTensor 槽位。"""
    name: str                 # 人格代号
    bedtime_hour: int         # 偏好就寝小时（0-23）
    social_score: float       # 社交倾向 [0, 1]
    outdoor_score: float      # 户外倾向 [0, 1]
    productivity_score: float # 效率敏感度 [0, 1]
    morning_person: float     # 晨型程度 [0, 1]


# 5 种区分明确的人格（避免太相似导致区分度天然低）
PERSONAS: List[Persona] = [
    Persona("晚睡型", bedtime_hour=2,  social_score=0.3, outdoor_score=0.2, productivity_score=0.7, morning_person=0.1),
    Persona("早起型", bedtime_hour=22, social_score=0.4, outdoor_score=0.5, productivity_score=0.8, morning_person=0.9),
    Persona("户外型", bedtime_hour=23, social_score=0.6, outdoor_score=0.9, productivity_score=0.4, morning_person=0.7),
    Persona("居家型", bedtime_hour=0,  social_score=0.2, outdoor_score=0.1, productivity_score=0.5, morning_person=0.3),
    Persona("社交型", bedtime_hour=1,  social_score=0.9, outdoor_score=0.6, productivity_score=0.3, morning_person=0.4),
]


def persona_to_feature_vector(p: Persona) -> List[float]:
    """把 Persona 转成 5 维特征向量，用于训练时作为'真实标签'。"""
    return [
        p.bedtime_hour / 24.0,
        p.social_score,
        p.outdoor_score,
        p.productivity_score,
        p.morning_person,
    ]


# 每个 persona 100 个序列样本，每个序列 50 步
SEQUENCES_PER_PERSONA = 100
SEQUENCE_LENGTH = 50
INPUT_DIM = 5  # 与 Persona feature 维度对齐


def generate_persona_sequences(p: Persona, n: int = SEQUENCES_PER_PERSONA, seq_len: int = SEQUENCE_LENGTH, seed: int = 42):
    """
    为每个 persona 生成 n 条带噪声的特征序列。

    数据生成逻辑（最简）：
    - 基础信号 = persona 的 5 维特征向量
    - 每步加入 N(0, 0.1) 噪声模拟"日常波动"
    - 序列长度 50 步 = 模拟一周左右的高频事件

    返回：torch.Tensor [n, seq_len, INPUT_DIM]
    """
    import torch
    torch.manual_seed(seed + hash(p.name) % 10000)

    base = torch.tensor(persona_to_feature_vector(p), dtype=torch.float32)
    sequences = base.unsqueeze(0).unsqueeze(0).expand(n, seq_len, INPUT_DIM).clone()
    sequences += torch.randn_like(sequences) * 0.1
    return sequences


def generate_all_personas_data(seed: int = 42):
    """
    返回训练数据 + 标签：

    X: [N, seq_len, INPUT_DIM]   N = 5 personas × 100 = 500
    y: [N]                       persona index 0-4
    user_ids: [N]                与 y 相同（每个 persona 一个 user_id，0-4）
    """
    import torch
    all_X, all_y, all_uids = [], [], []
    for idx, p in enumerate(PERSONAS):
        X = generate_persona_sequences(p, seed=seed)
        all_X.append(X)
        all_y.extend([idx] * SEQUENCES_PER_PERSONA)
        all_uids.extend([idx] * SEQUENCES_PER_PERSONA)
    X = torch.cat(all_X, dim=0)
    y = torch.tensor(all_y, dtype=torch.long)
    user_ids = torch.tensor(all_uids, dtype=torch.long)
    return X, y, user_ids


if __name__ == "__main__":
    import torch
    X, y, uids = generate_all_personas_data()
    print(f"Generated X: {X.shape}, y: {y.shape}, user_ids: {uids.shape}")
    print(f"Personas: {[p.name for p in PERSONAS]}")
    for idx, p in enumerate(PERSONAS):
        print(f"  [{idx}] {p.name}: feature = {persona_to_feature_vector(p)}")
