"""测试 Fusion Layer + 5 Readout Heads · spec §2.1 + §2.3"""
import torch

from poc.lnn.heads import (
    FusionLayer, FusionConfig,
    ReadoutHeads, ReadoutHeadsConfig,
    freeze_module, unfreeze_module,
)


def test_fusion_forward_shape():
    """52d → 64d 投影"""
    fusion = FusionLayer()
    x = torch.randn(8, 30, 52)
    y = fusion(x)
    assert y.shape == (8, 30, 64)


def test_fusion_works_on_2d_input():
    """单帧 (batch, 52) 也应 work"""
    fusion = FusionLayer()
    x = torch.randn(8, 52)
    y = fusion(x)
    assert y.shape == (8, 64)


def test_readout_heads_outputs():
    """5 heads 输出维度全对"""
    heads = ReadoutHeads()
    h = torch.randn(8, 64)
    out = heads(h)

    assert out["state_logits"].shape == (8, 8)
    assert out["va_cont"].shape == (8, 2)
    assert out["intent_logits"].shape == (8, 5)
    assert out["confidence"].shape == (8,)
    assert out["hidden_repr"].shape == (8, 64)

    # va_cont in [-1, 1]
    assert out["va_cont"].abs().max().item() <= 1.0
    # confidence in [0, 1]
    assert 0 <= out["confidence"].min().item() <= out["confidence"].max().item() <= 1.0


def test_readout_heads_num_params():
    """spec §2.2: 5 heads total < 1.5K params"""
    heads = ReadoutHeads()
    n = heads.num_parameters()
    # 64*8+8 + 64*2+2 + 64*5+5 + 64*1+1 = 520+130+325+65 = 1040
    assert n < 1500, f"heads params {n} 超出预算"
    assert n > 500, f"heads params {n} 异常低"


def test_freeze_backbone_for_loss4():
    """spec §3.4 铁律：Loss 4 训练时 backbone freeze 防止 readout 反流"""
    from poc.lnn import CustomCfCCell, CustomCfCCellConfig
    cell = CustomCfCCell(CustomCfCCellConfig(input_dim=52, hidden_dim=64))
    # 初始 all params trainable
    assert all(p.requires_grad for p in cell.parameters())

    freeze_module(cell)
    assert all(not p.requires_grad for p in cell.parameters())
    assert not cell.training  # eval mode

    unfreeze_module(cell)
    assert all(p.requires_grad for p in cell.parameters())
    assert cell.training


def test_fusion_norm_stability():
    """LayerNorm 应让输出 std 接近 1"""
    fusion = FusionLayer()
    x = torch.randn(100, 52) * 10  # input scale 大
    y = fusion(x)
    # GELU + dropout 不会保持完美 std=1，但应在合理范围
    assert 0.1 < y.std().item() < 3.0, f"fusion std={y.std().item()} 异常"


def test_intent_logits_softmax_consistent_with_5_classes():
    """spec ADR-018 IntentType 5 类，确保 head_intent 输出 5 维"""
    cfg = ReadoutHeadsConfig(num_intents=5)
    heads = ReadoutHeads(cfg)
    out = heads(torch.randn(4, 64))
    assert out["intent_logits"].shape == (4, 5)
    # softmax 归一应为 5 维概率分布
    probs = out["intent_logits"].softmax(dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
