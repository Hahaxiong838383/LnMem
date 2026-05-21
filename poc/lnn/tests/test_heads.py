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
    assert all(p.requires_grad for p in cell.parameters())
    freeze_module(cell)
    assert all(not p.requires_grad for p in cell.parameters())
    assert not cell.training
    unfreeze_module(cell)
    assert all(p.requires_grad for p in cell.parameters())
    assert cell.training


def test_freeze_representation_pipeline_critical():
    """**gemini F-4 CRITICAL**: 仅 freeze backbone 不够，必须同时 freeze fusion 防梯度穿透污染"""
    from poc.lnn import CustomCfCCell, CustomCfCCellConfig
    from poc.lnn.heads import freeze_representation_pipeline, unfreeze_representation_pipeline

    fusion = FusionLayer()
    cell = CustomCfCCell(CustomCfCCellConfig(input_dim=64, hidden_dim=64))
    heads = ReadoutHeads()

    # 训练前：全部 trainable
    assert all(p.requires_grad for p in fusion.parameters())
    assert all(p.requires_grad for p in cell.parameters())

    # freeze 表征生成流水线
    freeze_representation_pipeline(fusion, cell)
    assert all(not p.requires_grad for p in fusion.parameters()), \
        "gemini F-4: fusion 必须 freeze 否则 readout grad 反流污染"
    assert all(not p.requires_grad for p in cell.parameters())
    assert all(p.requires_grad for p in heads.parameters()), \
        "readout heads 应保持 trainable"

    # 模拟 Loss 4 训练: heads loss backward 后 fusion grad 必须为 None（确认无穿透）
    x = torch.randn(4, 52)
    h_fused = fusion(x)
    # CustomCfCCell 单步 forward 需 persona_idx
    h_prev = torch.zeros(4, 64)
    persona_idx = torch.zeros(4, dtype=torch.long)
    h = cell(h_fused, h_prev, persona_idx)
    out = heads(h)
    # 触发所有 head 的 backward
    total = (out["state_logits"].sum() + out["va_cont"].sum()
             + out["intent_logits"].sum() + out["confidence"].sum())
    total.backward()

    # 验证：heads 有 grad（至少 state head 因输出依赖最强），fusion 无 grad（无梯度穿透）
    state_w_grad = heads.head_state.weight.grad
    assert state_w_grad is not None and state_w_grad.abs().sum().item() > 0, \
        "head_state 应有非零 grad"
    fusion_grads = [p.grad for p in fusion.parameters()]
    assert all(g is None for g in fusion_grads), \
        f"gemini F-4 漏洞重现：fusion grad 非 None {[g for g in fusion_grads if g is not None]}"

    unfreeze_representation_pipeline(fusion, cell)
    assert all(p.requires_grad for p in fusion.parameters())
    assert all(p.requires_grad for p in cell.parameters())


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
