"""TDD: poc/src/lnn/extend_persona.py — vboss_self 入 ckpt 工具。

覆盖
----
1. parse_persona_markdown：HEXACO + ECR + motivation 抽取正确 / partial 字段标注 / id mismatch
2. hexaco_to_embedding：8 维形状 / 中心化范围 / 数值由自评决定
3. extend_ckpt_with_persona：embedding +1 行 / persona_to_idx 加 / config.n_personas+1 / 不破坏 base
4. **端到端**：扩展后的 ckpt 加载到 StateRecognitionLNN 后 forward(x, idx=new) 不报错
5. partial 字段进 metadata（CAPS / 多模态 / Schwartz）
6. PersonaParseError：persona_id 已存在 / dst 已存在不 overwrite / md 缺字段
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.extend_persona import (
    PersonaParseError,
    extend_ckpt_with_persona,
    hexaco_to_embedding,
    parse_persona_markdown,
)
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.personalization import DefaultPersonalizationAdapter


# ---------- Fixtures ----------


VBOSS_SELF_FRONTMATTER = """---
id: vboss_self
name: 测试自评
hexaco:
  honesty_humility: 0.667
  emotionality: 0.500
  extraversion: 0.750
  agreeableness: 0.667
  conscientiousness: 0.833
  openness: 0.917
attachment:
  anxiety: 0.292
  avoidance: 0.708
  type: avoidant
  ecr_style: dismissive_avoidant
  confidence: 0.72
motivation:
  self_efficacy: 1.000
  autonomy_need: 1.000
  competence_need: 1.000
  relatedness_need: 1.000
values_top3: [self-direction, achievement, power]
values_bottom: [tradition, conformity, security]
data_source: personalt-test v0.2 (test fixture)
---

# vboss_self test fixture
"""


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "vboss_self_fixture.md"
    p.write_text(VBOSS_SELF_FRONTMATTER, encoding="utf-8")
    return p


@pytest.fixture
def src_ckpt(tmp_path: Path) -> Path:
    """写一个最小 ckpt (n_personas=3) 模拟 day7_v2 结构。"""
    config = LNNTrainConfig(
        feature_dim=8,
        n_state_classes=4,
        n_personas=3,
        persona_embed_dim=8,  # 注意：extend_persona.py 写死 embed_dim=8
        hidden_dim=8,
        sequence_length=4,
        sequence_stride=2,
        sequence_min_frames=2,
        batch_size=2,
        epochs=1,
        device="cpu",
    )
    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(config)
    boundary_model = factory.create_boundary_model(config)
    ckpt = {
        "config": config,
        "persona_to_idx": {"a": 0, "b": 1, "c": 2},
        "state_model": state_model.state_dict(),
        "boundary_model": boundary_model.state_dict(),
        "epoch_metrics": (),
        "final_state_acc": 0.5,
        "final_boundary_f1": 0.0,
    }
    p = tmp_path / "src_ckpt.pt"
    torch.save(ckpt, p)
    return p


# ---------- parse_persona_markdown ----------


class TestParseMarkdown:
    def test_hexaco_extracted(self, md_file: Path) -> None:
        data = parse_persona_markdown(md_file)
        assert data["id"] == "vboss_self"
        assert data["hexaco"]["openness"] == pytest.approx(0.917)
        assert data["hexaco"]["conscientiousness"] == pytest.approx(0.833)

    def test_attachment_extracted(self, md_file: Path) -> None:
        data = parse_persona_markdown(md_file)
        assert data["attachment"]["avoidance"] == pytest.approx(0.708)
        assert data["attachment"]["ecr_style"] == "dismissive_avoidant"

    def test_motivation_all_one(self, md_file: Path) -> None:
        data = parse_persona_markdown(md_file)
        for k in ("self_efficacy", "autonomy_need", "competence_need", "relatedness_need"):
            assert data["motivation"][k] == 1.0

    def test_partial_fields_annotated(self, md_file: Path) -> None:
        """personalt-test 来源应自动标注 CAPS / 多模态 / Schwartz partial。"""
        data = parse_persona_markdown(md_file)
        flags = set(data["partial_fields"])
        assert "caps_if_then_patterns_empty" in flags
        assert "multimodal_signature_confidence_0.3" in flags
        assert "schwartz_values_full_10dim_partial" in flags
        # attachment.confidence < 0.8 也应标注
        assert "attachment.confidence_low" in flags

    def test_missing_hexaco_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.md"
        p.write_text("---\nid: x\n---\n", encoding="utf-8")
        with pytest.raises(PersonaParseError):
            parse_persona_markdown(p)

    def test_missing_frontmatter_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "noyaml.md"
        p.write_text("# no frontmatter\n", encoding="utf-8")
        with pytest.raises(PersonaParseError):
            parse_persona_markdown(p)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(PersonaParseError):
            parse_persona_markdown(tmp_path / "nonexistent.md")


# ---------- hexaco_to_embedding ----------


class TestHexacoEmbedding:
    def test_shape_8d(self, md_file: Path) -> None:
        data = parse_persona_markdown(md_file)
        emb = hexaco_to_embedding(data, embed_dim=8)
        assert emb.shape == (8,)

    def test_centered_range(self, md_file: Path) -> None:
        data = parse_persona_markdown(md_file)
        emb = hexaco_to_embedding(data)
        # 自评所有原始值 ∈ [0, 1]，中心化后 ∈ [-0.5, +0.5]
        assert float(emb.min()) >= -0.5 - 1e-6
        assert float(emb.max()) <= 0.5 + 1e-6

    def test_dim_meaning_hexaco_first6(self, md_file: Path) -> None:
        """前 6 维直接对应 HEXACO 6 因素 - 0.5。"""
        data = parse_persona_markdown(md_file)
        emb = hexaco_to_embedding(data)
        h = data["hexaco"]
        assert emb[0].item() == pytest.approx(h["honesty_humility"] - 0.5)
        assert emb[1].item() == pytest.approx(h["emotionality"] - 0.5)
        assert emb[2].item() == pytest.approx(h["extraversion"] - 0.5)
        assert emb[3].item() == pytest.approx(h["agreeableness"] - 0.5)
        assert emb[4].item() == pytest.approx(h["conscientiousness"] - 0.5)
        assert emb[5].item() == pytest.approx(h["openness"] - 0.5)

    def test_dim6_avoidance_dim7_motiv_mean(self, md_file: Path) -> None:
        data = parse_persona_markdown(md_file)
        emb = hexaco_to_embedding(data)
        assert emb[6].item() == pytest.approx(data["attachment"]["avoidance"] - 0.5)
        motiv_mean = sum(data["motivation"].values()) / 4
        assert emb[7].item() == pytest.approx(motiv_mean - 0.5)

    def test_unsupported_embed_dim_raises(self, md_file: Path) -> None:
        data = parse_persona_markdown(md_file)
        with pytest.raises(PersonaParseError):
            hexaco_to_embedding(data, embed_dim=16)


# ---------- extend_ckpt_with_persona ----------


class TestExtendCkpt:
    def test_embedding_grows_by_one(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "extended.pt"
        summary = extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        assert summary["n_personas_old"] == 3
        assert summary["n_personas_new"] == 4
        assert summary["persona_idx"] == 3

        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        emb = new_ckpt["state_model"]["personalization.embedding.weight"]
        assert emb.shape == (4, 8)

    def test_persona_to_idx_extended(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "extended.pt"
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert new_ckpt["persona_to_idx"]["vboss_self"] == 3
        # 旧 persona 不动
        assert new_ckpt["persona_to_idx"]["a"] == 0

    def test_config_n_personas_updated(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "extended.pt"
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert new_ckpt["config"].n_personas == 4
        # 其他 config 字段不变
        assert new_ckpt["config"].hidden_dim == 8

    def test_base_embedding_rows_unchanged(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        """扩展不能改变原 30/3 行 embedding（base 训练成果保留）。"""
        old_ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        old_emb = old_ckpt["state_model"]["personalization.embedding.weight"].clone()
        dst = tmp_path / "extended.pt"
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        new_emb = new_ckpt["state_model"]["personalization.embedding.weight"]
        assert torch.allclose(new_emb[:3], old_emb)

    def test_metadata_records_partial(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "extended.pt"
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        meta = new_ckpt["extension_metadata"]
        assert meta["extended_persona_id"] == "vboss_self"
        assert meta["extended_idx"] == 3
        assert "caps_if_then_patterns_empty" in meta["partial_fields"]

    def test_persona_id_mismatch_raises(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "extended.pt"
        with pytest.raises(PersonaParseError):
            extend_ckpt_with_persona(
                src_ckpt, dst,
                persona_id="wrong_id",
                persona_md_path=md_file,
            )

    def test_persona_already_exists(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        # 先 extend 一次成功
        dst = tmp_path / "extended1.pt"
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        # 再用扩展过的 ckpt 当 src，应抛
        dst2 = tmp_path / "extended2.pt"
        with pytest.raises(PersonaParseError, match="already in ckpt"):
            extend_ckpt_with_persona(
                dst, dst2, persona_id="vboss_self", persona_md_path=md_file
            )

    def test_dst_exists_no_overwrite(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "occupied.pt"
        dst.write_bytes(b"existing")
        with pytest.raises(PersonaParseError, match="exists"):
            extend_ckpt_with_persona(
                src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
            )

    def test_dst_exists_with_overwrite_ok(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "occupied.pt"
        dst.write_bytes(b"existing")
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self",
            persona_md_path=md_file, overwrite=True,
        )
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert new_ckpt["persona_to_idx"]["vboss_self"] == 3


# ---------- 端到端 forward ----------


class TestEndToEndForward:
    """扩展后的 ckpt 加载到 StateRecognitionLNN，forward(x, idx=new) 不报错。"""

    def test_forward_with_new_persona_idx(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "extended.pt"
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        config = new_ckpt["config"]

        # 重建 model（n_personas=4）并加载 state_dict
        factory = DefaultLNNFactory()
        state_model = factory.create_state_model(config)
        state_model.load_state_dict(new_ckpt["state_model"])

        # forward with new persona idx=3 (vboss_self)
        x = torch.randn(1, config.sequence_length, config.feature_dim)
        idx = torch.tensor([3], dtype=torch.long)
        logits = state_model(x, idx)
        assert logits.shape == (1, config.sequence_length, config.n_state_classes)
        assert torch.isfinite(logits).all()

    def test_old_persona_idx_still_works(
        self, src_ckpt: Path, md_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "extended.pt"
        extend_ckpt_with_persona(
            src_ckpt, dst, persona_id="vboss_self", persona_md_path=md_file
        )
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        config = new_ckpt["config"]

        factory = DefaultLNNFactory()
        state_model = factory.create_state_model(config)
        state_model.load_state_dict(new_ckpt["state_model"])

        # 老 persona idx=0 (a) 仍 forward 通
        x = torch.randn(2, config.sequence_length, config.feature_dim)
        idx = torch.tensor([0, 1], dtype=torch.long)
        logits = state_model(x, idx)
        assert logits.shape == (2, config.sequence_length, config.n_state_classes)
