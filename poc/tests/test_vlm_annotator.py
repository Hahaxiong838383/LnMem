"""Tests for VLMAnnotator."""
import base64
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root on path
sys.path.insert(0, ".")

from poc.src.vision.vlm_annotator import VLMAnnotator, VLMLabel, _parse_vlm_json


class TestParseVlmJson(unittest.TestCase):
    def test_clean_json(self):
        raw = '{"au_description":"AU04皱眉","valence":-0.3,"arousal":0.6,"confidence":0.8}'
        label = _parse_vlm_json(raw, "test-model")
        self.assertIsNotNone(label)
        self.assertEqual(label.au_description, "AU04皱眉")
        self.assertAlmostEqual(label.valence, -0.3)
        self.assertEqual(label.model_used, "test-model")

    def test_markdown_wrapped(self):
        raw = '```json\n{"au_description":"smile","valence":0.7,"arousal":0.4,"confidence":0.9}\n```'
        label = _parse_vlm_json(raw, "m")
        self.assertIsNotNone(label)
        self.assertAlmostEqual(label.valence, 0.7)

    def test_empty_string(self):
        self.assertIsNone(_parse_vlm_json("", "m"))

    def test_no_json(self):
        self.assertIsNone(_parse_vlm_json("sorry I cannot analyze", "m"))

    def test_partial_json_missing_fields(self):
        raw = '{"au_description":"neutral"}'
        label = _parse_vlm_json(raw, "m")
        self.assertIsNotNone(label)
        self.assertAlmostEqual(label.valence, 0.0)   # default
        self.assertAlmostEqual(label.arousal, 0.5)   # default
        self.assertAlmostEqual(label.confidence, 0.7) # default

    def test_with_think_tag_stripped(self):
        raw = '<think>long reasoning...</think>{"au_description":"AU12","valence":0.5,"arousal":0.3,"confidence":0.85}'
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        label = _parse_vlm_json(cleaned, "m")
        self.assertIsNotNone(label)
        self.assertAlmostEqual(label.valence, 0.5)


class TestVLMLabel(unittest.TestCase):
    def test_immutable(self):
        label = VLMLabel("AU04", -0.3, 0.6, 0.8, "model")
        with self.assertRaises((AttributeError, TypeError)):
            label.valence = 0.0  # type: ignore

    def test_to_dict_rounds(self):
        label = VLMLabel("AU04", -0.3333, 0.6666, 0.88888, "model")
        d = label.to_dict()
        self.assertEqual(d["valence"], -0.333)
        self.assertEqual(d["arousal"], 0.667)
        self.assertEqual(d["confidence"], 0.889)


class TestVLMAnnotatorChain(unittest.TestCase):
    def _make_annotator(self):
        return VLMAnnotator(proxy_key="fake-proxy", minimax_key="fake-mm", timeout_s=5)

    def test_build_chain_default(self):
        a = self._make_annotator()
        chain = a._build_chain()
        self.assertEqual(chain[0][0], "gemini-3.1-flash-lite-preview")
        self.assertEqual(chain[1][0], "gemini-3.1-pro-preview")
        self.assertEqual(chain[2][0], "gemini-2.5-flash")

    def test_build_chain_override(self):
        a = VLMAnnotator(proxy_key="k", minimax_key="k", provider_override="gemini-2.5-flash")
        chain = a._build_chain()
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0][0], "gemini-2.5-flash")

    def test_annotate_returns_none_on_all_failure(self):
        a = self._make_annotator()
        # Patch _call to always return None
        a._call = MagicMock(return_value=None)
        a._encode_frame = MagicMock(return_value="fakeb64")
        result = a.annotate("fake_frame")
        self.assertIsNone(result)

    def test_annotate_uses_first_working_model(self):
        a = self._make_annotator()
        good_json = '{"au_description":"AU12","valence":0.5,"arousal":0.3,"confidence":0.9}'
        call_count = 0

        def fake_call(model_id, api_type, img_b64):
            nonlocal call_count
            call_count += 1
            if model_id == "gemini-3.1-flash-lite-preview":
                return None  # first fails
            return good_json  # second succeeds

        a._call = fake_call
        a._encode_frame = MagicMock(return_value="fakeb64")
        result = a.annotate("fake")
        self.assertIsNotNone(result)
        self.assertEqual(result.model_used, "gemini-3.1-pro-preview")
        self.assertEqual(call_count, 2)  # stopped after second

    def test_encode_frame_file_path(self):
        a = self._make_annotator()
        import tempfile, os
        # Write a minimal JPEG
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # fake JPEG header
            path = f.name
        try:
            b64 = a._encode_frame(path)
            self.assertIsNotNone(b64)
        finally:
            os.unlink(path)

    def test_encode_frame_none(self):
        a = self._make_annotator()
        self.assertIsNone(a._encode_frame(None))

    def test_call_minimax_uses_vlm_endpoint(self):
        """MiniMax VLM 必须走 /v1/coding_plan/vlm，不是 chat/completions。"""
        import poc.src.vision.vlm_annotator as mod
        orig = mod._post_json
        calls = []

        def fake_post(url, headers, body, timeout):
            calls.append({"url": url, "body": body})
            assert "coding_plan/vlm" in url, f"Wrong endpoint: {url}"
            assert "prompt" in body, "Should use 'prompt' key not 'messages'"
            assert "image_url" in body, "Should have 'image_url' key"
            return 200, {"content": '{"au_description":"AU4","valence":-0.3,"arousal":0.5,"confidence":0.8}',
                         "base_resp": {"status_code": 0}}

        mod._post_json = fake_post
        try:
            result = self._make_annotator()._call_minimax("fakeb64")
            self.assertIsNotNone(result)
            self.assertEqual(len(calls), 1)
        finally:
            mod._post_json = orig

    def test_call_minimax_base_resp_error(self):
        """Non-zero base_resp.status_code 应返回 None。"""
        import poc.src.vision.vlm_annotator as mod
        orig = mod._post_json
        mod._post_json = lambda *a, **k: (200, {"content": "text",
                                                "base_resp": {"status_code": 1001}})
        try:
            self.assertIsNone(self._make_annotator()._call_minimax("fakeb64"))
        finally:
            mod._post_json = orig


if __name__ == "__main__":
    unittest.main()

    def test_call_minimax_uses_vlm_endpoint(self):
        """MiniMax VLM 必须走 /v1/coding_plan/vlm，不是 chat/completions。"""
        import poc.src.vision.vlm_annotator as mod
        orig = mod._post_json
        calls = []

        def fake_post(url, headers, body, timeout):
            calls.append({"url": url, "body": body})
            assert "coding_plan/vlm" in url, f"Wrong endpoint: {url}"
            assert "prompt" in body, "Should use 'prompt' key"
            assert "image_url" in body, "Should have 'image_url' key"
            return 200, {"content": '{"au_description":"AU4","valence":-0.3,"arousal":0.5,"confidence":0.8}',
                         "base_resp": {"status_code": 0}}

        mod._post_json = fake_post
        a = self.__class__._make_annotator(self)
        try:
            result = a._call_minimax("fakeb64")
            self.assertIsNotNone(result)
            self.assertEqual(len(calls), 1)
        finally:
            mod._post_json = orig

    def test_call_minimax_base_resp_error(self):
        """Non-zero base_resp.status_code 应返回 None。"""
        import poc.src.vision.vlm_annotator as mod
        orig = mod._post_json
        mod._post_json = lambda *a, **k: (200, {"content": "text",
                                                "base_resp": {"status_code": 1001}})
        ann = self.__class__._make_annotator(self)
        try:
            self.assertIsNone(ann._call_minimax("fakeb64"))
        finally:
            mod._post_json = orig

