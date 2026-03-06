import numpy as np
import torch

from nanotron.data.clm_collator import DataCollatorForSDSPWithPositionIds
from nanotron.models.qwen import _normalize_sdsp_pair_masks


class _FakeTokenizer:
    _ENCODINGS = {
        "<assistant>": [1, 2, 3],
        "</assistant>": [4, 2, 3],
        "\n": [5],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._ENCODINGS[text])

    def convert_ids_to_tokens(self, ids):
        mapping = {
            1: "<",
            2: "assistant",
            3: ">",
            4: "</",
            5: "\n",
        }
        return [mapping.get(token_id, f"tok_{token_id}") for token_id in ids]


class _FakeParallelContext:
    def __init__(self, context_parallel_size: int):
        self.pp_pg = object()
        self.cp_pg = object()
        self.context_parallel_size = context_parallel_size


def test_sdsp_collator_excludes_each_doc_first_text_token_from_pair_mask(monkeypatch):
    monkeypatch.setattr(
        "nanotron.data.clm_collator.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    collator = DataCollatorForSDSPWithPositionIds(
        sequence_length=32,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=None,
        tokenizer_name_or_path="fake",
        feedback_open_tag="<assistant>",
        feedback_close_tag="</assistant>",
        strip_single_newline_after_feedback=True,
        eos_token_id=99,
    )

    # Two packed docs:
    # doc1 text tokens -> 3, so only the last 2 should be SDSP-paired
    # doc2 text tokens -> 4, so only the last 3 should be SDSP-paired
    packed = np.asarray(
        [
            1, 2, 3, 11, 12, 4, 2, 3, 5, 21, 22, 23, 99,
            1, 2, 3, 13, 4, 2, 3, 5, 24, 25, 26, 27, 99,
        ],
        dtype=np.int64,
    )
    positions = np.arange(len(packed), dtype=np.int64)

    result = collator._build_sdsp_example(input_ids=packed, positions=positions)

    assert int(result["base_sdsp_pair_mask"].sum()) == 5
    assert int(result["cond_sdsp_pair_mask"].sum()) == 5


def test_sdsp_collator_fallback_positions_reset_after_eos(monkeypatch):
    monkeypatch.setattr(
        "nanotron.data.clm_collator.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    collator = DataCollatorForSDSPWithPositionIds(
        sequence_length=25,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=_FakeParallelContext(context_parallel_size=1),
        tokenizer_name_or_path="fake",
        feedback_open_tag="<assistant>",
        feedback_close_tag="</assistant>",
        strip_single_newline_after_feedback=True,
        eos_token_id=99,
    )

    # Two docs concatenated with EOS separators, but without explicit `positions`.
    packed = np.asarray(
        [
            1, 2, 3, 11, 12, 4, 2, 3, 5, 21, 22, 23, 99,
            1, 2, 3, 13, 4, 2, 3, 5, 24, 25, 26, 27, 99,
        ],
        dtype=np.int64,
    )

    monkeypatch.setattr("nanotron.data.clm_collator.dist.get_rank", lambda _: 0)
    result = collator([{"input_ids": packed}])
    cond_positions = result["cond_position_ids"][0]

    # After EOS at index 12, next token index 13 should start a new document at position 0.
    assert cond_positions[13] == 0


def test_sdsp_collator_keeps_global_positions_for_cp(monkeypatch):
    monkeypatch.setattr(
        "nanotron.data.clm_collator.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    collator = DataCollatorForSDSPWithPositionIds(
        sequence_length=8,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=_FakeParallelContext(context_parallel_size=2),
        tokenizer_name_or_path="fake",
        feedback_open_tag="<assistant>",
        feedback_close_tag="</assistant>",
        strip_single_newline_after_feedback=True,
        eos_token_id=99,
    )

    # Example with no SDSP tags: this still exercises CP slicing behavior.
    sample = np.asarray([10, 11, 12, 13, 14, 15, 16, 17, 99], dtype=np.int64)

    monkeypatch.setattr("nanotron.data.clm_collator.dist.get_rank", lambda _: 0)
    result = collator([{"input_ids": sample}])

    # Inputs are CP-local (sequence_length / cp_size), positions stay global.
    assert result["base_input_ids"].shape == (1, 4)
    assert result["cond_input_ids"].shape == (1, 4)
    assert result["base_position_ids"].shape == (1, 8)
    assert result["cond_position_ids"].shape == (1, 8)


def test_sdsp_pair_mask_normalization_keeps_order_aligned_masks():
    base_label_mask = torch.ones((1, 6), dtype=torch.bool)
    cond_label_mask = torch.ones((1, 6), dtype=torch.bool)
    base_sdsp_pair_mask = torch.tensor([[True, True, True, False, False, False]])
    cond_sdsp_pair_mask = torch.tensor([[False, False, False, True, True, True]])

    normalized_base, normalized_cond = _normalize_sdsp_pair_masks(
        base_sdsp_pair_mask=base_sdsp_pair_mask,
        cond_sdsp_pair_mask=cond_sdsp_pair_mask,
        base_label_mask=base_label_mask,
        cond_label_mask=cond_label_mask,
    )

    assert torch.equal(normalized_base, base_sdsp_pair_mask)
    assert torch.equal(normalized_cond, cond_sdsp_pair_mask)
