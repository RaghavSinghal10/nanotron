import dataclasses
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from nanotron import distributed as dist
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


@dataclasses.dataclass
class DataCollatorForCLM:
    """
    Data collator used for causal language modeling.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    # torch vs numpy
    use_numpy: bool = True

    @torch.profiler.record_function("DataCollatorForCLM.__call__")
    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:

        vstack = np.vstack if self.use_numpy else torch.vstack
        ones = np.ones if self.use_numpy else torch.ones
        bool_dtype = np.bool_ if self.use_numpy else torch.bool

        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "sample_dataset_index": TensorPointer(group_rank=self.output_pp_rank),
            }

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, torch.LongTensor, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
        result["sample_dataset_index"] = TensorPointer(group_rank=self.output_pp_rank)

        assert (
            expanded_input_length == self.sequence_length + 1
        ), f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        # Process inputs: last token is the label
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["input_mask"] = ones((batch_size, self.sequence_length), dtype=bool_dtype)

            # Context Parallelism: Each CP rank gets a slice of the input_ids and input_mask
            cp_rank, cp_size = dist.get_rank(self.parallel_context.cp_pg), self.parallel_context.context_parallel_size
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size, (cp_rank + 1) * self.sequence_length // cp_size
            )
            result["input_ids"] = result["input_ids"][:, local_slice]  # (b, s/cp_size)
            result["input_mask"] = result["input_mask"][:, local_slice]  # (b, s/cp_size)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            sample_dataset_indices = np.asarray(
                [int(examples[i].get("_dataset_index", -1)) for i in range(len(examples))],
                dtype=np.int64,
            )

            # Create label mask based on position_ids
            if "positions" in examples[0]:
                # Get position_ids for the labels (shifted right by 1 to align with label_ids)
                position_ids = np.vstack([examples[i]["positions"] for i in range(len(examples))])
                position_ids = position_ids[:, 1:]  # Shift right to align with labels

                # Create mask: True for all tokens except the one before position_id == 0
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

                # Find where position_ids is 0
                zeros = position_ids == 0
                # Mask the current token where we found zeros (since labels are already shifted right)
                result["label_mask"] &= ~zeros
            else:
                # Default: all tokens are used for loss
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            # Context Parallelism: Each CP rank gets a slice of the label_ids and label_mask
            cp_rank, cp_size = dist.get_rank(self.parallel_context.cp_pg), self.parallel_context.context_parallel_size
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size, (cp_rank + 1) * self.sequence_length // cp_size
            )
            result["label_ids"] = result["label_ids"][:, local_slice]  # (b, s/cp_size)
            result["label_mask"] = result["label_mask"][:, local_slice]  # (b, s/cp_size)
            result["sample_dataset_index"] = sample_dataset_indices

        if (
            not isinstance(result["input_ids"], TensorPointer)
            and result["input_ids"].shape[-1] != self.sequence_length // cp_size
        ):
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length // cp_size}."
            )
        if (
            not isinstance(result["label_ids"], TensorPointer)
            and result["label_ids"].shape[-1] != self.sequence_length // cp_size
        ):
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length // cp_size}."
            )

        # # Maybe cast np.array to torch.Tensor
        # result = {
        #     k: v if isinstance(v, TensorPointer) else (torch.from_numpy(v).contiguous() if self.use_numpy else v)
        #     for k, v in result.items()
        # }  # TODO: @nouamane in case of memory issues, try keeping numpy here.
        # # assert contiguous
        # for k, v in result.items():
        #     if not isinstance(v, TensorPointer):
        #         assert v.is_contiguous(), f"{k} is not contiguous"
        #         assert not v.is_cuda, f"{k} is in cuda. Bad for pinning memory"
        return result


@dataclasses.dataclass
class DataCollatorForCLMWithPositionIds:
    """
    Data collator used for causal language modeling with position IDs.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    use_doc_masking: bool = True
    cp_return_global_position_ids: bool = True
    eos_token_id: Optional[int] = None
    split_feedback_loss_logging: bool = False
    tokenizer_name_or_path: Optional[str] = None
    feedback_open_tag: str = "<assistant>"
    feedback_close_tag: str = "</assistant>"

    def __post_init__(self):
        self._feedback_tokenizer = None
        self._open_first_piece = None
        self._assistant_piece = None
        self._open_last_piece_prefix = None
        self._close_last_piece_prefix = None
        if not self.split_feedback_loss_logging:
            return

        if self.tokenizer_name_or_path is None:
            raise ValueError("tokenizer_name_or_path must be provided when split_feedback_loss_logging is enabled")

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        open_pieces = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(self.feedback_open_tag, add_special_tokens=False)
        )
        close_pieces = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(self.feedback_close_tag, add_special_tokens=False)
        )
        if len(open_pieces) < 3 or len(close_pieces) < 3:
            raise ValueError(
                "Feedback open/close tags tokenized unexpectedly; "
                f"open={self.feedback_open_tag!r}, close={self.feedback_close_tag!r}"
            )
        self._feedback_tokenizer = tokenizer
        self._open_first_piece = open_pieces[0]
        self._assistant_piece = open_pieces[1]
        self._open_last_piece_prefix = open_pieces[2]
        self._close_last_piece_prefix = close_pieces[-1]

    def _build_fallback_positions(self, input_ids: np.ndarray) -> np.ndarray:
        # Reconstruct per-document position resets from EOS when explicit positions are absent.
        if self.eos_token_id is None:
            return np.arange(input_ids.shape[0], dtype=np.int64)

        positions = np.empty(input_ids.shape[0], dtype=np.int64)
        cursor = 0
        for idx, token_id in enumerate(input_ids.tolist()):
            positions[idx] = cursor
            if token_id == self.eos_token_id:
                cursor = 0
            else:
                cursor += 1
        return positions

    def _find_feedback_span(self, pieces: List[str]) -> Optional[Tuple[int, int]]:
        if self._open_first_piece is None:
            return None

        open_start = None
        for idx in range(0, len(pieces) - 2):
            if (
                pieces[idx] == self._open_first_piece
                and pieces[idx + 1] == self._assistant_piece
                and pieces[idx + 2].startswith(self._open_last_piece_prefix)
            ):
                open_start = idx
                break
        if open_start is None:
            return None

        for idx in range(open_start + 2, len(pieces) - 2):
            if (
                "</" in pieces[idx]
                and pieces[idx + 1] == self._assistant_piece
                and pieces[idx + 2].startswith(self._close_last_piece_prefix)
            ):
                return open_start, idx + 3
        return None

    def _build_feedback_token_types(self, input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Returns per-token type ids:
          0: ignored (e.g., EOS/padding)
          1: text token
          2: reflection token (inside <assistant>...</assistant>)
        """
        token_ids = input_ids.astype(np.int64).tolist()
        position_ids = positions.astype(np.int64).tolist()
        token_types = np.zeros(len(token_ids), dtype=np.uint8)

        doc_starts = [idx for idx, pos in enumerate(position_ids) if pos == 0]
        if self.eos_token_id is not None and len(doc_starts) <= 1:
            eos_doc_starts = [0]
            for idx, tok in enumerate(token_ids[:-1]):
                if tok == self.eos_token_id:
                    eos_doc_starts.append(idx + 1)
            doc_starts = sorted({idx for idx in eos_doc_starts if 0 <= idx < len(token_ids)})

        segments: List[Tuple[int, int]] = []
        if not doc_starts:
            segments.append((0, len(token_ids)))
        else:
            if doc_starts[0] != 0:
                segments.append((0, doc_starts[0]))
            for idx, start in enumerate(doc_starts):
                end = doc_starts[idx + 1] if idx + 1 < len(doc_starts) else len(token_ids)
                segments.append((start, end))

        for start, end in segments:
            doc_tokens = token_ids[start:end]
            if len(doc_tokens) == 0:
                continue

            for rel_idx, tok in enumerate(doc_tokens):
                if self.eos_token_id is not None and tok == self.eos_token_id:
                    continue
                token_types[start + rel_idx] = 1

            if self._feedback_tokenizer is None:
                continue

            pieces = self._feedback_tokenizer.convert_ids_to_tokens(doc_tokens)
            feedback_span = self._find_feedback_span(pieces)
            if feedback_span is None:
                continue

            reflection_start, reflection_end = feedback_span
            for rel_idx in range(reflection_start, min(reflection_end, len(doc_tokens))):
                tok = doc_tokens[rel_idx]
                if self.eos_token_id is not None and tok == self.eos_token_id:
                    continue
                token_types[start + rel_idx] = 2

        return token_types

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [self.input_pp_rank, self.output_pp_rank]:
            assert all(len(example) == 0 for example in examples)
            result = {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "positions": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "sample_dataset_index": TensorPointer(group_rank=self.output_pp_rank),
            }
            if self.split_feedback_loss_logging:
                result["text_label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
                result["reflection_label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
            return result

        # input_ids[0,:20]
        # array([  198,    50,    30, 12532,  3589,   198,    51,    30, 30618,
        #         198,    52,    30,  8279, 11274,   198, 21350,    42,   340,
        #         0,  1780])
        # position_ids[0,:20]
        # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        #        17, 18,  0])
        # result["label_ids"][0,:20]
        # array([   50,    30, 12532,  3589,   198,    51,    30, 30618,   198,
        #         52,    30,  8279, 11274,   198, 21350,    42,   340,     0,
        #         1780,   314])
        # -> label_id for 0 is 1780 -> need to mask 1780
        # result["label_mask"][0,:20]
        # array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        #         True,  True,  True,  True,  True,  True,  True,  True,  True,
        #     False,  True])

        # document starts with first token, and last token is eos_token (0)
        # label_mask should be 1 for all tokens except the last one

        # Stack input_ids
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape
        has_explicit_positions = "positions" in examples[0]
        if self.use_doc_masking and has_explicit_positions:
            position_ids = np.vstack([examples[i]["positions"] for i in range(len(examples))])
        elif self.use_doc_masking:
            position_ids = np.vstack([self._build_fallback_positions(examples[i]["input_ids"]) for i in range(len(examples))])
        else:
            position_ids = None

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        # Initialize all fields as TensorPointers
        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["position_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
        result["sample_dataset_index"] = TensorPointer(group_rank=self.output_pp_rank)
        if self.split_feedback_loss_logging:
            result["text_label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
            result["reflection_label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert expanded_input_length == self.sequence_length + 1, (
            f"Samples should be of length {self.sequence_length + 1} (seq_len+1), " f"but got {expanded_input_length}"
        )

        cp_rank, cp_size = dist.get_rank(self.parallel_context.cp_pg), self.parallel_context.context_parallel_size
        local_slice = slice(cp_rank * self.sequence_length // cp_size, (cp_rank + 1) * self.sequence_length // cp_size)

        token_type_ids = None
        if self.split_feedback_loss_logging:
            if position_ids is None:
                full_positions = np.arange(expanded_input_length, dtype=np.int64)[None, :].repeat(batch_size, axis=0)
            else:
                full_positions = position_ids
            token_type_ids = np.vstack(
                [
                    self._build_feedback_token_types(input_ids=input_ids[i], positions=full_positions[i])
                    for i in range(batch_size)
                ]
            )

        # Process inputs
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]

            if position_ids is not None:
                # Simply drop the last position ID for each example
                result["positions"] = position_ids[:, :-1]
            else:
                # Default: sequential position ids
                result["positions"] = np.arange(self.sequence_length)[None, :].repeat(batch_size, axis=0)

            # Context Parallelism: Each CP rank gets a slice of the input_ids and position_ids
            result["input_ids"] = result["input_ids"][:, local_slice]  # (b, s/cp_size)
            if not self.cp_return_global_position_ids:
                result["positions"] = result["positions"][:, local_slice]  # (b, s/cp_size)
            result["position_ids"] = result.pop("positions")

        # Process labels
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            sample_dataset_indices = np.asarray(
                [int(examples[i].get("_dataset_index", -1)) for i in range(len(examples))],
                dtype=np.int64,
            )

            # Create label mask based on position_ids
            if position_ids is not None:
                # Get position_ids for the labels (shifted right by 1 to align with label_ids)
                label_position_ids = position_ids[:, 1:]  # Shift right to align with labels

                # Create mask: True for all tokens except the one before position_id == 0
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

                # Find where position_ids is 0
                zeros = label_position_ids == 0
                # Mask the current token where we found zeros (since labels are already shifted right)
                result["label_mask"] &= ~zeros
            else:
                # Default: all tokens are used for loss
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            if self.split_feedback_loss_logging:
                assert token_type_ids is not None
                label_token_types = token_type_ids[:, 1:]
                result["text_label_mask"] = (label_token_types == 1) & result["label_mask"]
                result["reflection_label_mask"] = (label_token_types == 2) & result["label_mask"]

            # Context Parallelism: Each CP rank gets a slice of the label_ids and label_mask
            result["label_ids"] = result["label_ids"][:, local_slice]  # (b, s/cp_size)
            result["label_mask"] = result["label_mask"][:, local_slice]  # (b, s/cp_size)
            if self.split_feedback_loss_logging:
                result["text_label_mask"] = result["text_label_mask"][:, local_slice]
                result["reflection_label_mask"] = result["reflection_label_mask"][:, local_slice]
            result["sample_dataset_index"] = sample_dataset_indices

        # Validate shapes
        if (
            isinstance(result["input_ids"], torch.Tensor)
            and result["input_ids"].shape[-1] != self.sequence_length // cp_size
        ):
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. Length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length // cp_size}."
            )
        if (
            isinstance(result["label_ids"], torch.Tensor)
            and result["label_ids"].shape[-1] != result["input_ids"].shape[-1]
        ):
            raise ValueError(
                f"`label_ids` are incorrectly preprocessed. Length is {result['label_ids'].shape[-1]}, but should be"
                f" {result['input_ids'].shape[-1]}."
            )

        # # Cast np.array to torch.Tensor
        # result = {
        #     k: v if isinstance(v, TensorPointer) else torch.from_numpy(v).contiguous() for k, v in result.items()
        # }

        # # assert contiguous
        # for k, v in result.items():
        #     if not isinstance(v, TensorPointer):
        #         assert v.is_contiguous(), f"{k} is not contiguous"
        #         assert not v.is_cuda, f"{k} is in cuda. Bad for pinning memory"

        return result


@dataclasses.dataclass
class DataCollatorForSDSPWithPositionIds:
    """
    Data collator for SDSP-style pretraining.

    It emits two aligned autoregressive views per sample:
    - `cond_*`: the original feedback + text sequence
    - `base_*`: a text-only sequence obtained by removing the configured feedback prefix
      (`assistant_xml` tags or `reflection_newline` prefix)

    `*_sdsp_pair_mask` marks aligned text targets that can participate in the SDSP correction.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    tokenizer_name_or_path: str
    feedback_format: str = "assistant_xml"
    feedback_open_tag: str = "<assistant>"
    feedback_close_tag: str = "</assistant>"
    strip_single_newline_after_feedback: bool = True
    eos_token_id: Optional[int] = None
    cp_return_global_position_ids: bool = True

    def __post_init__(self):
        self._get_tokenizer()
        self._sdsp_enabled_dataset_indices = self._parse_sdsp_enabled_dataset_indices()

    def _parse_sdsp_enabled_dataset_indices(self) -> Optional[Tuple[int, ...]]:
        raw = os.environ.get("SDSP_ENABLE_DATASET_INDICES", "").strip()
        if raw == "":
            return None
        indices: List[int] = []
        for part in raw.split(","):
            token = part.strip()
            if token == "":
                continue
            try:
                indices.append(int(token))
            except ValueError as exc:
                raise ValueError(
                    "Invalid SDSP_ENABLE_DATASET_INDICES value. "
                    f"Expected comma-separated integers, got: {raw!r}"
                ) from exc
        if not indices:
            return tuple()
        # Preserve order while deduplicating.
        deduped = tuple(dict.fromkeys(indices))
        return deduped

    def _sample_allows_sdsp(self, dataset_index: Optional[int]) -> bool:
        # Backward-compatible default: if gating is not configured, keep previous behavior.
        if self._sdsp_enabled_dataset_indices is None:
            return True
        # If gating is configured but source index is unavailable, do not gate to avoid
        # changing behavior in non-blended/single-dataset pipelines.
        if dataset_index is None:
            return True
        return int(dataset_index) in self._sdsp_enabled_dataset_indices

    def _get_tokenizer(self):
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_tokens) == 0:
            raise ValueError("Newline tokenization unexpectedly returned no ids")

        if self.feedback_format == "assistant_xml":
            open_pieces = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(self.feedback_open_tag, add_special_tokens=False)
            )
            close_pieces = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(self.feedback_close_tag, add_special_tokens=False)
            )

            if len(open_pieces) < 3:
                raise ValueError(
                    f"Feedback open tag {self.feedback_open_tag!r} tokenized unexpectedly: {open_pieces}"
                )
            if len(close_pieces) < 3:
                raise ValueError(
                    f"Feedback close tag {self.feedback_close_tag!r} tokenized unexpectedly: {close_pieces}"
                )

            self._open_first_piece = open_pieces[0]
            self._assistant_piece = open_pieces[1]
            self._open_last_piece_prefix = open_pieces[2]
            self._close_last_piece_prefix = close_pieces[-1]
        elif self.feedback_format != "reflection_newline":
            raise ValueError(
                f"Unsupported SDSP feedback format: {self.feedback_format!r}. "
                "Expected one of {'assistant_xml', 'reflection_newline'}."
            )

        self._tokenizer = tokenizer
        self._newline_token_ids = tuple(int(tok) for tok in newline_tokens)
        self._newline_token_id = newline_tokens[0] if len(newline_tokens) == 1 else None
        self._newline_token_contains_cache: Dict[int, bool] = {}
        return tokenizer

    def _has_feedback_prefix(self, pieces: List[str]) -> bool:
        return (
            len(pieces) >= 3
            and pieces[0] == self._open_first_piece
            and pieces[1] == self._assistant_piece
            and pieces[2].startswith(self._open_last_piece_prefix)
        )

    def _find_feedback_end(self, pieces: List[str]) -> Optional[int]:
        for idx in range(2, len(pieces) - 2):
            if (
                "</" in pieces[idx]
                and pieces[idx + 1] == self._assistant_piece
                and pieces[idx + 2].startswith(self._close_last_piece_prefix)
            ):
                return idx + 2
        return None

    def _token_contains_newline(
        self,
        token_id: int,
        tokenizer,
        token_piece: Optional[str] = None,
    ) -> bool:
        cached = self._newline_token_contains_cache.get(int(token_id))
        if cached is not None:
            return cached

        if token_piece is None:
            token_piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]

        # Common textual forms for newline in token pieces:
        # - literal "\n"
        # - GPT2/byte-BPE marker "Ċ"
        # - byte fallback marker "<0x0A>"
        contains_newline = (
            ("\n" in token_piece)
            or ("Ċ" in token_piece)
            or ("<0x0A>" in token_piece)
        )

        if not contains_newline:
            # Robust fallback across tokenizers where newline can be embedded in
            # merged tokens (e.g. punctuation+newline).
            try:
                decoded_piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
            except TypeError:
                decoded_piece = tokenizer.decode([int(token_id)])
            contains_newline = "\n" in decoded_piece

        self._newline_token_contains_cache[int(token_id)] = contains_newline
        return contains_newline

    def _find_first_newline_end(self, token_ids: List[int], tokenizer) -> Optional[int]:
        newline = self._newline_token_ids
        width = len(newline)
        for idx in range(0, len(token_ids) - width + 1):
            if tuple(token_ids[idx : idx + width]) == newline:
                return idx + width - 1

        # Fallback for merged newline pieces where encode("\n") does not appear
        # as a standalone id sequence (e.g. token ".Ċ").
        token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
        for idx, (token_id, token_piece) in enumerate(zip(token_ids, token_pieces)):
            if self._token_contains_newline(token_id=token_id, tokenizer=tokenizer, token_piece=token_piece):
                return idx
        return None

    def _detect_text_start(self, doc_tokens: List[int], tokenizer) -> tuple[bool, int]:
        if self.feedback_format == "assistant_xml":
            pieces = tokenizer.convert_ids_to_tokens(doc_tokens)
            if not self._has_feedback_prefix(pieces):
                return False, 0
            feedback_end = self._find_feedback_end(pieces)
            if feedback_end is None:
                return False, 0
            text_start = feedback_end + 1
            if (
                self.strip_single_newline_after_feedback
                and self._newline_token_id is not None
                and text_start < len(doc_tokens)
                and doc_tokens[text_start] == self._newline_token_id
            ):
                text_start += 1
            return True, text_start

        newline_end = self._find_first_newline_end(doc_tokens, tokenizer=tokenizer)
        if newline_end is None:
            return False, 0
        text_start = newline_end + 1
        if text_start >= len(doc_tokens):
            return False, 0
        return True, text_start

    def _build_fallback_positions(self, input_ids: np.ndarray) -> np.ndarray:
        # Reconstruct packed-document position resets when explicit positions are absent.
        # If EOS is unavailable, fall back to a monotonic arange.
        if self.eos_token_id is None:
            return np.arange(input_ids.shape[0], dtype=np.int64)

        positions = np.empty(input_ids.shape[0], dtype=np.int64)
        cursor = 0
        for idx, token_id in enumerate(input_ids.tolist()):
            positions[idx] = cursor
            if token_id == self.eos_token_id:
                cursor = 0
            else:
                cursor += 1
        return positions

    def _build_sdsp_example(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        dataset_index: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        tokenizer = self._get_tokenizer()
        token_ids = input_ids.astype(np.int64).tolist()
        position_ids = positions.astype(np.int64).tolist()
        sample_allows_sdsp = self._sample_allows_sdsp(dataset_index=dataset_index)

        doc_starts = [idx for idx, pos in enumerate(position_ids) if pos == 0]
        # Some dataloader variants do not emit per-document position resets (positions are arange fallback).
        # In that case, recover document boundaries from EOS delimiters so SDSP pairing can still trigger.
        if self.eos_token_id is not None and len(doc_starts) <= 1:
            eos_doc_starts = [0]
            for idx, tok in enumerate(token_ids[:-1]):
                if tok == self.eos_token_id:
                    eos_doc_starts.append(idx + 1)
            # Keep only valid starts and preserve order/uniqueness.
            eos_doc_starts = sorted({idx for idx in eos_doc_starts if 0 <= idx < len(token_ids)})
            if len(eos_doc_starts) > len(doc_starts):
                doc_starts = eos_doc_starts
        segments: List[tuple[int, int, bool]] = []
        if not doc_starts:
            segments.append((0, len(token_ids), False))
        else:
            if doc_starts[0] != 0:
                segments.append((0, doc_starts[0], False))
            for idx, start in enumerate(doc_starts):
                end = doc_starts[idx + 1] if idx + 1 < len(doc_starts) else len(token_ids)
                segments.append((start, end, True))

        # 0 = ignored, 1 = base CE only, 2 = SDSP-paired text token
        cond_kinds = [0] * len(token_ids)
        base_tokens: List[int] = []
        base_positions: List[int] = []
        base_kinds: List[int] = []

        for start, end, starts_at_doc_boundary in segments:
            doc_tokens = token_ids[start:end]
            # Keep document EOS trainable in base CE, matching regular collator behavior.
            doc_kinds = [1] * len(doc_tokens)

            paired = False
            text_start = 0
            if starts_at_doc_boundary and doc_tokens:
                paired, text_start = self._detect_text_start(doc_tokens=doc_tokens, tokenizer=tokenizer)
                if not sample_allows_sdsp:
                    paired = False

            if paired:
                for rel_idx in range(text_start + 1, len(doc_tokens)):
                    if self.eos_token_id is not None and doc_tokens[rel_idx] == self.eos_token_id:
                        continue
                    cond_kinds[start + rel_idx] = 2

                doc_tokens = doc_tokens[text_start:]
                # The first text token stays in the base CE stream but is not SDSP-paired:
                # the conditioned branch predicts it with feedback in context, while the base
                # branch predicts it without that prefix.
                doc_kinds = []
                for rel_idx, tok in enumerate(doc_tokens):
                    if self.eos_token_id is not None and tok == self.eos_token_id:
                        # EOS remains base-only CE and is not SDSP-paired.
                        doc_kinds.append(1)
                    elif rel_idx == 0:
                        doc_kinds.append(1)
                    else:
                        doc_kinds.append(2)

            base_tokens.extend(doc_tokens)
            base_positions.extend(range(len(doc_tokens)))
            base_kinds.extend(doc_kinds)

        target_length = len(token_ids)
        pad_token_id = self.eos_token_id if self.eos_token_id is not None else token_ids[-1]
        if len(base_tokens) < target_length:
            pad_count = target_length - len(base_tokens)
            base_tokens.extend([pad_token_id] * pad_count)
            base_positions.extend([0] * pad_count)
            base_kinds.extend([0] * pad_count)
        else:
            base_tokens = base_tokens[:target_length]
            base_positions = base_positions[:target_length]
            base_kinds = base_kinds[:target_length]

        base_label_mask = np.asarray([kind in (1, 2) for kind in base_kinds[1:]], dtype=np.bool_)
        # Match regular CLM masking semantics: do not train the first token at each
        # (shifted) document boundary where position_ids reset to 0.
        base_label_positions = np.asarray(base_positions[1:], dtype=np.int64)
        base_label_mask &= base_label_positions != 0
        base_sdsp_pair_mask = np.asarray([kind == 2 for kind in base_kinds[1:]], dtype=np.bool_)
        cond_label_mask = np.asarray([kind == 2 for kind in cond_kinds[1:]], dtype=np.bool_)
        cond_sdsp_pair_mask = cond_label_mask.copy()

        if int(base_sdsp_pair_mask.sum()) != int(cond_sdsp_pair_mask.sum()):
            # When document boundaries are inferred (e.g., from EOS), slight drift can happen.
            # Preserve sequential alignment by trimming the longer side instead of intersecting
            # same absolute positions, since the base and conditioned branches live on different
            # token indices after feedback removal.
            pair_count = min(int(base_sdsp_pair_mask.sum()), int(cond_sdsp_pair_mask.sum()))
            base_pair_indices = np.flatnonzero(base_sdsp_pair_mask)[:pair_count]
            cond_pair_indices = np.flatnonzero(cond_sdsp_pair_mask)[:pair_count]

            base_sdsp_pair_mask = np.zeros_like(base_sdsp_pair_mask)
            cond_sdsp_pair_mask = np.zeros_like(cond_sdsp_pair_mask)
            base_sdsp_pair_mask[base_pair_indices] = True
            cond_sdsp_pair_mask[cond_pair_indices] = True

        return {
            "base_input_ids": np.asarray(base_tokens[:-1], dtype=np.int64),
            "base_position_ids": np.asarray(base_positions[:-1], dtype=np.int64),
            "base_label_ids": np.asarray(base_tokens[1:], dtype=np.int64),
            "base_label_mask": base_label_mask,
            "base_sdsp_pair_mask": base_sdsp_pair_mask,
            "cond_input_ids": input_ids[:-1].astype(np.int64, copy=False),
            "cond_position_ids": positions[:-1].astype(np.int64, copy=False),
            "cond_label_ids": input_ids[1:].astype(np.int64, copy=False),
            "cond_label_mask": cond_label_mask,
            "cond_sdsp_pair_mask": cond_sdsp_pair_mask,
        }

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        cp_rank = dist.get_rank(self.parallel_context.cp_pg)
        cp_size = self.parallel_context.context_parallel_size
        local_slice = slice(
            cp_rank * self.sequence_length // cp_size,
            (cp_rank + 1) * self.sequence_length // cp_size,
        )

        if current_pp_rank not in [self.input_pp_rank, self.output_pp_rank]:
            assert all(len(example) == 0 for example in examples)
            return {
                "base_input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "base_position_ids": TensorPointer(group_rank=self.input_pp_rank),
                "cond_input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "cond_position_ids": TensorPointer(group_rank=self.input_pp_rank),
                "base_label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "base_label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "base_sdsp_pair_mask": TensorPointer(group_rank=self.output_pp_rank),
                "cond_label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "cond_label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "cond_sdsp_pair_mask": TensorPointer(group_rank=self.output_pp_rank),
                "sample_dataset_index": TensorPointer(group_rank=self.output_pp_rank),
            }

        processed = []
        sample_dataset_indices: List[int] = []
        for example in examples:
            input_ids = np.asarray(example["input_ids"])
            dataset_index = example.get("_dataset_index")
            if dataset_index is not None:
                dataset_index = int(dataset_index)
            else:
                # Keep backwards compatibility for single-dataset pipelines that do not
                # expose source indices in the dataloader output.
                dataset_index = -1
            sample_dataset_indices.append(dataset_index)
            if "positions" in example:
                positions = np.asarray(example["positions"])
            else:
                # Fallback for dataset variants that do not emit explicit positions.
                positions = self._build_fallback_positions(input_ids=input_ids)
            processed.append(
                self._build_sdsp_example(
                    input_ids=input_ids,
                    positions=positions,
                    dataset_index=dataset_index,
                )
            )

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {
            "base_input_ids": TensorPointer(group_rank=self.input_pp_rank),
            "base_position_ids": TensorPointer(group_rank=self.input_pp_rank),
            "cond_input_ids": TensorPointer(group_rank=self.input_pp_rank),
            "cond_position_ids": TensorPointer(group_rank=self.input_pp_rank),
            "base_label_ids": TensorPointer(group_rank=self.output_pp_rank),
            "base_label_mask": TensorPointer(group_rank=self.output_pp_rank),
            "base_sdsp_pair_mask": TensorPointer(group_rank=self.output_pp_rank),
            "cond_label_ids": TensorPointer(group_rank=self.output_pp_rank),
            "cond_label_mask": TensorPointer(group_rank=self.output_pp_rank),
            "cond_sdsp_pair_mask": TensorPointer(group_rank=self.output_pp_rank),
            "sample_dataset_index": TensorPointer(group_rank=self.output_pp_rank),
        }

        if current_pp_rank == self.input_pp_rank:
            for key in ("base_input_ids", "cond_input_ids"):
                result[key] = np.vstack([example[key] for example in processed])[:, local_slice]
            for key in ("base_position_ids", "cond_position_ids"):
                stacked_positions = np.vstack([example[key] for example in processed])
                if self.cp_return_global_position_ids:
                    result[key] = stacked_positions
                else:
                    result[key] = stacked_positions[:, local_slice]

        if current_pp_rank == self.output_pp_rank:
            for key in (
                "base_label_ids",
                "base_label_mask",
                "base_sdsp_pair_mask",
                "cond_label_ids",
                "cond_label_mask",
                "cond_sdsp_pair_mask",
            ):
                result[key] = np.vstack([example[key] for example in processed])[:, local_slice]
            result["sample_dataset_index"] = np.asarray(sample_dataset_indices, dtype=np.int64)

        return result
