"""Inference-only Qwen4-Exp (text + VL) on the Qwen3.5 backbone."""

import math
from contextlib import nullcontext
from typing import Any, Iterable, Optional, Set, Tuple

import msgspec
import sympy
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_all_reduce,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_size,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import logger

# Decode/verify-sized batches only: at prefill sizes both chains are compute
# bound and serializing them on one stream is faster than contending.
_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD = 1024



# ==== NVFP4-packed PLE support (local patch; env SGLANG_QWEN4_PLE_NVFP4=1) ====
_NVFP4_LUT = None
_NVFP4_MIDS = None


def _nvfp4_ple_enabled() -> bool:
    import os

    return os.environ.get("SGLANG_QWEN4_PLE_NVFP4", "0") == "1"


def _nvfp4_get_lut(device):
    global _NVFP4_LUT
    if _NVFP4_LUT is None or _NVFP4_LUT.device != device:
        mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        _NVFP4_LUT = torch.tensor(
            mags + [-m for m in mags], dtype=torch.bfloat16, device=device
        )
    return _NVFP4_LUT


def _nvfp4_get_mids(device):
    global _NVFP4_MIDS
    if _NVFP4_MIDS is None or _NVFP4_MIDS.device != device:
        _NVFP4_MIDS = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            dtype=torch.float32,
            device=device,
        )
    return _NVFP4_MIDS


def _nvfp4_quantize_rows(w, chunk_rows: int = 262144):
    """rows [N, D] (fp8/bf16, any device) -> (uint8 [N, D//2], bf16 [N, D//16]) on cuda."""
    n, d = w.shape
    assert d % 16 == 0
    dev = torch.device("cuda")
    packed_out = torch.empty(n, d // 2, dtype=torch.uint8, device=dev)
    gscale_out = torch.empty(n, d // 16, dtype=torch.float8_e4m3fn, device=dev)
    mids = _nvfp4_get_mids(dev)
    for st in range(0, n, chunk_rows):
        en = min(st + chunk_rows, n)
        x = w[st:en].to(device=dev).to(torch.float32).view(en - st, d // 16, 16)
        amax = x.abs().amax(dim=-1)
        scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
        q = x / scale.unsqueeze(-1)
        idx = torch.bucketize(q.abs(), mids).to(torch.uint8)
        code = torch.where(q < 0, idx + 8, idx).view(en - st, d)
        packed_out[st:en] = code[:, 0::2] | (code[:, 1::2] << 4)
        gscale_out[st:en] = scale.to(torch.float8_e4m3fn)
        del x, amax, scale, q, idx, code
    return packed_out, gscale_out


def _nvfp4_convert_embedding(emb) -> None:
    """Swap a VocabParallelEmbedding weight for packed-NVFP4 buffers (tp1 only)."""
    w = emb.weight
    n, d = w.shape
    device = w.device
    del emb._parameters["weight"]
    emb.register_buffer(
        "weight_packed",
        torch.zeros(n, d // 2, dtype=torch.uint8, device=device),
        persistent=False,
    )
    emb.register_buffer(
        "weight_gscale",
        torch.zeros(n, d // 16, dtype=torch.float8_e4m3fn, device=device),
        persistent=False,
    )
    emb.nvfp4_packed = True
    torch.cuda.empty_cache()


def _nvfp4_gather(emb, ids: torch.Tensor) -> torch.Tensor:
    d = emb.weight_packed.shape[1] * 2
    p = emb.weight_packed[ids]
    lut = _nvfp4_get_lut(p.device)
    lo = lut[(p & 0xF).to(torch.long)]
    hi = lut[(p >> 4).to(torch.long)]
    codes = torch.stack((lo, hi), dim=-1).reshape(*ids.shape, d)
    gs = emb.weight_gscale[ids].to(torch.bfloat16)
    out = codes.view(*ids.shape, d // 16, 16) * gs.unsqueeze(-1)
    return out.reshape(*ids.shape, d)


# ==== end NVFP4-packed PLE support ====


def _get_ple_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
    if forward_batch._original_forward_mode is not None:
        return forward_batch._original_forward_mode
    return forward_batch.forward_mode


def _get_processed_token_count(
    forward_batch: ForwardBatch, physical_tokens: int
) -> int:
    processed_tokens = forward_batch.num_token_non_padded_cpu
    if processed_tokens is None and forward_batch.extend_seq_lens_cpu is not None:
        processed_tokens = sum(forward_batch.extend_seq_lens_cpu)
    if processed_tokens is None:
        return physical_tokens
    processed_tokens = int(processed_tokens)
    if not 0 <= processed_tokens <= physical_tokens:
        raise RuntimeError(
            f"invalid PLE token counts: {processed_tokens=}, {physical_tokens=}"
        )
    return processed_tokens


class _PLEBatch(msgspec.Struct, frozen=True):
    mode: ForwardMode
    use_decode_fast_path: bool
    physical_tokens: int
    processed_tokens: int
    lengths: torch.Tensor
    row_width: int
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: Optional[torch.Tensor]
    ngram_eos_token_id: Optional[int]


def _prepare_ple_batch(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    ngram_size: Optional[int],
    ngram_eos_token_id: Optional[int],
) -> Optional[_PLEBatch]:
    """Prepare the token layout and the shared N-gram history once per forward."""

    if forward_batch.tbo_parent_token_range is not None:
        raise NotImplementedError("Qwen4 PLE is not compatible with two-batch overlap")
    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and spec_algorithm.is_ngram():
        raise NotImplementedError("Qwen4 PLE does not support NGRAM speculation")
    if (
        forward_batch.spec_info is not None
        and getattr(forward_batch.spec_info, "topk", 1) != 1
    ):
        raise NotImplementedError("Qwen4 PLE speculative decoding supports only topk=1")

    mode = _get_ple_forward_mode(forward_batch)
    get_req_to_token_pool().ple_window_cache = None
    if mode.is_idle():
        return None
    use_decode_fast_path = (
        envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get() and mode.is_decode()
    )

    if input_ids.dim() > 1:
        input_ids = input_ids.reshape(-1)
    physical_tokens = input_ids.shape[0]
    processed_tokens = _get_processed_token_count(forward_batch, physical_tokens)
    tokens = input_ids[:processed_tokens]
    positions = torch.arange(processed_tokens, device=tokens.device, dtype=torch.long)

    if mode.is_target_verify():
        assert forward_batch.spec_info is not None
        row_width = int(forward_batch.spec_info.draft_token_num)
        if row_width <= 0 or processed_tokens % row_width != 0:
            raise RuntimeError(
                "target verify rows must contain complete draft strides: "
                f"{processed_tokens=} {row_width=}"
            )
        sequence_count = processed_tokens // row_width
        # Eager verify can carry a valid length smaller than its fixed row
        # stride.  Ignore the synthetic one-token lengths created when DP
        # attention temporarily rewrites verify as EXTEND.
        lengths = (
            forward_batch.extend_seq_lens[:sequence_count].long()
            if forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_seq_lens is not None
            else torch.full(
                (sequence_count,),
                row_width,
                dtype=torch.long,
                device=tokens.device,
            )
        )
        if lengths.shape[0] != sequence_count:
            raise RuntimeError(
                "target verify length metadata does not match its fixed rows: "
                f"{lengths.shape[0]=} {sequence_count=}"
            )
        req_indices = torch.div(positions, row_width, rounding_mode="floor")
        token_offsets = positions - req_indices * row_width
    elif mode.is_decode():
        lengths = torch.ones(processed_tokens, dtype=torch.long, device=tokens.device)
        row_width = 1
        req_indices = positions
        token_offsets = torch.zeros_like(positions)
    else:
        if forward_batch.extend_seq_lens is None:
            raise RuntimeError(f"PLE requires sequence lengths in {mode!r}")
        lengths = forward_batch.extend_seq_lens.long()
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        row_width = (
            max(extend_seq_lens_cpu, default=0)
            if extend_seq_lens_cpu is not None
            else processed_tokens
        )
        query_start_loc = torch.cat(
            [lengths.new_zeros(1), torch.cumsum(lengths, dim=0)]
        )
        sequence_count = lengths.shape[0]
        req_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        if processed_tokens:
            req_indices = req_indices.clamp(min=0, max=sequence_count - 1)
        token_offsets = positions - query_start_loc.index_select(0, req_indices)

    sequence_count = lengths.shape[0]
    if use_decode_fast_path:
        # Decode has one real token position per row.  Retain explicit tensors for
        # the common PLE batch contract without launching an index-select solely
        # to prove that every token offset (zero) is below every length (one).
        valid_tokens = torch.ones(
            processed_tokens, dtype=torch.bool, device=tokens.device
        )
    else:
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)

    state_indices = (
        get_req_to_token_pool()
        .get_mamba_indices(forward_batch.req_pool_indices[:sequence_count])
        .long()
    )

    # CUDA graph padding uses request slot 0, which may belong to a real request.
    # Map padded sequences to the state pools' reserved dummy slot instead.
    out_cache_loc = forward_batch.out_cache_loc
    if use_decode_fast_path:
        if out_cache_loc is not None:
            state_indices = torch.where(
                out_cache_loc[:sequence_count].ne(0),
                state_indices,
                torch.zeros_like(state_indices),
            )
    else:
        valid = lengths.ne(0)
        if out_cache_loc is not None and mode.is_decode():
            valid = valid & out_cache_loc[:sequence_count].ne(0)
        elif out_cache_loc is not None and mode.is_target_verify():
            valid = valid & out_cache_loc[:processed_tokens].reshape(
                sequence_count, row_width
            ).ne(0).any(dim=1)
        state_indices = torch.where(
            valid, state_indices, torch.zeros_like(state_indices)
        )

    ngram_context = None
    if ngram_size is not None:
        assert ngram_eos_token_id is not None
        if use_decode_fast_path:
            # For decode, rows and tokens are one-to-one and valid_tokens is all
            # true.  The view is exactly the tensor the fill/where/index-put chain
            # would materialize.
            padded = tokens.unsqueeze(1)
        else:
            padded = tokens.new_full((sequence_count, row_width), ngram_eos_token_id)
            if processed_tokens:
                padded[req_indices, token_offsets] = torch.where(
                    valid_tokens,
                    tokens,
                    tokens.new_full((), ngram_eos_token_id),
                )
        history = get_req_to_token_pool().get_ngram_context(state_indices)
        if history.shape[1] != ngram_size - 1:
            raise RuntimeError(
                "Qwen4 PLE N-gram cache has the wrong context width: "
                f"{history.shape[1]=} {ngram_size=}"
            )
        ngram_context = torch.cat([history, padded], dim=1)

    return _PLEBatch(
        mode=mode,
        use_decode_fast_path=use_decode_fast_path,
        physical_tokens=physical_tokens,
        processed_tokens=processed_tokens,
        lengths=lengths,
        row_width=row_width,
        req_indices=req_indices,
        token_offsets=token_offsets,
        valid_tokens=valid_tokens,
        state_indices=state_indices,
        ngram_context=ngram_context,
        ngram_eos_token_id=ngram_eos_token_id,
    )


def _commit_ple_batch(batch: Optional[_PLEBatch], forward_batch: ForwardBatch) -> None:
    """Commit the shared N-gram history after every PLE layer consumed it."""

    if batch is None or batch.ngram_context is None or not batch.processed_tokens:
        return

    pool = get_req_to_token_pool()
    context = batch.ngram_context
    context_len = context.shape[1] - batch.row_width
    if batch.mode.is_target_verify():
        step_contexts = context.unfold(1, context_len, 1)[:, 1:]
        valid_steps = batch.valid_tokens.reshape(
            batch.lengths.shape[0], batch.row_width
        )
        pool.set_ngram_intermediate_context(
            torch.where(
                valid_steps.unsqueeze(-1),
                step_contexts,
                torch.full_like(step_contexts, batch.ngram_eos_token_id),
            )
        )
        return

    if batch.use_decode_fast_path:
        # Decode advances every two-token history by exactly one column.  Slicing
        # preserves the int64 values while avoiding arange + gather launches.
        next_context = context[:, batch.row_width :]
        pool.set_ngram_context(batch.state_indices, next_context)
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, _ = track
            pool.set_ngram_context(track_indices, next_context)
        return

    context_cols = torch.arange(context_len, device=context.device, dtype=torch.long)
    next_context = context.gather(
        1, batch.lengths.unsqueeze(1) + context_cols.unsqueeze(0)
    )
    pool.set_ngram_context(batch.state_indices, next_context)

    track = _ple_track_targets(forward_batch, batch)
    if track is not None:
        track_indices, track_offsets = track
        pool.set_ngram_context(
            track_indices,
            context.gather(1, track_offsets.unsqueeze(1) + context_cols.unsqueeze(0)),
        )


def _ple_track_targets(
    forward_batch: ForwardBatch, batch: _PLEBatch
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Destination slots and gather offsets for the extra-buffer track snapshot.

    With extra_buffer it is the ping-pong track slot : not the request's working
    slot : that `cache_{un,}finished_req` hand to the radix tree, so a side state
    that only writes the working slot gets cached with whatever the track slot's
    previous owner left behind.

    Both PLE side states are laid out `[incoming_state | this chunk's tokens]`, so
    the boundary value is the state's own gather at a smaller offset; callers differ
    only in tensor rank, hence returning offsets rather than doing the gather.
    Masked-off rows route to reserved slot 0 instead of being compacted, so there is
    no host sync and no data-dependent shape and this stays CUDA-graph capturable.

    None when tracking is inactive or its metadata is absent.
    """
    track_indices = forward_batch.mamba_track_indices
    track_mask = forward_batch.mamba_track_mask
    if track_indices is None or track_mask is None:
        return None

    rows = batch.lengths.shape[0]
    track_indices = track_indices[:rows]
    dst = torch.where(track_mask[:rows], track_indices, torch.zeros_like(track_indices))

    if batch.mode.is_decode():
        # One token per step, so the boundary offset is the current one. Decode never
        # carries mamba_track_seqlens, so this path must not consult it.
        return dst, batch.lengths

    aligned = forward_batch.mamba_track_aligned_lens()
    if aligned is None:
        return None

    return dst, aligned[:rows].clamp(min=0).minimum(batch.lengths)


def _pad_token_rows(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    if x.shape[0] == total_tokens:
        return x
    out = x.new_zeros((total_tokens, *x.shape[1:]))
    out[: x.shape[0]] = x
    return out


def _use_attn_tp_ngram() -> bool:
    return is_dp_attention_enabled() and envs.SGLANG_USE_ATTN_TP_NGRAM.get()


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(x, self.weight, self._jit_group_size, self.eps)
        compute_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        else:
            group_shape = x_float.shape[:-1] + (-1, self.group_size)
            variance = x_float.reshape(group_shape).pow(2).mean(dim=-1, keepdim=True)
            variance = variance.expand(group_shape).reshape_as(x_float)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        weight = self.weight.float() + 1.0
        return (x_norm * weight).to(compute_dtype)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PRIME_1 = 10007

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_layer_index: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )
        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(
            config.make_ngram_vocab_size_divisible_by
        )
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.enable_ple_fusion = envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = (
            self._build_head_vocab_and_offsets()
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.use_attn_tp_ngram = _use_attn_tp_ngram()
        self.gather_dp_tokens = (
            is_dp_attention_enabled()
            and get_attention_dp_size() > 1
            and not self.use_attn_tp_ngram
        )
        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim_per_ngram,
            params_dtype=(
                torch.float8_e4m3fn
                if (quant_config is not None and quant_config.get_name() == "fp8")
                or getattr(config, "ple_embedding_dtype", None) == "float8_e4m3fn"
                else torch.bfloat16
            ),
            output_dtype=torch.bfloat16,
            use_attn_tp_group=self.use_attn_tp_ngram,
        )
        self.ngram_embedding.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )
        if _nvfp4_ple_enabled():
            if getattr(config, "ple_offload_embedding", False):
                logger.info(
                    "NVFP4 PLE: disabling ple_offload_embedding (UMA box, packed storage)"
                )
                config.ple_offload_embedding = False
            _nvfp4_convert_embedding(self.ngram_embedding)
            logger.info(
                "PLE embedding using packed NVFP4 storage: rows=%d dim=%d",
                self.ngram_embedding.weight_packed.shape[0],
                self.ngram_embedding.weight_packed.shape[1] * 2,
            )

    @classmethod
    def _splitmix64(cls, x: int) -> int:
        x = (x + cls._SPLITMIX_GAMMA) & cls._MASK64
        x = ((x ^ (x >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        x = ((x ^ (x >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (x ^ (x >> 31)) & cls._MASK64

    def _build_layer_multipliers(self, size: int) -> torch.Tensor:
        seed = int(getattr(self.config, "seed", 1234))
        max_long = (1 << 63) - 1
        m_max = max_long // max(self.unigram_vocab_size, 1)
        half_bound = max(1, m_max // 2)
        values = []
        base_seed = seed + self._PRIME_1 * self.ple_layer_index
        for idx in range(size):
            x0 = (base_seed + self._SPLITMIX_GAMMA * (idx + 1)) & self._MASK64
            mixed = self._splitmix64(x0)
            values.append(int(2 * (mixed % half_bound) + 1))
        return torch.tensor(values, dtype=torch.long)

    @staticmethod
    def _find_nth_prime_after(start: int, n: int) -> int:
        prime = int(start)
        for _ in range(n):
            prime = int(sympy.nextprime(prime))
        return prime

    def _build_head_vocab_and_offsets(self):
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = self._find_nth_prime_after(
                self.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        return sizes, offsets, total

    def _embed_ngram_ids(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        if getattr(self.ngram_embedding, "nvfp4_packed", False):
            embeddings = _nvfp4_gather(self.ngram_embedding, lookup_ids)
        elif isinstance(self.ngram_embedding, Qwen4ExpPinnedHostEmbedding):
            embeddings = _ple_cached_gather(self.ngram_embedding, lookup_ids)
        else:
            embeddings = self.ngram_embedding(lookup_ids)
        embeddings = embeddings * self.ngram_embedding.weight_scale
        return self._finish_embedding_lookup(
            embeddings, semantic_tokens, forward_batch, physical_tokens
        )

    def _prepare_embedding_lookup(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> Tuple[torch.Tensor, int]:
        semantic_tokens = ngram_ids.shape[0]
        if not self.gather_dp_tokens:
            return ngram_ids, semantic_tokens

        padded_ngram_ids = _pad_token_rows(ngram_ids, physical_tokens)
        global_tokens = forward_batch.global_dp_buffer_len
        if global_tokens is None:
            raise RuntimeError(
                "global-TP Qwen4 N-gram lookup under DP attention requires a "
                "DP token layout; set SGLANG_USE_ATTN_TP_NGRAM=1 to shard the "
                "table within each attention-TP group"
            )

        global_ngram_ids = ngram_ids.new_empty((global_tokens, *ngram_ids.shape[1:]))
        dp_gather_replicate(
            global_ngram_ids, padded_ngram_ids.contiguous(), forward_batch
        )
        return global_ngram_ids, semantic_tokens

    def _finish_embedding_lookup(
        self,
        embeddings: torch.Tensor,
        semantic_tokens: int,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        if not self.gather_dp_tokens:
            return embeddings
        local_embeddings = embeddings.new_empty(
            (physical_tokens, *embeddings.shape[1:])
        )
        dp_scatter(local_embeddings, embeddings.contiguous(), forward_batch)
        return local_embeddings[:semantic_tokens]

    def _hash_contexts(
        self, contexts: torch.Tensor, *, decode_sized: bool = False
    ) -> torch.Tensor:
        contexts = contexts.to(torch.long)
        if self.enable_ple_fusion and decode_sized:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_ngram_hash,
                fused_qwen4_ngram_hash,
            )

            if can_fuse_qwen4_ngram_hash(
                contexts,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
            ):
                return fused_qwen4_ngram_hash(
                    contexts,
                    self.layer_multipliers,
                    self.ngram_heads_vocab_sizes,
                    self.ngram_heads_offsets,
                    self.eos_token_id,
                )

        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[1] is contexts and cached[2] is not None:
            shifted_tokens = cached[2]
            assert len(shifted_tokens) == self.ngram_size
        else:
            shifted_tokens = [contexts]
            for shift in range(1, self.ngram_size):
                shifted_tokens.append(self._shift_right_ignore_eos(contexts, shift))
            if cached is not None and cached[1] is contexts:
                pool.ple_window_cache = (cached[0], contexts, shifted_tokens)

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            ngram_idx = ngram - 2
            start_idx = ngram_idx * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mix = shifted_tokens[0] * self.layer_multipliers[0]
            for pos in range(1, ngram):
                mix = torch.bitwise_xor(
                    mix, shifted_tokens[pos] * self.layer_multipliers[pos]
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(
                mix[:, -1:].unsqueeze(-1), head_vocab_sizes.view(1, 1, -1)
            )
            ngram_ids = ngram_ids + head_offsets.view(1, 1, -1)
            blocks.append(ngram_ids[:, 0])
        return torch.cat(blocks, dim=-1)

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return tensor
        batch_size, seq_len = tensor.shape
        idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
        eos_mask = tensor == self.eos_token_id
        eos_pos = torch.where(eos_mask, idx, -1)
        prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat(
            [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = prev_eos + 1
        pos_in_segment = idx.unsqueeze(0) - segment_start
        src_idx = idx - n
        gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
        shifted = tensor.gather(dim=1, index=gather_idx)
        valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
        return torch.where(valid_mask, shifted, tensor.new_full((), self.eos_token_id))

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if not self.gather_dp_tokens:
            return
        input_ids = forward_batch.input_ids.reshape(-1)
        dummy_ids = input_ids.new_zeros((input_ids.shape[0], self.ngram_heads))
        self._embed_ngram_ids(dummy_ids, forward_batch, input_ids.shape[0])

    def forward(
        self,
        batch: _PLEBatch,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        ngram_ids = self.compute_ngram_ids(batch)
        embeddings = self._embed_ngram_ids(
            ngram_ids, forward_batch, batch.physical_tokens
        )
        return embeddings.flatten(start_dim=-2)

    def compute_ngram_ids(self, batch: _PLEBatch) -> torch.Tensor:
        assert batch.ngram_context is not None
        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[0] is batch:
            contexts = cached[1]
        else:
            if batch.use_decode_fast_path:
                contexts = batch.ngram_context
            else:
                contexts = batch.ngram_context.unfold(1, self.ngram_size, 1)[
                    batch.req_indices, batch.token_offsets
                ]
            contexts = contexts.to(torch.long)
            pool.ple_window_cache = (batch, contexts, None)
        return self._hash_contexts(
            contexts,
            decode_sized=batch.mode.is_decode() or batch.mode.is_target_verify(),
        )


@triton.jit
def _gather_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


_PLE_MMAP_DIR = None

_PLE_CACHE = None

# --------------------------------------------------------------------- prefetch
# The mmap'd PLE table is MADV_RANDOM (readahead off; correct for 160 B rows), so
# every row the gather touches is its own 4 KiB fault. A prefill chunk touches
# 8192 tok x 16 rows = 131072 rows scattered over 320M, and serial faulting reads
# them at ~84 MB/s. Issuing MADV_WILLNEED over the coalesced page set first lets
# the kernel fault them in parallel at ~1.76 GB/s: measured 6516 ms -> 437 ms
# (14.9x) on a cold 8192-token chunk, major faults 133071 -> 0.
#
# But when the pages are already resident the advise buys nothing and costs
# ~129 ms, which is 4.2x SLOWER than just gathering. So it is gated on observed
# major faults: enable when the gather is faulting, and probe periodically with
# it off to notice when the table has gone warm. mincore() is not usable for this
# (it reports every page resident on this box, including pages that then fault).
# Bench: bench_ple_prefetch.py / bench_ple_pmadvise.py / bench_ple_warm.py.
_PLE_MMAP_ADDR = 0
_PLE_MMAP_BYTES = 0
_PLE_ROW_BYTES = 0
_PLE_PF = {"on": True, "ema": 0.0, "calls": 0, "saved": 0, "pidfd": -1, "pending": None,
           "low_probes": 0, "last_faults": None}
_PLE_PF_MIN_ROWS = 4096      # below this it is a decode-sized gather: never advise, never steer
_PLE_PF_ON_FAULTS = 500.0    # an observed large gather above this turns prefetch (back) on
_PLE_PF_OFF_FAULTS = 50.0    # a probe below this counts as "table is warm"
_PLE_PF_OFF_PROBES = 3       # consecutive warm probes needed to turn off
_PLE_PF_PROBE_EVERY = 16     # while on, every 16th large gather is a probe (no advise)


def _ple_majflt():
    try:
        with open("/proc/self/stat", "rb") as f:
            return int(f.read().split()[11])
    except Exception:
        return -1


def _ple_prefetch_rows(flat_ids):
    """MADV_WILLNEED over the pages holding these rows. Returns True if issued."""
    import ctypes
    import os
    import numpy as np

    if not _PLE_MMAP_ADDR or _PLE_ROW_BYTES <= 0:
        return False
    try:
        ids = flat_ids.detach().to("cpu", non_blocking=False).numpy().astype(np.int64, copy=False)
        page = 4096
        off = ids * _PLE_ROW_BYTES
        pg = np.unique(np.concatenate([off // page, (off + _PLE_ROW_BYTES - 1) // page]))
        pg = pg[(pg >= 0) & (pg < _PLE_MMAP_BYTES // page)]
        if pg.size == 0:
            return False
        brk = np.flatnonzero(np.diff(pg) != 1)
        starts = np.concatenate([[pg[0]], pg[brk + 1]])
        ends = np.concatenate([pg[brk], [pg[-1]]])

        libc = ctypes.CDLL("libc.so.6", use_errno=True)

        class _IOVec(ctypes.Structure):
            _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]

        if _PLE_PF["pidfd"] < 0:
            _PLE_PF["pidfd"] = libc.syscall(ctypes.c_long(434), ctypes.c_int(os.getpid()),
                                            ctypes.c_uint(0))
        pidfd, iov_max, MADV_WILLNEED = _PLE_PF["pidfd"], 1024, 3
        n = len(starts)
        if pidfd >= 0:
            for i in range(0, n, iov_max):
                s, e = starts[i:i + iov_max], ends[i:i + iov_max]
                k = len(s)
                arr = (_IOVec * k)()
                for j in range(k):
                    arr[j].iov_base = ctypes.c_void_p(_PLE_MMAP_ADDR + int(s[j]) * page)
                    arr[j].iov_len = ctypes.c_size_t(int(e[j] - s[j] + 1) * page)
                if libc.syscall(ctypes.c_long(440), ctypes.c_int(pidfd), arr,
                                ctypes.c_size_t(k), ctypes.c_int(MADV_WILLNEED),
                                ctypes.c_uint(0)) < 0:
                    pidfd = _PLE_PF["pidfd"] = -1   # fall back to per-range madvise
                    break
        if pidfd < 0:
            for st_, en_ in zip(starts, ends):
                libc.madvise(ctypes.c_void_p(_PLE_MMAP_ADDR + int(st_) * page),
                             ctypes.c_size_t(int(en_ - st_ + 1) * page),
                             ctypes.c_int(3))
        return True
    except Exception as exc:   # never break the gather over an advisory hint
        import logging
        logging.getLogger(__name__).warning("PLE prefetch skipped (%s)", exc)
        return False


def _ple_prefetch_gate(flat_ids):
    """Decide whether to prefetch this gather. Returns (did_prefetch, majflt_before).

    Gate v2 (2026-09-05). Facts from the served debug arm: the only gathers with
    >= _PLE_PF_MIN_ROWS rows are prefill chunks (131072 rows for 8192 tokens); every
    other call is a decode-sized gather; and a gather's major faults are only visible
    at the NEXT gate call (the kernel runs async on the model's prefetch stream). The
    v1 gate let decode calls steer it and needed a prior observation before it would
    fire, so it flipped OFF within a second of every ON and never prefetched a chunk.
    Measured cost asymmetry: cold chunk 6516 -> 437 ms with the advise, warm chunk
    35 -> 150 ms. So v2 is optimistic: in auto mode every large gather is advised;
    every _PLE_PF_PROBE_EVERY-th large gather is a probe (no advise) whose faults are
    read at the next call; _PLE_PF_OFF_PROBES consecutive warm probes turn it off,
    and while off every large gather is observed so the first cold one turns it back
    on. Small gathers only serve to deliver the lagged observation.
    """
    import os as _os

    mode = _os.environ.get("SGLANG_PLE_PREFETCH", "off").strip().lower()
    if mode in ("0", "off", "false", "") or not _PLE_MMAP_ADDR:
        return False, -1
    if torch.cuda.is_current_stream_capturing():
        return False, -1        # never sync to host during capture
    st = _PLE_PF
    debug = _os.environ.get("SGLANG_PLE_PREFETCH_DEBUG") == "1"
    now = _ple_majflt()
    # 1) attribute the previous large gather's faults (lagged by one call)
    pend = st.get("pending")
    if pend is not None and now >= 0:
        did_prev, before_prev = pend
        faults = max(now - before_prev, 0)
        st["last_faults"] = faults
        if not did_prev:            # a probe (or an off-state call): steer on it
            st["ema"] = 0.8 * st["ema"] + 0.2 * faults
            import logging
            if st["on"]:
                if faults < _PLE_PF_OFF_FAULTS:
                    st["low_probes"] += 1
                    if st["low_probes"] >= _PLE_PF_OFF_PROBES:
                        st["on"] = False
                        st["low_probes"] = 0
                        logging.getLogger(__name__).info(
                            "PLE prefetch OFF (%d consecutive probes saw < %d faults; table is warm)",
                            _PLE_PF_OFF_PROBES, int(_PLE_PF_OFF_FAULTS))
                else:
                    st["low_probes"] = 0
            elif faults > _PLE_PF_ON_FAULTS:
                st["on"] = True
                st["low_probes"] = 0
                logging.getLogger(__name__).info(
                    "PLE prefetch ON (large gather took %d major faults)", faults)
            if debug:
                logging.getLogger(__name__).info(
                    "PLE prefetch observe: probe faults=%d ema=%.0f on=%s", faults, st["ema"], st["on"])
        st["pending"] = None
    # 2) decide for this call
    if flat_ids.numel() < _PLE_PF_MIN_ROWS:
        return False, -1
    st["calls"] += 1
    if mode in ("1", "on", "true"):
        want = True
    elif st["on"]:
        want = st["calls"] % _PLE_PF_PROBE_EVERY != 0
    else:
        want = False
    did = _ple_prefetch_rows(flat_ids) if want else False
    if did:
        st["saved"] += 1
    st["pending"] = (did, now)
    if debug:
        import logging
        logging.getLogger(__name__).info(
            "PLE prefetch gate: call=%d rows=%d mode=%s on=%s want=%s did=%s last_faults=%s ema=%.0f",
            st["calls"], int(flat_ids.numel()), mode, st["on"], want, did, st.get("last_faults"), st["ema"])
    return did, now


def _ple_prefetch_observe(before, did):
    """Kept for call-site compatibility; the lagged observation now happens at the
    start of the next _ple_prefetch_gate call (see there)."""
    return None




def _ple_cache_state(device):
    """Direct-mapped device cache over the host/mmap PLE table (lossless)."""
    global _PLE_CACHE
    if _PLE_CACHE is not None:
        return _PLE_CACHE
    import logging
    import os

    gb = float(os.environ.get("SGLANG_PLE_DEVICE_CACHE_GB", "0") or 0)
    if gb <= 0:
        _PLE_CACHE = False
        return False
    slots = int(gb * 1e9 / (160 + 8))  # fp8 row + int64 tag
    st = {
        "C": slots,
        "vals": torch.zeros(slots, 160, dtype=torch.float8_e4m3fn, device=device),
        "tags": torch.full((slots,), -1, dtype=torch.int64, device=device),
        "hits": 0,
        "total": 0,
        "calls": 0,
    }
    _PLE_CACHE = st
    logging.getLogger(__name__).info(
        "PLE device cache: %d slots (%.1f GB) in front of the host/mmap table",
        slots, gb,
    )
    return st


def _ple_cached_gather(emb, lookup_ids):
    """Cache-fronted gather. Returns bf16 embeddings like the pinned path."""
    st = _ple_cache_state(lookup_ids.device)
    if st is False:
        return emb(lookup_ids)
    flat = lookup_ids.reshape(-1).to(torch.int64)
    slots = flat % st["C"]
    hit = st["tags"][slots] == flat
    out = torch.empty(flat.numel(), 160, dtype=torch.bfloat16,
                      device=flat.device)
    n_hit = int(hit.sum())
    if n_hit:
        out[hit] = st["vals"][slots[hit]].to(torch.bfloat16)
    if n_hit < flat.numel():
        miss = ~hit
        mids = flat[miss]
        _pf_did, _pf_before = _ple_prefetch_gate(mids)
        fetched = emb(mids).reshape(-1, 160)  # slow pinned/mmap path
        _ple_prefetch_observe(_pf_before, _pf_did)
        out[miss] = fetched.to(torch.bfloat16)
        st["vals"][slots[miss]] = fetched.to(torch.float8_e4m3fn)
        st["tags"][slots[miss]] = mids
    st["hits"] += n_hit
    st["total"] += flat.numel()
    st["calls"] += 1
    if st["calls"] % 500 == 0:
        import logging

        logging.getLogger(__name__).info(
            "PLE cache hit rate: %.1f%% (%d/%d rows over %d calls)",
            100.0 * st["hits"] / max(st["total"], 1),
            st["hits"], st["total"], st["calls"],
        )
    return out.view(*lookup_ids.shape, 160)


def _alloc_ple_table(shape, dtype):
    """Backing store for the PLE n-gram table.

    With SGLANG_QWEN4_PLE_MMAP_DIR set, the table is a file-backed mmap on disk
    instead of pinned host RAM. On coherent CPU-GPU parts (GB10) the gather
    kernel dereferences that pageable host pointer directly, so the table never
    has to be resident. Without the variable, original behaviour.
    """
    import logging
    import os

    global _PLE_MMAP_DIR
    if _PLE_MMAP_DIR is None:
        _PLE_MMAP_DIR = os.environ.get("SGLANG_QWEN4_PLE_MMAP_DIR", "").strip()
    if not _PLE_MMAP_DIR:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    os.makedirs(_PLE_MMAP_DIR, exist_ok=True)
    path = os.path.join(_PLE_MMAP_DIR, "ple_table_%d_%d.bin" % (numel, nbytes))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        with open(path, "wb") as f:
            f.truncate(nbytes)
    logging.getLogger(__name__).info(
        "PLE table -> mmap %s (%.1f GiB, dtype=%s)", path, nbytes / 2**30, dtype
    )
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)

    # MADV_RANDOM: the table is purely random access (16 rows of 160 B per
    # token). Without it the kernel reads its whole readahead window around each
    # row. Measured cold: 1.4 MB of disk per token, ~560x more than is used.
    # Cheap in decode; a prefill chunk touches tens of thousands of rows.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        MADV_RANDOM = 1
        rc = libc.madvise(
            ctypes.c_void_p(storage.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(MADV_RANDOM),
        )
        logging.getLogger(__name__).info(
            "PLE table: madvise(MADV_RANDOM) %s", "ok" if rc == 0 else "failed"
        )
    except Exception as exc:  # not critical: affects I/O only, never correctness
        logging.getLogger(__name__).warning("PLE table: madvise not applied (%s)", exc)

    global _PLE_MMAP_ADDR, _PLE_MMAP_BYTES, _PLE_ROW_BYTES
    table = storage.view(dtype).view(*shape)
    _PLE_MMAP_ADDR = int(table.data_ptr())
    _PLE_MMAP_BYTES = int(nbytes)
    _PLE_ROW_BYTES = int(shape[-1]) * table.element_size()
    logging.getLogger(__name__).info(
        "PLE prefetch armed: base=0x%x rows=%d row_bytes=%d (SGLANG_PLE_PREFETCH=%s)",
        _PLE_MMAP_ADDR, int(shape[0]), _PLE_ROW_BYTES,
        os.environ.get("SGLANG_PLE_PREFETCH", "off"))
    return table


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    """PLE table read directly from pinned host memory.

    The table stays in its checkpoint storage dtype (fp8 with a per-tensor
    weight_scale for fp8 checkpoints, bf16 otherwise); gathers emit bf16.
    """

    _COPIED_ATTRIBUTES = (
        "quant_config",
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "num_embeddings",
        "org_vocab_size",
        "padding_size",
        "num_added_embeddings",
        "use_presharded_weights",
        "org_vocab_size_padded",
        "num_embeddings_padded",
        "shard_indices",
        "embedding_dim",
        "num_embeddings_per_partition",
        "num_org_embeddings_per_partition",
        "num_added_embeddings_per_partition",
    )

    def __init__(self, embedding: VocabParallelEmbedding) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE embedding offload requires an unquantized embedding table"
            )
        if embedding.weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(
                "PLE embedding offload requires bfloat16 or fp8 weights, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "PLE embedding offload does not support added vocabulary rows"
            )
        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        # The unquantized CUDA post-load hook is a no-op. Exclude this CPU-only
        # table so the generic loader does not stage it back to GPU unnecessarily.
        self.quant_method = None

        source_weight = embedding.weight
        cpu_weight = nn.Parameter(
            _alloc_ple_table(source_weight.shape, source_weight.dtype),
            requires_grad=False,
        )
        for name, value in vars(source_weight).items():
            setattr(cpu_weight, name, value)
        cpu_weight.weight_loader = self.weight_loader
        self.register_parameter("weight", cpu_weight)
        # The scale is tiny; keep it with the model instead of offloading it
        # with the table.
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            # The gather kernel emits bf16 rows regardless of the table dtype.
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            st = _ple_cache_state(flat_ids.device)
            if st is False or torch.cuda.is_current_stream_capturing():
                _pf_did, _pf_before = _ple_prefetch_gate(flat_ids)
                _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                    self.weight.data_ptr(),
                    flat_ids,
                    output,
                    embedding_dim=self.embedding_dim,
                    tp_vocab_start=self.shard_indices.org_vocab_start_index,
                    tp_vocab_end=self.shard_indices.org_vocab_end_index,
                    is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                    BLOCK_D=self._block_d,
                )
                _ple_prefetch_observe(_pf_before, _pf_did)
            else:
                out2 = output.view(-1, self.embedding_dim)
                slots = flat_ids % st["C"]
                hit = st["tags"][slots] == flat_ids
                n_hit = int(hit.sum())
                if n_hit:
                    out2[hit] = st["vals"][slots[hit]].to(torch.bfloat16)
                if n_hit < flat_ids.numel():
                    miss = ~hit
                    mids = flat_ids[miss].contiguous()
                    mout = torch.empty(
                        mids.numel(), self.embedding_dim,
                        dtype=torch.bfloat16, device=flat_ids.device,
                    )
                    _gather_ple_embedding_from_pinned_kernel[(mids.numel(),)](
                        self.weight.data_ptr(),
                        mids,
                        mout,
                        embedding_dim=self.embedding_dim,
                        tp_vocab_start=self.shard_indices.org_vocab_start_index,
                        tp_vocab_end=self.shard_indices.org_vocab_end_index,
                        is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                        BLOCK_D=self._block_d,
                    )
                    out2[miss] = mout
                    st["vals"][slots[miss]] = mout.to(torch.float8_e4m3fn)
                    st["tags"][slots[miss]] = mids
                st["hits"] += n_hit
                st["total"] += flat_ids.numel()
                st["calls"] += 1
                if st["calls"] % 200 == 0:
                    import logging

                    logging.getLogger(__name__).info(
                        "PLE cache hit rate: %.1f%% (%d/%d rows, %d calls)",
                        100.0 * st["hits"] / max(st["total"], 1),
                        st["hits"], st["total"], st["calls"],
                    )
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))


class Qwen4ExpPLELayer(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: Optional[int] = None,
        ple_layer_index: int = 0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.ple_embed_dim = config.ple_embed_dim
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            self.ple_embed_dim,
            ple_layer_index=ple_layer_index,
            quant_config=quant_config,
        )
        if config.ple_offload_embedding:
            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                self.ple_embedding.ngram_embedding
            )
        self.short_conv_dilation = self.ple_embedding.ngram_size
        self.short_conv_state_len = (
            self.conv_kernel_size - 1
        ) * self.short_conv_dilation
        self.conv_channels = self.hc_hidden_size
        self.key_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.conv_channels,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_hidden = self.hc_hidden_size
        norm_group = self.hidden_size
        self.norm_key = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_query = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_conv = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_channels,
            padding=(self.conv_kernel_size - 1) * self.short_conv_dilation,
            dilation=self.short_conv_dilation,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)
        self._prefetch_stream = (
            torch.cuda.Stream() if config.ple_offload_embedding else None
        )
        self._graph_prefetch_buffers = {}
        self._eager_prefetch_buffer = None
        self._prefetch_state = None

    def _apply_ple_norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = norm(x.flatten(-2, -1))
        return y.unflatten(-1, (self.hc_count, self.hidden_size))

    def _short_conv(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        pool = get_req_to_token_pool()
        conv_state = pool.short_conv_layer_cache(self.layer_id)

        if batch.use_decode_fast_path:
            # Preserve the native depthwise-convolution and SiLU implementations;
            # only remove decode identities around them.  With row_width=1 the
            # padded/transpose path is exactly x.unsqueeze(-1), and every state
            # boundary is the contiguous one-column shift below.
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_short_conv_state,
                fused_qwen4_short_conv_state,
            )

            fused_state = can_fuse_qwen4_short_conv_state(
                conv_state, batch.state_indices, x
            )
            if fused_state:
                conv_input = fused_qwen4_short_conv_state(
                    conv_state, batch.state_indices, x
                )
            else:
                state = conv_state.index_select(0, batch.state_indices).to(
                    dtype=x.dtype
                )
                conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)
            conv_output = F.conv1d(
                conv_input,
                self.conv1d.weight.to(dtype=x.dtype),
                bias=None,
                dilation=self.short_conv_dilation,
                groups=self.conv_channels,
            ).squeeze(-1)
            next_state = conv_input[:, :, batch.row_width :]
            if not fused_state:
                conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, _ = track
                conv_state[track_indices] = next_state.to(dtype=conv_state.dtype)
            return F.silu(conv_output)

        state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
        padded_seq = x.new_zeros(
            (batch.lengths.shape[0], batch.row_width, self.conv_channels)
        )
        padded_seq[batch.req_indices, batch.token_offsets] = x
        conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
        conv_output = F.conv1d(
            conv_input,
            self.conv1d.weight.to(dtype=x.dtype),
            bias=None,
            dilation=self.short_conv_dilation,
            groups=self.conv_channels,
        ).transpose(1, 2)

        if batch.mode.is_target_verify():
            intermediate_cache = pool.short_conv_layer_intermediate_cache(self.layer_id)
            if intermediate_cache is not None:
                if self.short_conv_state_len:
                    intermediate_state = (
                        conv_input.unfold(2, self.short_conv_state_len, 1)[
                            :, :, 1 : batch.row_width + 1
                        ]
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                else:
                    intermediate_state = x.new_empty(
                        (
                            batch.lengths.shape[0],
                            batch.row_width,
                            self.conv_channels,
                            0,
                        )
                    )
                valid_steps = batch.valid_tokens.reshape(
                    batch.lengths.shape[0], batch.row_width, 1, 1
                )
                intermediate_state = torch.where(
                    valid_steps,
                    intermediate_state,
                    torch.zeros_like(intermediate_state),
                )
                intermediate_cache[: batch.lengths.shape[0], : batch.row_width].copy_(
                    intermediate_state.to(dtype=intermediate_cache.dtype)
                )
        else:
            state_cols = torch.arange(
                self.short_conv_state_len, device=x.device, dtype=torch.long
            )

            def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
                return conv_input.gather(
                    2,
                    (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
                    .unsqueeze(1)
                    .expand(-1, self.conv_channels, -1),
                )

            next_state = _gather_at(batch.lengths)
            conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            # Same boundary mamba uses, into the slot the radix tree reads.
            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, track_offsets = track
                conv_state[track_indices] = _gather_at(track_offsets).to(
                    dtype=conv_state.dtype
                )

        return F.silu(conv_output[batch.req_indices, batch.token_offsets])

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if self._prefetch_state is not None:
            self._consume_prefetched_embeddings(forward_batch)
        else:
            self.ple_embedding.forward_idle(forward_batch)

    def _allocate_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        offloaded_embedding = self.ple_embedding.ngram_embedding
        return offloaded_embedding.allocate_output(
            (lookup_tokens, self.ple_embed_dim), lookup_ids.device
        )

    def _get_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        if get_is_capture_mode():
            buffer = self._graph_prefetch_buffers.get(lookup_tokens)
            if buffer is None:
                buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
                self._graph_prefetch_buffers[lookup_tokens] = buffer
            return buffer

        buffer = self._eager_prefetch_buffer
        if buffer is None or buffer.shape[0] < lookup_tokens:
            buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
            self._eager_prefetch_buffer = buffer
        return buffer[:lookup_tokens]

    def start_prefetch(
        self,
        batch: Optional[_PLEBatch],
        forward_batch: ForwardBatch,
    ) -> None:
        """Gather PLE rows via UVA while the preceding decoder layer runs."""
        if self._prefetch_stream is None:
            return
        if self._prefetch_state is not None:
            raise RuntimeError("PLE prefetch state was not consumed before reuse")
        if batch is None:
            if not self.ple_embedding.gather_dp_tokens:
                return
            physical_tokens = forward_batch.input_ids.numel()
            ngram_ids = forward_batch.input_ids.new_zeros(
                (physical_tokens, self.ple_embedding.ngram_heads)
            )
        else:
            physical_tokens = batch.physical_tokens
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)

        lookup_ids, semantic_tokens = self.ple_embedding._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            return
        prefetched = self._get_prefetch_buffer(lookup_tokens, lookup_ids)
        output_view = prefetched.view(lookup_tokens, self.ple_embedding.ngram_heads, -1)
        offloaded_embedding = self.ple_embedding.ngram_embedding

        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        lookup_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            offloaded_embedding.gather(lookup_ids, out=output_view)
        self._prefetch_state = prefetched, semantic_tokens, physical_tokens

    def _consume_prefetched_embeddings(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if self._prefetch_state is None:
            raise RuntimeError("PLE prefetch state is missing")
        embeddings, semantic_tokens, physical_tokens = self._prefetch_state
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        embeddings = self.ple_embedding.ngram_embedding.reduce(embeddings)
        embeddings = embeddings * self.ple_embedding.ngram_embedding.weight_scale
        embeddings = self.ple_embedding._finish_embedding_lookup(
            embeddings,
            semantic_tokens,
            forward_batch,
            physical_tokens,
        )
        self._prefetch_state = None
        return embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
    ) -> torch.Tensor:
        hidden_states = hidden_states[: batch.processed_tokens]
        if self._prefetch_state is not None:
            embeddings = self._consume_prefetched_embeddings(forward_batch)
        else:
            embeddings = self.ple_embedding(batch, forward_batch)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        hidden_size = self.hidden_size
        hc_count = self.hc_count
        if hidden_states.shape[-1] != hc_count * hidden_size:
            raise RuntimeError(
                "PLE hidden size does not match its hyper-connection layout: "
                f"expected {hc_count * hidden_size}, got {hidden_states.shape[-1]}"
            )
        key = key.reshape(token_count, hc_count, hidden_size)
        query = hidden_states.reshape(token_count, hc_count, hidden_size)
        key_normed = self._apply_ple_norm(self.norm_key, key)
        query_normed = self._apply_ple_norm(self.norm_query, query)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(hidden_size)
        fused_gate_value = False
        if batch.use_decode_fast_path:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_gate_value,
                fused_qwen4_gate_value,
            )

            fused_gate_value = can_fuse_qwen4_gate_value(gate, value)
        if fused_gate_value:
            gated_value = fused_qwen4_gate_value(gate, value)
        else:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = torch.sigmoid(gate)
            gated_value = gate * value.unsqueeze(-2)
        gated_value_normed = self._apply_ple_norm(self.norm_conv, gated_value)
        gated_value = gated_value.flatten(-2)
        gated_value_normed = gated_value_normed.flatten(-2)
        conv_output = self._short_conv(
            gated_value_normed,
            forward_batch,
            batch,
        )
        output = gated_value + conv_output
        if not batch.use_decode_fast_path:
            output = torch.where(
                batch.valid_tokens.unsqueeze(-1),
                output,
                torch.zeros_like(output),
            )
        return _pad_token_rows(output, batch.physical_tokens)


class Qwen4ExpLayerExtensionMixin:
    def _init_qwen4_exp_layer_extensions(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.ple = None

        for attr_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "layer_communicator",
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        if (layer_id + 1) in config.ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(config.ple_layer_ids))
            ple_layer_index = {
                abs_id: index for index, abs_id in enumerate(ple_layer_ids_sorted)
            }[layer_id + 1]
            # PLE is a sibling of the attn block (self.ple), so strip the block-type
            # segment like the dense mlp; else quant prefix misses ckpt skip-list → NaN.
            ple_prefix = prefix.replace(".linear_attn", "").replace(".self_attn", "")
            self.ple = Qwen4ExpPLELayer(
                config,
                quant_config=quant_config,
                prefix=f"{ple_prefix}.ple" if ple_prefix else "ple",
                layer_id=layer_id,
                ple_layer_index=ple_layer_index,
            )

        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
        )

    def _prepare_qwen4_exp_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        *,
        ple_batch: Optional[_PLEBatch],
    ):
        hc_dim = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] != hc_dim:
            assert hidden_states.shape[-1] == self.hidden_size
            hidden_states = torch.cat(
                [hidden_states for _ in range(self.hc_count)], dim=-1
            )

        if self.ple is not None:
            if ple_batch is None:
                if not _get_ple_forward_mode(forward_batch).is_idle():
                    raise RuntimeError(
                        "non-idle Qwen4 PLE forward is missing its batch"
                    )
                self.ple.forward_idle(forward_batch)
            else:
                ple_query = (
                    hidden_states if residual is None else hidden_states + residual
                )
                hidden_states = hidden_states + self.ple(
                    ple_query, forward_batch, ple_batch
                )

        hidden_states, residual = self.attn_hyper_connection.mix(hidden_states)
        return hidden_states, residual

    def _prepare_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        if not forward_batch.forward_mode.is_idle():
            hidden_states = attn_tp_all_reduce(hidden_states)
        hidden_states = self.attn_hyper_connection.combine(hidden_states, residual)
        hidden_states, residual = self.mlp_hyper_connection.mix(hidden_states)
        return hidden_states, residual

    def _qwen4_exp_use_dp_moe_gather(self) -> bool:
        return get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none()

    def _qwen4_exp_use_attn_tp_a2a_scatter(self) -> bool:
        return get_parallel().attn_tp_size > 1 and not get_moe_a2a_backend().is_none()

    def _run_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not self.config.num_experts:
            return self.mlp(hidden_states)

        use_dp_moe_gather = self._qwen4_exp_use_dp_moe_gather()
        use_attn_tp_a2a_scatter = self._qwen4_exp_use_attn_tp_a2a_scatter()

        if use_dp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        elif hidden_states.shape[0] == 0 and get_moe_a2a_backend().is_none():
            # Only safe to short-circuit an empty batch when the MoE holds no collective;
            # under deepep an idle DP rank must still join dispatch/combine or peers hang.
            return hidden_states

        attn_tp_chunks = None
        if use_attn_tp_a2a_scatter:
            attn_tp_size = get_parallel().attn_tp_size
            attn_tp_chunks = list(hidden_states.tensor_split(attn_tp_size))
            hidden_states = attn_tp_chunks[get_parallel().attn_tp_rank].contiguous()

        hidden_states = self.mlp(hidden_states, forward_batch)

        if use_dp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if should_use_dp_reduce_scatterv():
                get_tp_group().reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
        elif use_attn_tp_a2a_scatter:
            assert attn_tp_chunks is not None
            gathered = [torch.empty_like(t) for t in attn_tp_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        return hidden_states

    def _postprocess_qwen4_exp_layer(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.mlp_hyper_connection.combine(hidden_states, residual)
        return hidden_states, None


class Qwen4ExpLinearDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5LinearDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(hidden_states, forward_batch)

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


class Qwen4ExpAttentionDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5AttentionDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        config.attn_output_gate = True
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        from sglang.srt.layers.attention.qsa.config import is_qwen_qsa
        from sglang.srt.layers.attention.qsa.glue import build_qsa_indexer

        self.is_qsa = is_qwen_qsa(config)
        if self.is_qsa:
            self.indexer = build_qsa_indexer(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer" if prefix else "indexer",
                rotary_emb=self.rotary_emb,
            )
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def _compute_qsa_topk_indices(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.qsa.glue import (
            get_qsa_indexer_metadata,
            resolve_qsa_sparse_backend,
        )

        backend = get_attn_backend()
        sparse_backend = resolve_qsa_sparse_backend(backend)
        should_reuse = getattr(sparse_backend, "should_reuse_mtp_sparse_indices", None)
        if should_reuse is not None and should_reuse(forward_batch):
            # MTP decode steps reuse the draft-extend's target-aligned
            # selection; the indexer never runs inside the decode graph.
            return sparse_backend.lookup_mtp_sparse_indices(
                forward_batch, self.layer_id
            )
        indexer_metadata = get_qsa_indexer_metadata(
            backend, self.layer_id, forward_batch
        )
        topk_indices = self.indexer(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
        )
        should_capture = getattr(
            sparse_backend, "should_capture_mtp_sparse_indices", None
        )
        if should_capture is not None and should_capture(forward_batch):
            sparse_backend.capture_mtp_sparse_indices(
                topk_indices, forward_batch, self.layer_id, metadata=indexer_metadata
            )
        return topk_indices

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        overlap_indexer = (
            self.is_qsa
            and self.alt_stream is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] < _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD
        )
        attention_kwargs = {}
        if overlap_indexer:
            # The indexer chain reads only hidden_states/positions and writes
            # QSA-private pool buffers, so it runs concurrently with the main
            # qkv projection + norm/rope chain.
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                topk_indices = self._compute_qsa_topk_indices(
                    hidden_states, positions, forward_batch
                )

        q, k, v, gate = self._prepare_qkv_gate(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        if overlap_indexer:
            current_stream.wait_stream(self.alt_stream)
            # Allocated on alt_stream, consumed by attention on the current
            # stream; tell the caching allocator before alt_stream is reused.
            topk_indices.record_stream(current_stream)
            attention_kwargs["topk_indices"] = topk_indices
        elif self.is_qsa:
            attention_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )

        attn_output = self.attn(q, k, v, forward_batch, **attention_kwargs)
        if gate is not None:
            if attn_output.is_cuda:
                # The strided 3D gate view feeds the kernel directly, so the
                # gate reshape copy disappears along with the sigmoid + mul.
                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
            else:
                gate = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
                attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen4ExpAttentionDecoderLayer,
    "full_attention": Qwen4ExpAttentionDecoderLayer,
    "linear_attention": Qwen4ExpLinearDecoderLayer,
}


class Qwen4ExpModel(Qwen3_5ForCausalLM):
    decoder_layer_types = ALL_DECODER_LAYER_TYPES

    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            use_attn_tp_group=is_dp_attention_enabled(),
        )

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        super().__init__(config, quant_config, prefix, is_nextn)
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.has_ple = bool(config.ple_layer_ids)
        self.ple_ngram_size = int(config.ngram_size) if self.has_ple else None
        self.ple_ngram_eos_token_id = (
            int(config.eos_token_id) if self.ple_ngram_size is not None else None
        )
        if hasattr(self, "norm"):
            delattr(self, "norm")
        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(hc_config, use_combine=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        ple_batch = (
            _prepare_ple_batch(
                input_ids,
                forward_batch,
                ngram_size=self.ple_ngram_size,
                ngram_eos_token_id=self.ple_ngram_eos_token_id,
            )
            if self.has_ple
            else None
        )
        residual = None
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i + 1 < self.end_layer:
                next_ple = getattr(self.layers[i + 1], "ple", None)
                if next_ple is not None:
                    next_ple.start_prefetch(ple_batch, forward_batch)
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    ple_batch=ple_batch,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )

        _commit_ple_batch(ple_batch, forward_batch)

        hc_hidden_states = hidden_states
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        if not forward_batch.forward_mode.is_idle():
            return hidden_states, hc_hidden_states

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Qwen4ExpVLModel(Qwen4ExpModel):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.last_hc_hidden_states = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[Any] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.last_hc_hidden_states = None
        # mm routine passes input_ids=None; PLE needs the real ids.
        if input_ids is None:
            input_ids = forward_batch.input_ids
        model_output = super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=input_embeds,
        )
        if isinstance(model_output, tuple):
            hidden_states, self.last_hc_hidden_states = model_output
            return hidden_states
        return model_output


class Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration):
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None

    def __init__(
        self,
        config: Qwen4ExpConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen4ExpVLModel,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = (
            "mrope_section" in rope_config and not self.language_model_only
        )
        self.deepstack_visual_indexes = (
            self.visual.deepstack_visual_indexes if self.visual is not None else []
        )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        hc_hidden_states = self.model.last_hc_hidden_states
        if hc_hidden_states is not None and isinstance(output, LogitsProcessorOutput):
            output.hidden_states = hc_hidden_states
        return output

    def _load_qwen4_exp_ple_buffer(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        buffers: dict,
        loaded_buffers: Set[str],
    ) -> bool:
        if ".ple.ple_embedding." not in name:
            return False
        buffer_name = name.rsplit(".", 1)[-1]
        if buffer_name.startswith("hashstats_"):
            return True
        if buffer_name == "token_lookup":
            return True
        if buffer_name not in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
            "weight_scale",
        }:
            return False
        buffer = buffers.get(name)
        if buffer is None:
            return False
        if buffer.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(buffer.shape)}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
        loaded_buffers.add(name)
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # qwen4 checkpoints are qwen3.5 format (head-first in_proj). These merge head-first
            # into the fused in_proj_qkvz/in_proj_ba, which is correct because the linear-attn
            # layer uses Qwen3_5GatedDeltaNet (head-first forward), not qwen3-next's group-first.
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        num_experts = getattr(self.config, "num_experts", None)
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
            if num_experts is not None
            else []
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale_inv",
            "_weight_scale_inv",
            ".input_scale_inv",
            "_input_scale_inv",
            "_weight_scale",
            "_input_scale",
        )

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            for expert_id in range(num_experts):
                weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        def copy_ple_rows_to_tp_embedding(
            emb, loaded_weight: torch.Tensor, row_start: int, row_end: int
        ) -> None:
            tp_start = emb.shard_indices.org_vocab_start_index
            tp_end = emb.shard_indices.org_vocab_end_index
            ov_start = max(row_start, tp_start)
            ov_end = min(row_end, tp_end)
            if ov_start < ov_end:
                local_start = ov_start - tp_start
                src_start = ov_start - row_start
                n_rows = ov_end - ov_start
                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )

        def load_qwen4_exp_ple_shard(name: str, loaded_weight: torch.Tensor) -> bool:
            if ".ngram_embedding.shard_" not in name:
                return False
            import re

            match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)
            if not match:
                return False
            shard_idx = int(match.group(1))
            mod_prefix = name[: name.index(".ngram_embedding.shard_")]
            ple_mod = ple_modules.get(mod_prefix)
            if ple_mod is None:
                return False
            emb = ple_mod.ngram_embedding
            if getattr(emb, "nvfp4_packed", False):
                shard_size = (
                    emb.org_vocab_size + ple_num_sync_shards - 1
                ) // ple_num_sync_shards
                row_start = shard_idx * shard_size
                row_end = row_start + loaded_weight.shape[0]
                tp_start = emb.shard_indices.org_vocab_start_index
                tp_end = emb.shard_indices.org_vocab_end_index
                ov_start = max(row_start, tp_start)
                ov_end = min(row_end, tp_end)
                if ov_start < ov_end:
                    local_start = ov_start - tp_start
                    src_start = ov_start - row_start
                    n_rows = ov_end - ov_start
                    pk, gs = _nvfp4_quantize_rows(
                        loaded_weight[src_start : src_start + n_rows]
                    )
                    emb.weight_packed[local_start : local_start + n_rows].copy_(pk)
                    emb.weight_gscale[local_start : local_start + n_rows].copy_(gs)
                    del pk, gs
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                torch.cuda.empty_cache()
                return True
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb.weight.dtype != torch.float8_e4m3fn
            ):
                if isinstance(emb, Qwen4ExpPinnedHostEmbedding):
                    # offload gathers from pinned host memory; a swapped-in
                    # pageable tensor would fault in the Triton kernel.
                    raise ValueError(
                        "fp8 PLE auto-switch is unsupported with "
                        "ple_offload_embedding; set "
                        'text_config.ple_embedding_dtype="float8_e4m3fn" instead'
                    )
                logger.info(
                    "PLE embedding switched to fp8 storage: %s (%s)",
                    mod_prefix,
                    tuple(emb.weight.data.shape),
                )
                old_weight_data = emb.weight.data
                # StartupWeightLoadManager enforces tensor identity/dtype; this
                # swap breaks that contract if the model is ever enrolled.
                emb.weight = torch.nn.Parameter(
                    torch.empty_like(old_weight_data, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                del old_weight_data
                # params_dict was snapshotted before the loop; drop the stale
                # entry or it pins the old bf16 storage until load end.
                params_dict.pop(f"{mod_prefix}.ngram_embedding.weight", None)
                torch.cuda.empty_cache()
            if (
                emb.weight.dtype == torch.float8_e4m3fn
                and loaded_weight.dtype != torch.float8_e4m3fn
            ):
                if not getattr(load_qwen4_exp_ple_shard, "_warned_downcast", False):
                    load_qwen4_exp_ple_shard._warned_downcast = True
                    logger.warning(
                        "PLE checkpoint shards are %s but the embedding storage "
                        "is fp8 (ple_embedding_dtype / fp8 quant config); "
                        "downcasting is lossy",
                        loaded_weight.dtype,
                    )
            shard_size = (
                emb.org_vocab_size + ple_num_sync_shards - 1
            ) // ple_num_sync_shards
            shard_start = shard_idx * shard_size
            actual_rows = loaded_weight.shape[0]
            shard_end = shard_start + actual_rows
            copy_ple_rows_to_tp_embedding(emb, loaded_weight, shard_start, shard_end)
            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            return True

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())

        ple_modules = {
            mod_name: mod
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Qwen4ExpNGramEmbedding)
        }
        text_config = getattr(self.config, "text_config", self.config)
        ple_num_sync_shards = int(
            getattr(
                text_config,
                "split_ngram_parts",
                getattr(self.config, "split_ngram_parts", 512),
            )
        )
        loaded_params: Set[str] = set()
        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        skipped_visual_count = 0

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name and self.language_model_only:
                skipped_visual_count += 1
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if name.endswith(".k_proj.k_scale"):
                name = name.replace(".k_proj.k_scale", ".attn.k_scale")
            elif name.endswith(".v_proj.v_scale"):
                name = name.replace(".v_proj.v_scale", ".attn.v_scale")

            if self._load_qwen4_exp_ple_buffer(
                name, loaded_weight, buffers, loaded_buffers
            ):
                continue
            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
            if ".ple.ple_embedding.ngram_embedding." in name and name.endswith(
                ".weight"
            ):
                raise ValueError(
                    f"unsupported PLE weight layout (expected shard_N shards): {name}"
                )

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and "lm_head.weight" in params_dict
            ):
                lm_head_param = params_dict["lm_head.weight"]
                weight_loader = getattr(
                    lm_head_param, "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            is_fused_expert = (
                "experts.gate_up_proj" in name or "experts.down_proj" in name
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if (
                    mapped_name.endswith(ignore_suffixes)
                    and mapped_name not in params_dict
                ):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                is_expert_weight = False
                current_expert_params_mapping = (
                    fused_expert_params_mapping
                    if is_fused_expert
                    else expert_params_mapping
                )
                for mapping in current_expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                gate_weight,
                                "w1",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                up_weight,
                                "w3",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                        else:
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                    else:
                        if (
                            mapped_name.endswith(ignore_suffixes)
                            and mapped_name not in params_dict
                        ):
                            continue
                        param = params_dict[mapped_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = mapped_name
                    break
                else:
                    if is_expert_weight:
                        continue
                    if "visual" in name:
                        name = name.replace("attn.qkv.", "attn.qkv_proj.")
                        name = name.replace("model.visual.", "visual.")
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name.endswith("_scale") and name not in params_dict:
                        assert (
                            abs(loaded_weight.item() - 1.0) < 1e-6
                        ), f"Expected 1.0, got {loaded_weight.item()} in skipped {name}"
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            "Parameter %s not found while loading Qwen4-Exp VL weights",
                            name,
                        )
                        continue
            loaded_params.add(name)

        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if skipped_visual_count > 0:
            logger.info(
                f"[language_model_only] Qwen4 load_weights: skipped "
                f"{skipped_visual_count} visual weights"
            )

        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()

        # SDnvfp2: optional statistics capture OR load-time NVFP4
        # W4A16 (stage 3) first, then FP8 (stage 1) for the rest, then the hc mixers (stage 4)
        if not _maybe_install_dense_calib(self):
            _maybe_convert_dense_to_nvfp4(self)
            _maybe_convert_dense_to_fp8(self)
            _maybe_convert_hc_mixers(self)
            _maybe_prepare_draft_head_slice(self)

        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        if getattr(text_config, "num_experts", None) is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen4ExpForConditionalGeneration]


# =============================================================================
# SDnvfp2: load-time FP8 (e4m3, per-output-channel weight scale, dynamic
# per-token activation scale) for the BF16 dense GEMMs. See fp8_dense_patch.py.
# =============================================================================
import os as _fp8_os

_FP8_GROUPS = {
    "gdn": ("linear_attn.in_proj_qkvz", "linear_attn.out_proj"),
    # Flash-Next's attention decoder layer owns the projections directly:
    # model.layers.N.qkv_proj / model.layers.N.o_proj (suffix match covers both layouts)
    "attn": ("qkv_proj", "o_proj"),
    "shared": ("mlp.shared_expert.gate_up_proj", "mlp.shared_expert.down_proj"),
    "lm_head": ("lm_head",),
}
_FP8_MAX = 448.0


class _LoadTimeFp8LinearMethod:
    """quant_method replacement: weight is fp8 [K, N] (transposed view of a contiguous
    [N, K]), weight_scale fp32 [N, 1]. Activations quantized per token on the fly."""

    def __init__(self, name):
        self.name = name

    # the loader calls this on every module's quant_method after load_weights
    def process_weights_after_loading(self, layer):
        return None

    def create_weights(self, *a, **k):
        raise RuntimeError("load-time FP8 method is installed after weights exist")

    def apply(self, layer, x, bias=None):
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        return apply_fp8_linear(
            x, layer.weight, layer.weight_scale, input_scale=None, bias=bias,
            use_per_token_if_dynamic=True,
        )

    # embedding-style call used by some LogitsProcessor paths
    def embedding(self, layer, x):
        raise NotImplementedError


def _fp8_convert_module(name, mod):
    w = mod.weight.data
    if w.dtype not in (torch.bfloat16, torch.float16):
        return 0
    N, K = w.shape
    if N % 16 or K % 16:
        return 0
    wf = w.float()
    s = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / _FP8_MAX
    q = (wf / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).contiguous()
    del wf
    saved = w.numel() * w.element_size() - q.numel()
    mod.weight = torch.nn.Parameter(q.t(), requires_grad=False)          # [K, N] view
    mod.weight_scale = torch.nn.Parameter(s.to(torch.float32).contiguous(), requires_grad=False)
    mod.quant_method = _LoadTimeFp8LinearMethod(name)
    return saved


def _maybe_convert_dense_to_fp8(model):
    spec = _fp8_os.environ.get("SGLANG_DENSE_FP8", "").strip()
    if not spec:
        return
    groups = set(g.strip() for g in spec.split(",") if g.strip())
    if "all" in groups:
        groups = set(_FP8_GROUPS)
    suffixes = tuple(s for g in groups for s in _FP8_GROUPS.get(g, ()))
    if not suffixes:
        logger.warning("SGLANG_DENSE_FP8=%r matched no groups", spec)
        return
    converted, saved = 0, 0
    gdn_parents = []
    for name, mod in model.named_modules():
        if name.startswith("visual") or ".visual" in name or "mtp" in name:
            continue
        if not any(name == s or name.endswith("." + s) for s in suffixes):
            continue
        if not hasattr(mod, "weight") or not hasattr(mod, "quant_method"):
            continue
        n = _fp8_convert_module(name, mod)
        if n:
            converted += 1
            saved += n
            if name.endswith("linear_attn.in_proj_qkvz"):
                gdn_parents.append(name.rsplit(".", 1)[0])
    # GDN: the fused qkvz+ba buffer must be dropped (qkvz is fp8 now); keep ba bf16.
    mods = dict(model.named_modules())
    for p in gdn_parents:
        gdn = mods.get(p)
        if gdn is None:
            continue
        if getattr(gdn, "_fused_in_proj_weight", None) is not None:
            gdn.in_proj_ba.weight.data = gdn.in_proj_ba.weight.data.clone().contiguous()
            gdn._fused_in_proj_weight = None
    torch.cuda.empty_cache()
    logger.info(
        "[sdnvfp2 dense-fp8] groups=%s converted %d modules, freed %.2f GB (weights now fp8 + fp32 per-channel scale)",
        sorted(groups), converted, saved / 1e9,
    )


# =============================================================================
# SDnvfp2: load-time NVFP4 W4A16 for the dense GEMMs.
# Large-M path (prefill): SGLANG_DENSE_NVFP4_BIGM=bf16|fp8|off (default bf16: exact, 7.8 vs 20.8 ms at M=8192) switches GEMMs
# with >= SGLANG_DENSE_NVFP4_BIGM_ROWS rows (default 1024) to a per-call dequant of the 4-bit
# weight into an fp8 (per-row scale) or bf16 scratch followed by the corresponding dense
# GEMM; cute-dsl W4A16 is compute-bound there (0.7x bf16 at M=1024).
# =============================================================================
import triton as _nvfp4_triton
import triton.language as _nvfp4_tl


@_nvfp4_triton.jit
def _nvfp4_e2m1(c):
    """4-bit code (sign, e1, e0, m) -> value: e==0 -> m*0.5, else (1 + m/2) * 2^(e-1)."""
    e = (c >> 1) & 3
    m = (c & 1).to(_nvfp4_tl.float32)
    mag = _nvfp4_tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * _nvfp4_tl.exp2((e - 1).to(_nvfp4_tl.float32)))
    return _nvfp4_tl.where(((c >> 3) & 1) == 1, -mag, mag)


@_nvfp4_triton.jit
def _nvfp4_dequant_kernel(
    codes_ptr, sf_ptr, out_ptr, rowscale_ptr, alpha_ptr,
    N, K,
    stride_cn, stride_sn, stride_on,
    OUT_FP8: _nvfp4_tl.constexpr, BLOCK_N: _nvfp4_tl.constexpr, BLOCK_K: _nvfp4_tl.constexpr,
):
    """codes [N, K/2] uint8 (low nibble = even k), sf [N, K/16] e4m3-as-uint8, alpha fp32 [1].
    OUT_FP8=False: out bf16 [N, K] = code * sf * alpha.
    OUT_FP8=True:  out e4m3 [N, K] = code * sf * alpha / rowscale[n] (rowscale chosen so the
    row maximum lands on 448; see _nvfp4_row_scale)."""
    pid_n = _nvfp4_tl.program_id(0)
    pid_k = _nvfp4_tl.program_id(1)
    offs_n = pid_n * BLOCK_N + _nvfp4_tl.arange(0, BLOCK_N)
    offs_kb = pid_k * (BLOCK_K // 2) + _nvfp4_tl.arange(0, BLOCK_K // 2)      # packed byte index
    offs_kg = pid_k * (BLOCK_K // 16) + _nvfp4_tl.arange(0, BLOCK_K // 16)    # scale group index
    n_mask = offs_n < N
    packed = _nvfp4_tl.load(codes_ptr + offs_n[:, None] * stride_cn + offs_kb[None, :],
                     mask=n_mask[:, None] & (offs_kb[None, :] < K // 2), other=0)
    sf_u8 = _nvfp4_tl.load(sf_ptr + offs_n[:, None] * stride_sn + offs_kg[None, :],
                    mask=n_mask[:, None] & (offs_kg[None, :] < K // 16), other=0)
    sf = sf_u8.to(_nvfp4_tl.float8e4nv, bitcast=True).to(_nvfp4_tl.float32)          # [BN, BK/16]
    alpha = _nvfp4_tl.load(alpha_ptr)
    # expand each scale over its 8 packed bytes: [BN, BK/16] -> [BN, BK/2]
    sf8 = _nvfp4_tl.reshape(_nvfp4_tl.broadcast_to(sf[:, :, None], (BLOCK_N, BLOCK_K // 16, 8)), (BLOCK_N, BLOCK_K // 2))
    lo = (packed & 0xF).to(_nvfp4_tl.int32)
    hi = ((packed >> 4) & 0xF).to(_nvfp4_tl.int32)
    v_lo = _nvfp4_e2m1(lo) * sf8 * alpha
    v_hi = _nvfp4_e2m1(hi) * sf8 * alpha
    if OUT_FP8:
        rs = _nvfp4_tl.load(rowscale_ptr + offs_n, mask=n_mask, other=1.0)
        v_lo = v_lo / rs[:, None]
        v_hi = v_hi / rs[:, None]
    # interleave low/high back to k = 2i, 2i+1
    v = _nvfp4_tl.reshape(_nvfp4_tl.join(v_lo, v_hi), (BLOCK_N, BLOCK_K))
    offs_k = pid_k * BLOCK_K + _nvfp4_tl.arange(0, BLOCK_K)
    out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_k[None, :]
    o_mask = n_mask[:, None] & (offs_k[None, :] < K)
    if OUT_FP8:
        _nvfp4_tl.store(out_ptrs, v.to(_nvfp4_tl.float8e4nv), mask=o_mask)
    else:
        _nvfp4_tl.store(out_ptrs, v.to(_nvfp4_tl.bfloat16), mask=o_mask)


def _nvfp4_row_scale(sf_lin_u8, alpha, fp8_max=448.0):
    """Per-row fp32 scale for the fp8 scratch: the largest |value| in a row is at most
    6 * max_k(sf) * alpha (e2m1 max is 6), so dividing by that /448 never clips."""
    sf = sf_lin_u8.view(torch.float8_e4m3fn).float()
    return (sf.amax(dim=1) * 6.0 * float(alpha) / fp8_max).clamp(min=1e-30).contiguous()


def nvfp4_dequant(codes, sf_lin_u8, alpha, out_dtype, rowscale=None, out=None):
    """codes [N, K/2] uint8, sf_lin_u8 [N, K/16] uint8 (e4m3 bits, linear layout),
    alpha fp32 [1] tensor -> [N, K] bf16, or e4m3 scaled by rowscale [N] (fp32)."""
    N, K2 = codes.shape
    K = K2 * 2
    fp8 = out_dtype == torch.float8_e4m3fn
    if out is None:
        out = torch.empty(N, K, dtype=out_dtype, device=codes.device)
    if fp8:
        assert rowscale is not None
    BLOCK_N, BLOCK_K = 32, 256 if K % 256 == 0 else 128
    grid = (_nvfp4_triton.cdiv(N, BLOCK_N), _nvfp4_triton.cdiv(K, BLOCK_K))
    _nvfp4_dequant_kernel[grid](
        codes, sf_lin_u8, out, rowscale if fp8 else out, alpha,
        N, K, codes.stride(0), sf_lin_u8.stride(0), out.stride(0),
        OUT_FP8=fp8, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4,
    )
    return out

_NVFP4_FP8_MAX, _NVFP4_E2M1_MAX = 448.0, 6.0


def _nvfp4_sf_layout(flashinfer):
    layout = getattr(flashinfer, "SfLayout", None)
    if layout is None:
        from flashinfer.fp4_quantization import SfLayout as layout
    return layout.layout_128x4


class _LoadTimeNvfp4LinearMethod:
    """quant_method replacement: weight / weight_scale / weight_alpha hold the
    backend-prepared tensors from flashinfer.prepare_bf16_fp4_weights."""

    def __init__(self, name, backend):
        self.name = name
        self.backend = backend

    def process_weights_after_loading(self, layer):
        return None

    def create_weights(self, *a, **k):
        raise RuntimeError("load-time NVFP4 method is installed after weights exist")

    def apply(self, layer, x, bias=None):
        import flashinfer

        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if _NVFP4_BIGM != "off" and x2.shape[0] >= _NVFP4_BIGM_ROWS and hasattr(layer, "weight_codes"):
            if _NVFP4_BIGM == "bf16":
                w = nvfp4_dequant(layer.weight_codes, layer.weight_sf_lin, layer.weight_alpha, torch.bfloat16)
                out = torch.matmul(x2, w.t())
            else:
                from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

                w = nvfp4_dequant(layer.weight_codes, layer.weight_sf_lin, layer.weight_alpha,
                                  torch.float8_e4m3fn, rowscale=layer.weight_rowscale)
                out = apply_fp8_linear(x2, w.t(), layer.weight_rowscale.reshape(-1, 1),
                                       input_scale=None, use_per_token_if_dynamic=True)
        else:
            out = flashinfer.mm_bf16_fp4(x2, layer.weight, layer.weight_scale, layer.weight_alpha,
                                         backend=self.backend)
        if bias is not None:
            out = out + bias
        return out.reshape(*shp[:-1], out.shape[-1])

    def embedding(self, layer, x):
        raise NotImplementedError


_NVFP4_SIDECAR_USED = []
_NVFP4_BIGM = _fp8_os.environ.get("SGLANG_DENSE_NVFP4_BIGM", "bf16").strip().lower() or "bf16"
_NVFP4_BIGM_ROWS = int(_fp8_os.environ.get("SGLANG_DENSE_NVFP4_BIGM_ROWS", "1024") or 1024)


def _nvfp4_pack_weight(w, backend, keep_linear=True):
    """bf16 [N, K] -> (prepared packed codes, prepared block scales, alpha) for mm_bf16_fp4,
    plus (codes [N, K/2] u8, linear block scales [N, K/16] u8, rowscale fp32 [N]) for the
    large-M dequant path when keep_linear."""
    import flashinfer

    N, K = w.shape
    wf = w.float()
    g = (wf.abs().amax() / (_NVFP4_FP8_MAX * _NVFP4_E2M1_MAX)).clamp(min=1e-30)
    del wf
    ginv = (1.0 / g).reshape(1).to(torch.float32)
    alpha = g.reshape(1).to(torch.float32).contiguous()
    q, sf = flashinfer.nvfp4_quantize(w.contiguous(), ginv, sfLayout=_nvfp4_sf_layout(flashinfer), do_shuffle=False)
    prepared = flashinfer.prepare_bf16_fp4_weights(q.view(torch.uint8), sf, alpha, backend=backend)
    if not keep_linear or _NVFP4_BIGM == "off":
        return prepared, None
    _, sf_lin = flashinfer.nvfp4_quantize(w.contiguous(), ginv, sfLayout=flashinfer.SfLayout.layout_linear, do_shuffle=False)
    codes = q.view(torch.uint8).reshape(N, K // 2)
    if codes.data_ptr() != prepared[0].data_ptr():
        codes = codes.contiguous()          # prepare() made its own copy: keep the canonical one too
    sf_lin = sf_lin.view(torch.uint8).reshape(N, K // 16).contiguous()
    rowscale = _nvfp4_row_scale(sf_lin, alpha.item())
    return prepared, (codes, sf_lin, rowscale)


_NVFP4_CODES_DIR = _fp8_os.environ.get("SGLANG_DENSE_NVFP4_CODES_DIR", "").strip()


def _nvfp4_load_sidecar(name, N, K, backend):
    """Precomputed NVFP4 codes shipped with the checkpoint: <dir>/<module>.pt with
    codes u8 [N, K/2], sf_lin u8 [N, K/16] (e4m3 bits), alpha fp32 [1]. Returns the same
    tuple as _nvfp4_pack_weight or None."""
    import flashinfer

    if not _NVFP4_CODES_DIR:
        return None
    path = _fp8_os.path.join(_NVFP4_CODES_DIR, name + ".pt")
    if not _fp8_os.path.exists(path):
        return None
    d = torch.load(path, map_location="cuda")
    codes, sf_lin, alpha = d["codes"].contiguous(), d["sf_lin"].contiguous(), d["alpha"].float().reshape(1).contiguous()
    assert tuple(codes.shape) == (N, K // 2) and tuple(sf_lin.shape) == (N, K // 16), (name, codes.shape, sf_lin.shape)
    sf_sw = flashinfer.nvfp4_block_scale_interleave(sf_lin)
    prepared = flashinfer.prepare_bf16_fp4_weights(codes, sf_sw, alpha, backend=backend)
    if _NVFP4_BIGM == "off":
        return prepared, None
    return prepared, (codes, sf_lin, _nvfp4_row_scale(sf_lin, alpha.item()))


def _nvfp4_convert_module(name, mod, backend):
    w = mod.weight.data
    if w.dtype not in (torch.bfloat16, torch.float16):
        return 0
    N, K = w.shape
    if N % 16 or K % 16:
        return 0
    side = _nvfp4_load_sidecar(name, N, K, backend)
    if side is not None:
        (b_p, s_p, a_p), lin = side
        _NVFP4_SIDECAR_USED.append(name)
    else:
        (b_p, s_p, a_p), lin = _nvfp4_pack_weight(w.to(torch.bfloat16), backend)
    saved = w.numel() * w.element_size() - (b_p.numel() * b_p.element_size()
                                            + s_p.numel() * s_p.element_size())
    mod.weight = torch.nn.Parameter(b_p, requires_grad=False)
    mod.weight_scale = torch.nn.Parameter(s_p, requires_grad=False)
    mod.weight_alpha = torch.nn.Parameter(a_p, requires_grad=False)
    if lin is not None:
        codes, sf_lin, rowscale = lin
        mod.weight_codes = codes
        mod.weight_sf_lin = sf_lin
        mod.weight_rowscale = rowscale
        extra = sf_lin.numel() + rowscale.numel() * 4 + (0 if codes.data_ptr() == b_p.data_ptr() else codes.numel())
        saved -= extra
    mod.quant_method = _LoadTimeNvfp4LinearMethod(name, backend)
    return saved


def _maybe_convert_dense_to_nvfp4(model):
    spec = _fp8_os.environ.get("SGLANG_DENSE_NVFP4", "").strip()
    if not spec:
        return
    backend = _fp8_os.environ.get("SGLANG_DENSE_NVFP4_BACKEND", "cute-dsl").strip() or "cute-dsl"
    groups = set(g.strip() for g in spec.split(",") if g.strip())
    if "all" in groups:
        groups = set(_FP8_GROUPS)
    suffixes = tuple(s for g in groups for s in _FP8_GROUPS.get(g, ()))
    if not suffixes:
        logger.warning("SGLANG_DENSE_NVFP4=%r matched no groups", spec)
        return
    converted, saved = 0, 0
    gdn_parents = []
    for name, mod in model.named_modules():
        if name.startswith("visual") or ".visual" in name or "mtp" in name:
            continue
        if not any(name == s or name.endswith("." + s) for s in suffixes):
            continue
        if not hasattr(mod, "weight") or not hasattr(mod, "quant_method"):
            continue
        n = _nvfp4_convert_module(name, mod, backend)
        if n:
            converted += 1
            saved += n
            if name.endswith("linear_attn.in_proj_qkvz"):
                gdn_parents.append(name.rsplit(".", 1)[0])
    mods = dict(model.named_modules())
    for p in gdn_parents:  # same fused-buffer fixup as the FP8 stage
        gdn = mods.get(p)
        if gdn is None:
            continue
        if getattr(gdn, "_fused_in_proj_weight", None) is not None:
            gdn.in_proj_ba.weight.data = gdn.in_proj_ba.weight.data.clone().contiguous()
            gdn._fused_in_proj_weight = None
    torch.cuda.empty_cache()
    logger.info(
        "[sdnvfp2 dense-nvfp4] groups=%s backend=%s converted %d modules (%d from sidecar codes in %r), freed %.2f GB (W4A16: bf16 act x nvfp4 weight; big-M path %s from %d rows)",
        sorted(groups), backend, converted, len(_NVFP4_SIDECAR_USED), _NVFP4_CODES_DIR, saved / 1e9, _NVFP4_BIGM, _NVFP4_BIGM_ROWS,
    )


# =============================================================================
# SDnvfp2: optional activation-statistics capture (env-gated, off in normal serving).
# SGLANG_DENSE_CALIB_DIR=<dir>: the dense modules stay bf16; each module's weight is saved
# once to <dir>/W/<module>.pt at load, and its quant_method.apply is wrapped to accumulate
# H = sum x^T x (fp32, on device) over every call made outside CUDA-graph capture (i.e. the
# prefill of whatever you send). A background thread dumps <dir>/H.pt (all modules, cpu fp32)
# whenever <dir>/DUMP_NOW appears, then writes <dir>/H_DONE. Groups via SGLANG_DENSE_CALIB
# (default gdn,attn,lm_head).
# =============================================================================
_CALIB = {"mods": {}, "tokens": 0}


class _CalibApply:
    def __init__(self, name, inner, K):
        self.name, self.inner = name, inner
        self.H = torch.zeros(K, K, dtype=torch.float32, device="cuda")
        self.n = 0

    def __getattr__(self, item):            # everything else (process_weights_after_loading, ...)
        return getattr(self.inner, item)

    def apply(self, layer, x, bias=None):
        if not torch.cuda.is_current_stream_capturing():
            x2 = x.reshape(-1, x.shape[-1])
            if x2.shape[0] > 0:
                xf = x2.float()
                self.H.addmm_(xf.t(), xf)
                self.n += int(x2.shape[0])
        return self.inner.apply(layer, x, bias) if bias is not None else self.inner.apply(layer, x)


def _maybe_install_dense_calib(model):
    d = _fp8_os.environ.get("SGLANG_DENSE_CALIB_DIR", "").strip()
    if not d:
        return False
    import threading, time as _time
    spec = _fp8_os.environ.get("SGLANG_DENSE_CALIB", "gdn,attn,lm_head")
    groups = set(g.strip() for g in spec.split(",") if g.strip())
    if "all" in groups:
        groups = set(_FP8_GROUPS)
    suffixes = tuple(s for g in groups for s in _FP8_GROUPS.get(g, ()))
    _fp8_os.makedirs(_fp8_os.path.join(d, "W"), exist_ok=True)
    for name, mod in model.named_modules():
        if name.startswith("visual") or ".visual" in name or "mtp" in name:
            continue
        if not any(name == s or name.endswith("." + s) for s in suffixes):
            continue
        if not hasattr(mod, "weight") or not hasattr(mod, "quant_method"):
            continue
        w = mod.weight.data
        if w.dtype not in (torch.bfloat16, torch.float16) or w.dim() != 2:
            continue
        wp = _fp8_os.path.join(d, "W", name + ".pt")
        if not _fp8_os.path.exists(wp):
            torch.save({"weight": w.detach().to("cpu")}, wp)
        mod.quant_method = _CalibApply(name, mod.quant_method, w.shape[1])
        _CALIB["mods"][name] = mod
    logger.info("[sdnvfp2 dense-calib] capturing H for %d modules into %s (touch %s/DUMP_NOW to dump)",
                len(_CALIB["mods"]), d, d)

    def _watch():
        flag = _fp8_os.path.join(d, "DUMP_NOW")
        while True:
            _time.sleep(5)
            if _fp8_os.path.exists(flag):
                try:
                    torch.cuda.synchronize()
                    out = {n: {"H": m.quant_method.H.detach().to("cpu"), "n": m.quant_method.n}
                           for n, m in _CALIB["mods"].items()}
                    torch.save(out, _fp8_os.path.join(d, "H.pt"))
                    open(_fp8_os.path.join(d, "H_DONE"), "w").write(str(sum(v["n"] for v in out.values())))
                    logger.info("[sdnvfp2 dense-calib] dumped H for %d modules (%s tokens on the first)",
                                len(out), next(iter(out.values()))["n"] if out else 0)
                except Exception as exc:
                    logger.warning("[sdnvfp2 dense-calib] dump failed: %s", exc)
                try:
                    _fp8_os.remove(flag)
                except OSError:
                    pass

    threading.Thread(target=_watch, daemon=True).start()
    return True


# =============================================================================
# SDnvfp2 (inactive by default): hyperconnection mixer GEMMs (input_mix_weight_down / _up) in fp8 or nvfp4.
# =============================================================================
def _maybe_convert_hc_mixers(model):
    mode = _fp8_os.environ.get("SGLANG_DENSE_HC", "off").strip().lower()
    if mode in ("", "off", "0"):
        return
    if mode not in ("fp8", "nvfp4"):
        logger.warning("SGLANG_DENSE_HC=%r not in fp8|nvfp4|off; ignoring", mode)
        return
    import types
    import torch.nn.functional as _F
    from sglang.srt.layers.hyperconnection import GatedResidual

    backend = _fp8_os.environ.get("SGLANG_DENSE_NVFP4_BACKEND", "cute-dsl").strip() or "cute-dsl"

    def _to_fp8(w):
        wf = w.float()
        s = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / _FP8_MAX
        q = (wf / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).contiguous()
        return q.t(), s.to(torch.float32).contiguous()

    def _linear_fp8(x, wq, ws):
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        return apply_fp8_linear(x, wq, ws, input_scale=None, use_per_token_if_dynamic=True)

    def _linear_nvfp4(x, packed):
        import flashinfer

        b_p, s_p, a_p = packed
        return flashinfer.mm_bf16_fp4(x.contiguous(), b_p, s_p, a_p, backend=backend)

    def _mix(self, hyper_input):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype)
            return mixed_input, (hyper_input, hyper_input)
        if self.config.hc_per_branch_norm:
            normed = self.hc_norm(hyper_input)
        else:
            normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))).flatten(-2)
        x = normed.reshape(-1, normed.shape[-1]).to(torch.bfloat16)
        h = _F.silu(self._hc_lin(x, self._hc_down) / self.hc_count)
        g = torch.sigmoid(self._hc_lin(h, self._hc_up)).reshape(*normed.shape[:-1], -1)
        mixed = (g.unflatten(-1, (self.hc_count, self.hidden_size))
                 * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        return mixed.to(self.params_dtype), (hyper_input, normed)

    converted, saved = 0, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, GatedResidual) or "mtp" in name or "visual" in name:
            continue
        if not hasattr(mod, "input_mix_weight_down"):
            continue
        wd, wu = mod.input_mix_weight_down.weight.data, mod.input_mix_weight_up.weight.data
        if wd.dtype != torch.bfloat16 or wu.dtype != torch.bfloat16:
            continue
        before = (wd.numel() + wu.numel()) * 2
        if mode == "fp8":
            mod._hc_down, mod._hc_up = _to_fp8(wd), _to_fp8(wu)
            mod._hc_lin = staticmethod(lambda x, p: _linear_fp8(x, *p))
            after = sum(t.numel() * t.element_size() for p in (mod._hc_down, mod._hc_up) for t in p)
        else:
            mod._hc_down, mod._hc_up = _nvfp4_pack_weight(wd, backend, keep_linear=False)[0], _nvfp4_pack_weight(wu, backend, keep_linear=False)[0]
            mod._hc_lin = staticmethod(lambda x, p: _linear_nvfp4(x, p))
            after = sum(t.numel() * t.element_size() for p in (mod._hc_down, mod._hc_up)
                        for t in p if t is not None)
        # drop the bf16 originals (keep tiny placeholders so state_dict / repr still work)
        mod.input_mix_weight_down.weight = torch.nn.Parameter(wd.new_empty(0), requires_grad=False)
        mod.input_mix_weight_up.weight = torch.nn.Parameter(wu.new_empty(0), requires_grad=False)
        mod._jit_mix_ok = False
        mod.mix = types.MethodType(_mix, mod)
        converted += 1
        saved += before - after
    torch.cuda.empty_cache()
    logger.info("[sdnvfp2 dense-hc] mode=%s converted %d GatedResidual mixers, freed %.2f GB",
                mode, converted, saved / 1e9)


# =============================================================================
# SDnvfp2: draft lm_head vocabulary slice.
# SGLANG_DRAFT_LMHEAD_SLICE=<ids.pt> ({"ids": int64 [n]}): after the dense conversions the
# hot rows of the (fp8 / nvfp4 / bf16) target head are stashed once. The NEXTN draft shares
# the target's lm_head module (set_lm_head_from_target); the patched draft model
# (qwen4_exp_mtp_slice.py) sets _DRAFT_SLICE["active"] around its forward, and the wrapped
# quant_method then computes logits only for the hot rows and scatters them into a
# full-width logits tensor filled with the dtype minimum. Greedy verify keeps a draft token
# only when it equals the target argmax, so this changes proposals, never acceptances.
# =============================================================================
_DRAFT_SLICE = {"active": False, "ids": None, "kind": None, "w": None, "s": None, "a": None,
                "V": None, "backend": None, "calls": 0}


class _DraftSliceAwareMethod:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def apply(self, layer, x, bias=None):
        st = _DRAFT_SLICE
        if _fp8_os.environ.get("SGLANG_DRAFT_SLICE_DEBUG") == "1" and st.get("dbg", 0) < 8:
            st["dbg"] = st.get("dbg", 0) + 1
            logger.info("[sdnvfp2 draft-slice dbg] apply: active=%s rows=%s layer=%s capturing=%s",
                        st["active"], tuple(x.shape), id(layer), torch.cuda.is_current_stream_capturing())
        if not st["active"] or st["ids"] is None:
            return self.inner.apply(layer, x, bias) if bias is not None else self.inner.apply(layer, x)
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if st["kind"] == "fp8":
            from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear
            part = apply_fp8_linear(x2, st["w"], st["s"], input_scale=None, use_per_token_if_dynamic=True)
        elif st["kind"] == "nvfp4":
            import flashinfer
            part = flashinfer.mm_bf16_fp4(x2.to(torch.bfloat16).contiguous(), st["w"], st["s"], st["a"], backend=st["backend"])[:, : st["ids"].numel()]
        else:
            part = torch.nn.functional.linear(x2, st["w"])
        out = torch.full((x2.shape[0], st["V"]), torch.finfo(part.dtype).min, dtype=part.dtype, device=part.device)
        out[:, st["ids"]] = part
        st["calls"] += 1
        return out.reshape(*shp[:-1], st["V"])


def _maybe_prepare_draft_head_slice(model):
    path = _fp8_os.environ.get("SGLANG_DRAFT_LMHEAD_SLICE", "").strip()
    if not path:
        return
    lm = getattr(model, "lm_head", None)
    if lm is None or not hasattr(lm, "weight"):
        logger.warning("[sdnvfp2 draft-slice] no lm_head on the model; skipping")
        return
    ids = torch.load(path, map_location="cpu")["ids"].to(torch.int64).to("cuda")
    qm = lm.quant_method
    st = _DRAFT_SLICE
    if isinstance(qm, _LoadTimeFp8LinearMethod):
        q = lm.weight.data.t()                        # [N, K] contiguous fp8 (the stage stored q.t())
        st["w"] = q[ids].contiguous().t()             # [K, n] view of contiguous [n, K]
        st["s"] = lm.weight_scale.data[ids].contiguous()
        st["kind"], st["V"] = "fp8", q.shape[0]
        nbytes = st["w"].numel() + st["s"].numel() * 4
    elif "_LoadTimeNvfp4LinearMethod" in type(qm).__name__ and hasattr(lm, "weight_codes"):
        import flashinfer
        # the cute-dsl packer needs N % 64 == 0: pad the row list by repeating the last id and
        # drop the padded logits at scatter time (st["n"] real rows)
        pad = (-ids.numel()) % 64
        ids_p = torch.cat([ids, ids[-1:].expand(pad)]) if pad else ids
        codes = lm.weight_codes[ids_p].contiguous()
        sf_lin = lm.weight_sf_lin[ids_p].contiguous()
        sf_sw = flashinfer.nvfp4_block_scale_interleave(sf_lin)
        st["w"], st["s"], st["a"] = flashinfer.prepare_bf16_fp4_weights(codes, sf_sw, lm.weight_alpha.data.reshape(1).float().contiguous(), backend=qm.backend)
        st["kind"], st["V"], st["backend"] = "nvfp4", lm.weight_codes.shape[0], qm.backend
        nbytes = st["w"].numel() * st["w"].element_size() + st["s"].numel()
    else:
        w = lm.weight.data
        if w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
            logger.warning("[sdnvfp2 draft-slice] unsupported lm_head layout %s %s; skipping", tuple(w.shape), w.dtype)
            return
        st["w"] = w[ids].contiguous()
        st["kind"], st["V"] = "bf16", w.shape[0]
        nbytes = st["w"].numel() * 2
    st["ids"] = ids
    lm.quant_method = _DraftSliceAwareMethod(qm)
    logger.info("[sdnvfp2 draft-slice] %d of %d rows (%.1f %%) as %s, %.0f MB; draft steps read the slice, target verify the full head (model=%s lm_head id=%s inner=%s)",
                int(ids.numel()), st["V"], 100.0 * ids.numel() / st["V"], st["kind"], nbytes / 1e6,
                type(model).__name__, id(lm), type(qm).__name__)
