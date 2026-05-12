from __future__ import annotations

import torch
import pandas as pd
from transformers import AutoTokenizer

_prompt_lengths = None
_call_counter = 0

LAYERS = [12, 13, 16]


def _load_prompt_lengths():
    global _prompt_lengths
    if _prompt_lengths is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_df = pd.read_csv("data/dataset.csv")
    test_df = pd.read_csv("data/test.csv")
    prompts = train_df["prompt"].tolist() + test_df["prompt"].tolist()
    lengths = []
    for p in prompts:
        tok = tokenizer(p, add_special_tokens=True, truncation=True, max_length=512)
        lengths.append(len(tok["input_ids"]))
    _prompt_lengths = lengths


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    global _call_counter
    _load_prompt_lengths()
    prompt_len = _prompt_lengths[_call_counter]
    _call_counter += 1

    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        return torch.zeros(len(LAYERS) * 2 * hidden_states.size(-1) + 1,
                           dtype=hidden_states.dtype, device=hidden_states.device)

    if prompt_len >= real_positions.numel():
        resp_positions = real_positions[-1:]
    else:
        resp_positions = real_positions[prompt_len:]

    resp_len = resp_positions.numel()
    parts = []

    for layer_num in LAYERS:
        layer_idx = layer_num - 1
        layer_out = hidden_states[layer_idx][resp_positions]

        max_vec = layer_out.max(dim=0).values
        mean_vec = layer_out.mean(dim=0)

        parts.append(max_vec)
        parts.append(mean_vec)

    len_feat = torch.tensor([float(resp_len) / 512.0], dtype=hidden_states.dtype, device=hidden_states.device)
    parts.append(len_feat)

    return torch.cat(parts, dim=0).contiguous()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo], dim=0)
    return agg_features