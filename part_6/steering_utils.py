from __future__ import annotations

import math
import random
from contextlib import contextmanager
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import SAE

from utils import LANG_CODE_TO_NAME, compute_top_index_per_lan_for_layer


def get_layer_module(model, layer_idx: int):
    return model.model.layers[layer_idx]


def num_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def window_layers(start_layer: int, k: int, model) -> List[int]:
    end = min(start_layer + k, num_layers(model))
    return list(range(start_layer, end))


def mean_layer_activation_for_texts(
    model,
    tokenizer,
    texts: Iterable[str],
    layer_idx: int,
    device: str,
    progress_callback=None,
) -> torch.Tensor:
    pooled = []

    def hook_fn(_m, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        pooled.append(hidden.detach().mean(dim=1).cpu())
        return out

    handle = get_layer_module(model, layer_idx).register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for text in texts:
                ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
                _ = model(ids)
                if progress_callback is not None:
                    progress_callback()
    finally:
        handle.remove()

    return torch.cat(pooled, dim=0).mean(dim=0)


def compute_steering_vector(
    model,
    tokenizer,
    pos_texts,
    neg_texts,
    layer_idx: int,
    device: str,
    normalize: bool = True,
    progress_callback=None,
):
    pos_mean = mean_layer_activation_for_texts(
        model, tokenizer, pos_texts, layer_idx, device, progress_callback=progress_callback
    )
    neg_mean = mean_layer_activation_for_texts(
        model, tokenizer, neg_texts, layer_idx, device, progress_callback=progress_callback
    )
    vec = (pos_mean - neg_mean).to(device)
    if normalize:
        vec = vec / (vec.norm() + 1e-8)
    return vec


@contextmanager
def apply_layer_patches(model, patch_specs):
    handles = []

    def make_hook(delta_vec, gate_fn=None):
        def hook_fn(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            rest = out[1:] if isinstance(out, tuple) else None
            delta = delta_vec.to(hidden.device, dtype=hidden.dtype).view(1, 1, -1)
            if gate_fn is None:
                hidden_new = hidden + delta
            else:
                gate = gate_fn(hidden).to(hidden.device, dtype=hidden.dtype)
                hidden_new = hidden + gate * delta
            if rest is None:
                return hidden_new
            return (hidden_new,) + rest

        return hook_fn

    for layer_idx, delta_vec, gate_fn in patch_specs:
        handles.append(get_layer_module(model, layer_idx).register_forward_hook(make_hook(delta_vec, gate_fn)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def build_patch_specs(method_name: str, k: int, base_layer: int, model, sv_bank, gate_bank, alpha: float = 1.0):
    if method_name == "No SV":
        return []

    specs = []
    for layer_idx in window_layers(base_layer, k, model):
        if layer_idx not in sv_bank:
            continue
        gate_fn = gate_bank.get(layer_idx) if method_name.startswith("SAE") else None
        specs.append((layer_idx, alpha * sv_bank[layer_idx], gate_fn))
    return specs


def try_load_sae_for_layer(release: str, layer_idx: int, device: str, dtype: torch.dtype | None = None):
    candidates = [
        f"layer_{layer_idx}",
        f"layer_{layer_idx}/width_16k/canonical",
    ]
    for candidate in candidates:
        try:
            sae = SAE.from_pretrained(release, candidate)
            if dtype is not None:
                return sae.to(device=device, dtype=dtype)
            return sae.to(device)
        except Exception:
            continue
    raise ValueError(f"Could not load SAE for layer {layer_idx} from release {release}.")


def make_sae_gate_fn(sae, feature_indices, threshold: float = 0.0):
    idx = torch.as_tensor(feature_indices, dtype=torch.long)
    sae_device = next(sae.parameters()).device

    def gate_fn(hidden_states):
        acts = sae.encode(hidden_states.to(device=sae_device, dtype=torch.float32))
        picked = acts[..., idx.to(acts.device)]
        return (picked > threshold).any(dim=-1, keepdim=True).to(hidden_states.dtype)

    return gate_fn


def build_sv_bank_and_gates(
    window_layers_to_use: Iterable[int],
    model,
    tokenizer,
    pos_texts,
    neg_texts,
    target_lan: List[str],
    multilingual_texts: List[str],
    source_lang: str,
    release: str,
    device: str,
    train_n: int,
    sae_device: str | None = None,
    sae_dtype: torch.dtype | None = None,
    memory_report_fn=None,
    progress_callback=None,
):
    sv_bank: Dict[int, torch.Tensor] = {}
    gate_bank: Dict[int, object] = {}
    source_lang_idx = target_lan.index(source_lang)

    for layer_idx in window_layers_to_use:
        sv_bank[layer_idx] = compute_steering_vector(
            model,
            tokenizer,
            pos_texts,
            neg_texts,
            layer_idx,
            device=device,
            normalize=True,
            progress_callback=progress_callback,
        )
        sae_layer = try_load_sae_for_layer(release, layer_idx, sae_device or device, dtype=sae_dtype)
        top_idx_layer, _ = compute_top_index_per_lan_for_layer(
            model,
            tokenizer,
            sae_layer,
            layer_idx,
            target_lan,
            multilingual_texts,
            device,
            n_texts_per_lan=train_n,
            progress_callback=progress_callback,
        )
        top2_src = top_idx_layer[source_lang_idx, :2].detach().cpu().tolist()
        gate_bank[layer_idx] = make_sae_gate_fn(sae_layer, top2_src, threshold=0.0)
        if memory_report_fn is not None:
            memory_report_fn(f"After layer {layer_idx} bank/gate build")

    return sv_bank, gate_bank


def target_token_ce_from_prompt(model, tokenizer, prompt: str, target_word: str, device: str, patch_specs=None) -> float:
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    if ids.size(1) < 1:
        return math.nan
    attention_mask = torch.ones_like(ids)

    target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)
    if not target_ids:
        return math.nan

    with torch.no_grad():
        if patch_specs:
            with apply_layer_patches(model, patch_specs):
                out = model(ids, attention_mask=attention_mask)
        else:
            out = model(ids, attention_mask=attention_mask)

    logits = out.logits[:, -1, :]
    y = torch.tensor([target_ids[0]], device=logits.device)
    return F.cross_entropy(logits, y).item()


def lm_ce_loss_on_text(model, tokenizer, text: str, device: str, patch_specs=None) -> float:
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    if ids.size(1) < 2:
        return math.nan
    attention_mask = torch.ones_like(ids)

    labels = ids.clone()
    labels[:, 0] = -100
    with torch.no_grad():
        if patch_specs:
            with apply_layer_patches(model, patch_specs):
                out = model(ids, attention_mask=attention_mask, labels=labels)
        else:
            out = model(ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.item())


def set_generation_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_continuation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    patch_specs=None,
    deterministic: bool = True,
    seed: int = 0,
    use_cache_when_patched: bool = False,
    **gen_kwargs,
) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    attention_mask = torch.ones_like(ids)

    if deterministic:
        set_generation_seed(seed)

    default_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not deterministic,
        "num_beams": 1,
        "use_cache": True if not patch_specs else bool(use_cache_when_patched), # If patched, default to not using cache to ensure patches are applied at every step, but allow override for efficiency when using a small number of patches.
    }
    if tokenizer.eos_token_id is not None and tokenizer.pad_token_id is None:
        default_kwargs["pad_token_id"] = tokenizer.eos_token_id
    default_kwargs.update(gen_kwargs)

    with torch.no_grad():
        if patch_specs:
            with apply_layer_patches(model, patch_specs):
                out = model.generate(ids, attention_mask=attention_mask, **default_kwargs)
        else:
            out = model.generate(ids, attention_mask=attention_mask, **default_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def first_n_words(text: str, n_words: int = 20) -> str:
    return " ".join(text.strip().split()[:n_words])


def nanmean(values) -> float:
    return float(np.nanmean(values)) if values else float("nan")


def build_lid_prompt(text: str) -> str:
    return (
        "Identify the language of the following text in one word.\n"
        f"Text: {text}\n"
        "Language:"
    )


def build_cont_prompt(text: str, target_lang_name: str) -> str:
    return (
        f"Continue the following text in {target_lang_name}.\n"
        f"Text: {text}\n"
        "Continuation:"
    )


def normalize_openlid_label(label: str) -> str:
    label = label.lower().replace("__label__", "")
    if label in LANG_CODE_TO_NAME:
        return label
    if "_" in label:
        prefix = label.split("_")[0]
        if prefix in LANG_CODE_TO_NAME:
            return prefix
    if "-" in label:
        prefix = label.split("-")[0]
        if prefix in LANG_CODE_TO_NAME:
            return prefix
    return label
