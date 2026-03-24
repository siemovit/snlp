from __future__ import annotations

import hashlib
import json
import math
import random
import gc
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
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


def build_patch_specs(
    method_name: str,
    k: int,
    base_layer: int,
    model,
    sv_bank,
    gate_bank,
    alpha: float = 1.0,
    sae_release: str | None = None,
    sae_device: str | None = None,
    sae_dtype: torch.dtype | None = None,
    gate_threshold: float = 0.0,
):
    if method_name == "No SV":
        return []

    specs = []
    for layer_idx in window_layers(base_layer, k, model):
        if layer_idx not in sv_bank:
            continue
        gate_fn = None
        if method_name.startswith("SAE"):
            feature_indices = gate_bank.get(layer_idx)
            if feature_indices is None:
                continue
            if sae_release is None:
                raise ValueError("sae_release is required to build SAE-gated patch specs.")
            sae = try_load_sae_for_layer(
                sae_release,
                layer_idx,
                sae_device or next(model.parameters()).device.type,
                dtype=sae_dtype,
            )
            gate_fn = make_sae_gate_fn(sae, feature_indices, threshold=gate_threshold)
        specs.append((layer_idx, alpha * sv_bank[layer_idx], gate_fn))
    return specs


@lru_cache(maxsize=None)
def _list_repo_files_cached(repo_id: str) -> Tuple[str, ...]:
    return tuple(HfApi().list_repo_files(repo_id=repo_id, repo_type="model"))


def _gemma_repo_id_for_release(release: str) -> str:
    if "/" in release:
        return release
    return f"google/{release}"


def _sort_gemma_sae_id(sae_id: str) -> tuple[int, int]:
    width_match = re.search(r"/width_(\d+)k/", sae_id)
    width = int(width_match.group(1)) if width_match else 10**9
    l0_match = re.search(r"/average_l0_(\d+)$", sae_id)
    l0_value = int(l0_match.group(1)) if l0_match else 10**9
    return width, l0_value


@lru_cache(maxsize=None)
def _discover_gemma_sae_ids(release: str, layer_idx: int) -> Tuple[str, ...]:
    repo_id = _gemma_repo_id_for_release(release)
    prefix = f"layer_{layer_idx}/"
    candidates = set()
    for path in _list_repo_files_cached(repo_id):
        if not path.startswith(prefix):
            continue
        if not path.endswith("params.npz"):
            continue
        candidates.add(path.rsplit("/", 1)[0])
    return tuple(sorted(candidates, key=_sort_gemma_sae_id))


def try_load_sae_for_layer(release: str, layer_idx: int, device: str, dtype: torch.dtype | None = None):
    if "gemma-scope" in release:
        candidates = list(_discover_gemma_sae_ids(release, layer_idx))
    else:
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


def _stable_cache_key(kind: str, cache_metadata: dict, layer_idx: int) -> str:
    payload = {"kind": kind, "layer_idx": layer_idx, **cache_metadata}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cache_file(cache_dir: Path, kind: str, cache_key: str, layer_idx: int) -> Path:
    return cache_dir / kind / f"{kind}_layer{layer_idx}_{cache_key}.pt"


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
    cache_dir: str | Path | None = None,
    cache_metadata: dict | None = None,
    gate_topk: int = 2,
):
    sv_bank: Dict[int, torch.Tensor] = {}
    gate_bank: Dict[int, List[int]] = {}
    source_lang_idx = target_lan.index(source_lang)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    cache_metadata = cache_metadata or {}

    if cache_dir is not None:
        (cache_dir / "sv").mkdir(parents=True, exist_ok=True)
        (cache_dir / "top_idx").mkdir(parents=True, exist_ok=True)

    for layer_idx in window_layers_to_use:
        sv_cache_path = None
        top_idx_cache_path = None
        if cache_dir is not None:
            sv_cache_path = _cache_file(cache_dir, "sv", _stable_cache_key("sv", cache_metadata, layer_idx), layer_idx)
            top_idx_cache_path = _cache_file(
                cache_dir, "top_idx", _stable_cache_key("top_idx", cache_metadata, layer_idx), layer_idx
            )

        if sv_cache_path is not None and sv_cache_path.exists():
            sv_bank[layer_idx] = torch.load(sv_cache_path, map_location="cpu", weights_only=True).to(device)
            if progress_callback is not None:
                for _ in range(len(pos_texts) + len(neg_texts)):
                    progress_callback()
        else:
            sv_bank[layer_idx] = compute_steering_vector(
                model,
                tokenizer,
                pos_texts,
                neg_texts,
                layer_idx,
                device=device,
                normalize=False,
                progress_callback=progress_callback,
            )
            if sv_cache_path is not None:
                torch.save(sv_bank[layer_idx].detach().cpu(), sv_cache_path)

        sae_layer = try_load_sae_for_layer(release, layer_idx, sae_device or device, dtype=sae_dtype)
        if top_idx_cache_path is not None and top_idx_cache_path.exists():
            top_idx_layer = torch.load(top_idx_cache_path, map_location="cpu", weights_only=True)
            if progress_callback is not None:
                for _ in range(len(target_lan) * train_n):
                    progress_callback()
        else:
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
            if top_idx_cache_path is not None:
                torch.save(top_idx_layer.detach().cpu(), top_idx_cache_path)
        topk_src = top_idx_layer[source_lang_idx, :gate_topk].detach().cpu().tolist()
        gate_bank[layer_idx] = topk_src
        del top_idx_layer
        del sae_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
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
