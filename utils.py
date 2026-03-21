from __future__ import annotations

import platform
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_LANGS = ["en", "es", "fr", "ja", "ko", "pt", "th", "vi", "zh", "ar"]

LANG_CODE_TO_NAME = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "th": "Thai",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ar": "Arabic",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_safe_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bytes_to_gb(value: int) -> float:
    return value / 1024**3


def get_device_memory_report(device: str) -> dict | None:
    if device == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_gb": _bytes_to_gb(free_bytes),
            "total_gb": _bytes_to_gb(total_bytes),
            "allocated_gb": _bytes_to_gb(torch.cuda.memory_allocated()),
            "reserved_gb": _bytes_to_gb(torch.cuda.memory_reserved()),
            "max_allocated_gb": _bytes_to_gb(torch.cuda.max_memory_allocated()),
        }
    return None


def print_device_memory_report(device: str, prefix: str) -> None:
    report = get_device_memory_report(device)
    if report is None:
        return
    print(
        f"{prefix} | cuda free={report['free_gb']:.2f}GB / {report['total_gb']:.2f}GB total | "
        f"allocated={report['allocated_gb']:.2f}GB | reserved={report['reserved_gb']:.2f}GB | "
        f"peak={report['max_allocated_gb']:.2f}GB"
    )


def ensure_min_free_memory(device: str, min_free_gb: float, stage: str) -> None:
    report = get_device_memory_report(device)
    if report is None:
        return
    if report["free_gb"] < min_free_gb:
        raise RuntimeError(
            f"Stopping before {stage}: only {report['free_gb']:.2f}GB CUDA memory free, "
            f"below threshold {min_free_gb:.2f}GB."
        )


def resolve_torch_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype != "auto":
        raise ValueError(f"Unsupported dtype: {dtype}")

    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(model_path: str | Path, device: str = "cpu", dtype: str = "auto"):
    torch_dtype = resolve_torch_dtype(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    return model, tokenizer


def load_multilingual_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    return pd.read_json(dataset_path, lines=True)


def build_language_texts(
    df: pd.DataFrame,
    target_langs: Iterable[str] = TARGET_LANGS,
) -> Dict[str, List[str]]:
    lang_texts: Dict[str, List[str]] = {}
    for code in target_langs:
        lang_texts[code] = df.loc[df["lan"] == code, "text"].tolist()
    return lang_texts


def flatten_language_texts(
    lang_texts: Dict[str, List[str]],
    target_langs: Iterable[str] = TARGET_LANGS,
) -> List[str]:
    flat: List[str] = []
    for code in target_langs:
        flat.extend(lang_texts[code])
    return flat


def build_lang_split(
    lang_texts: Dict[str, List[str]],
    train_n: int,
    eval_n: int,
    *,
    target_langs: Iterable[str] = TARGET_LANGS,
) -> Dict[str, Dict[str, List[str]]]:
    required = train_n + eval_n
    split: Dict[str, Dict[str, List[str]]] = {}
    for code in target_langs:
        texts = lang_texts[code]
        if len(texts) < required:
            raise ValueError(f"Language {code} has {len(texts)} texts, expected at least {required}.")
        split[code] = {
            "train": texts[:train_n],
            "eval": texts[train_n : train_n + eval_n],
        }
    return split


def gather_residual_activations(model, target_layer: int, inputs: torch.Tensor) -> torch.Tensor:
    target_act = None

    def gather_target_act_hook(_mod, _inputs, outputs):
        nonlocal target_act
        target_act = outputs[0] if isinstance(outputs, tuple) else outputs
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    try:
        with torch.no_grad():
            _ = model.forward(inputs)
    finally:
        handle.remove()
    if target_act is None:
        raise RuntimeError(f"Failed to capture residual activations for layer {target_layer}.")
    return target_act


def compute_top_index_per_lan_for_layer(
    model,
    tokenizer,
    sae,
    layer: int,
    target_lan: List[str],
    multilingual_texts: List[str],
    device: str,
    n_texts_per_lan: int = 10,
    progress_callback=None,
):
    block_size = len(multilingual_texts) // len(target_lan)
    avg_act_per_lan = []
    sae_device = next(sae.parameters()).device

    for i, lan in enumerate(target_lan):
        lang_texts = multilingual_texts[i * block_size : (i + 1) * block_size][:n_texts_per_lan]
        running_sum = None
        total_tokens = 0
        for text in lang_texts:
            inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
            target_act = gather_residual_activations(model, layer, inputs)
            sae_act = sae.encode(target_act.to(device=sae_device, dtype=torch.float32)).cpu()
            if sae_act.ndim == 2:
                sae_act = sae_act.unsqueeze(0)
            token_sum = sae_act.sum(dim=1).sum(dim=0)
            token_count = sae_act.shape[0] * sae_act.shape[1]
            if running_sum is None:
                running_sum = token_sum
            else:
                running_sum = running_sum + token_sum
            total_tokens += token_count
            if progress_callback is not None:
                progress_callback()
        if running_sum is None or total_tokens == 0:
            raise ValueError(f"No SAE activations collected for language {lan} at layer {layer}.")
        avg_act_per_lan.append((running_sum / total_tokens).unsqueeze(0))
    avg_act_per_lan = torch.cat(avg_act_per_lan)

    top_index_per_lan = []
    top_values_per_lan = []
    for i in range(len(avg_act_per_lan)):
        mean = avg_act_per_lan[i]
        gamma = torch.cat([avg_act_per_lan[:i], avg_act_per_lan[i + 1 :]]).mean(0)
        values, indices = torch.sort(mean - gamma, descending=True)
        top_values_per_lan.append(values)
        top_index_per_lan.append(indices)

    return torch.stack(top_index_per_lan), torch.stack(top_values_per_lan)
