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


def load_model_and_tokenizer(model_path: str | Path, device: str = "cpu"):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
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
):
    sae_activations_per_language = {}
    block_size = len(multilingual_texts) // len(target_lan)

    for i, lan in enumerate(target_lan):
        lang_texts = multilingual_texts[i * block_size : (i + 1) * block_size][:n_texts_per_lan]
        activations = []
        for text in lang_texts:
            inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
            target_act = gather_residual_activations(model, layer, inputs)
            sae_act = sae.encode(target_act).cpu()
            if sae_act.ndim == 2:
                sae_act = sae_act.unsqueeze(0)
            activations.append(sae_act)
        sae_activations_per_language[lan] = activations

    avg_act_per_lan = []
    for lan in target_lan:
        avg_act_per_lan.append(torch.cat(sae_activations_per_language[lan], dim=1).mean(-2))
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
