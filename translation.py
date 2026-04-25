from __future__ import annotations

from functools import lru_cache
import importlib.util
import sys
import types
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from config import LANGUAGE_NAME_TO_CODE, settings


FLORES_TARGET_BY_LANGUAGE = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
}

SOURCE_LANGUAGE = "hin_Deva"


def _ensure_transformers_onnx_shim() -> None:
    if "transformers.onnx" in sys.modules and "transformers.onnx.utils" in sys.modules:
        return

    onnx_module = types.ModuleType("transformers.onnx")
    utils_module = types.ModuleType("transformers.onnx.utils")

    class OnnxConfig:  # pragma: no cover - compatibility shim
        default_fixed_batch = 2
        default_fixed_sequence = 8

    class OnnxConfigWithPast(OnnxConfig):  # pragma: no cover - compatibility shim
        pass

    class OnnxSeq2SeqConfigWithPast(OnnxConfig):  # pragma: no cover - compatibility shim
        pass

    def compute_effective_axis_dimension(dim: int | None, fixed_dimension: int) -> int:
        if dim in (None, -1):
            return fixed_dimension
        return dim

    onnx_module.OnnxConfig = OnnxConfig
    onnx_module.OnnxConfigWithPast = OnnxConfigWithPast
    onnx_module.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    utils_module.compute_effective_axis_dimension = compute_effective_axis_dimension

    sys.modules["transformers.onnx"] = onnx_module
    sys.modules["transformers.onnx.utils"] = utils_module


def _ensure_transformers_tokenizer_shim() -> None:
    original_setattr = getattr(PreTrainedTokenizerBase, "_codex_original_setattr", None)
    if original_setattr is not None:
        return

    original_setattr = PreTrainedTokenizerBase.__setattr__

    def patched_setattr(self, key, value):  # pragma: no cover - compatibility shim
        if key in {
            "unk_token",
            "bos_token",
            "eos_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
            "additional_special_tokens",
        } and not hasattr(self, "_special_tokens_map"):
            object.__setattr__(self, "_special_tokens_map", {})
        return original_setattr(self, key, value)

    PreTrainedTokenizerBase._codex_original_setattr = original_setattr  # type: ignore[attr-defined]
    PreTrainedTokenizerBase.__setattr__ = patched_setattr  # type: ignore[assignment]


def _ensure_transformers_model_init_shim() -> None:
    original_init_weights = getattr(PreTrainedModel, "_codex_original_init_weights", None)
    if original_init_weights is not None:
        return

    original_init_weights = PreTrainedModel.init_weights

    def patched_init_weights(self):  # pragma: no cover - compatibility shim
        try:
            return original_init_weights(self)
        except TypeError as exc:
            message = str(exc)
            if "recompute_mapping" not in message or "unexpected keyword argument" not in message:
                raise
            if getattr(self, "_codex_init_weights_retry", False):
                raise
            setattr(self, "_codex_init_weights_retry", True)
            try:
                return self.tie_weights()
            finally:
                try:
                    delattr(self, "_codex_init_weights_retry")
                except Exception:
                    pass

    PreTrainedModel._codex_original_init_weights = original_init_weights  # type: ignore[attr-defined]
    PreTrainedModel.init_weights = patched_init_weights  # type: ignore[assignment]


def _ensure_namespace_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
        return

    if not hasattr(module, "__path__"):
        module.__path__ = [str(path)]  # type: ignore[attr-defined]


def _ensure_indictrans_model_shim() -> None:
    module_name = (
        "transformers_modules.indictrans2_hyphen_indic_hyphen_indic_hyphen_dist_hyphen_320M."
        "modeling_indictrans"
    )
    module = sys.modules.get(module_name)
    if module is None:
        cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
        package_path = cache_root / "indictrans2_hyphen_indic_hyphen_indic_hyphen_dist_hyphen_320M"
        module_path = package_path / "modeling_indictrans.py"
        if not module_path.exists():
            return

        _ensure_namespace_package("transformers_modules", cache_root)
        _ensure_namespace_package(
            "transformers_modules.indictrans2_hyphen_indic_hyphen_indic_hyphen_dist_hyphen_320M",
            package_path,
        )

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    model_cls = getattr(module, "IndicTransForConditionalGeneration", None)
    if model_cls is None:
        return

    original_tie_weights = getattr(model_cls, "_codex_original_tie_weights", None)
    if original_tie_weights is not None:
        return

    original_tie_weights = model_cls.tie_weights

    def patched_tie_weights(self, *args, **kwargs):  # pragma: no cover - compatibility shim
        return original_tie_weights(self)

    model_cls._codex_original_tie_weights = original_tie_weights  # type: ignore[attr-defined]
    model_cls.tie_weights = patched_tie_weights


def _split_hindi_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    try:
        sentences = [part.strip() for part in stripped.replace("।", ".").split(".")]
    except Exception:
        return [stripped]

    cleaned = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    return cleaned or [stripped]


class IndicTrans2Translator:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = torch.device(
            "cuda"
            if settings.translation_device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if not settings.translation_ckpt_dir:
            raise RuntimeError(
                "Set INDICTRANS2_CKPT_DIR to the downloaded IndicTrans2 HF checkpoint folder."
            )

        ckpt_dir = Path(settings.translation_ckpt_dir)
        if not ckpt_dir.exists():
            raise RuntimeError(f"IndicTrans2 checkpoint folder does not exist: {ckpt_dir}")

        _ensure_transformers_onnx_shim()
        _ensure_transformers_tokenizer_shim()
        _ensure_transformers_model_init_shim()
        _ensure_indictrans_model_shim()
        self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._model.to(self._device)
        self._model.eval()

    def _translate_sentences(self, sentences: list[str], target_language: str) -> list[str]:
        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None

        target_flores = FLORES_TARGET_BY_LANGUAGE[target_language]
        translated_sentences: list[str] = []

        for start in range(0, len(sentences), 4):
            batch = sentences[start : start + 4]
            prompts = [f"{SOURCE_LANGUAGE} {target_flores} {sentence}" for sentence in batch]
            inputs = self._tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                generated_tokens = self._model.generate(
                    **inputs,
                    num_beams=5,
                    max_new_tokens=256,
                )

            decoded = self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            translated_sentences.extend(sentence.strip() for sentence in decoded)

        return translated_sentences

    def translate(self, text: str, target_language: str) -> str:
        target_language = target_language.lower().strip()
        target_language = LANGUAGE_NAME_TO_CODE.get(target_language, target_language)
        if target_language not in FLORES_TARGET_BY_LANGUAGE:
            raise ValueError(f"Unsupported target language: {target_language}")

        sentences = _split_hindi_sentences(text)
        if not sentences:
            return ""

        translated_sentences = self._translate_sentences(sentences, target_language)
        return " ".join(sentence.strip() for sentence in translated_sentences if sentence.strip()).strip()


@lru_cache(maxsize=1)
def get_translator() -> IndicTrans2Translator:
    return IndicTrans2Translator()


def translate_hindi(text: str, target_language: str) -> str:
    return get_translator().translate(text, target_language)


def _split_hindi_sentences(text: str) -> list[str]:  # type: ignore[no-redef]
    stripped = text.strip()
    if not stripped:
        return []

    try:
        sentences = [part.strip() for part in stripped.replace("\u0964", ".").split(".")]
    except Exception:
        return [stripped]

    cleaned = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    return cleaned or [stripped]
