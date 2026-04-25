from __future__ import annotations

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from config import LANGUAGE_NAME_TO_CODE, settings

logger = logging.getLogger(__name__)

SUPPORTED_TARGET_LANGUAGES = {"mr", "kn", "ta"}


class TTSUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndicTTSPaths:
    model_path: str
    config_path: str
    vocoder_path: str
    vocoder_config_path: str


@dataclass(frozen=True)
class PiperPaths:
    model_path: str
    binary_path: str | None = None


def _normalize_language_code(language_code: str) -> str:
    value = language_code.lower().strip()
    return LANGUAGE_NAME_TO_CODE.get(value, value)


def _language_label(language_code: str) -> str:
    return {
        "mr": "Marathi",
        "kn": "Kannada",
        "ta": "Tamil",
    }.get(language_code, language_code)


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _indictts_paths(language_code: str) -> IndicTTSPaths | None:
    code = _normalize_language_code(language_code).upper()
    model_path = _env_value(f"INDICTTS_MODEL_PATH_{code}", "INDICTTS_MODEL_PATH")
    config_path = _env_value(f"INDICTTS_CONFIG_PATH_{code}", "INDICTTS_CONFIG_PATH")
    vocoder_path = _env_value(f"INDICTTS_VOCODER_PATH_{code}", "INDICTTS_VOCODER_PATH")
    vocoder_config_path = _env_value(
        f"INDICTTS_VOCODER_CONFIG_PATH_{code}",
        "INDICTTS_VOCODER_CONFIG_PATH",
    )

    if not all([model_path, config_path, vocoder_path, vocoder_config_path]):
        return None

    return IndicTTSPaths(
        model_path=model_path,
        config_path=config_path,
        vocoder_path=vocoder_path,
        vocoder_config_path=vocoder_config_path,
    )


def _piper_paths(language_code: str) -> PiperPaths | None:
    code = _normalize_language_code(language_code).upper()
    model_path = _env_value(f"PIPER_MODEL_PATH_{code}", "PIPER_MODEL_PATH")
    binary_path = _env_value("PIPER_BINARY_PATH", f"PIPER_BINARY_PATH_{code}")
    if not model_path:
        return None
    return PiperPaths(model_path=model_path, binary_path=binary_path)


class IndicTTSTTS:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._device = torch.device(
            "cuda" if settings.tts_device == "cuda" and torch.cuda.is_available() else "cpu"
        )

    def _load_backend(self, language_code: str) -> Any:
        language_code = _normalize_language_code(language_code)
        if language_code in self._cache:
            return self._cache[language_code]

        paths = _indictts_paths(language_code)
        if paths is None:
            raise TTSUnavailableError(
                "IndicTTS is not configured for this language. "
                "Set INDICTTS_MODEL_PATH_<LANG>, INDICTTS_CONFIG_PATH_<LANG>, "
                "INDICTTS_VOCODER_PATH_<LANG>, and INDICTTS_VOCODER_CONFIG_PATH_<LANG> "
                "for mr, kn, and ta."
            )

        try:
            from TTS.api import TTS as CoquiTTS
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailableError(
                "The Coqui TTS package is not installed. Add TTS to requirements.txt and reinstall."
            ) from exc

        logger.info(
            "Loading IndicTTS for %s on %s",
            _language_label(language_code),
            self._device,
        )
        try:
            backend = CoquiTTS(
                model_path=paths.model_path,
                config_path=paths.config_path,
                vocoder_path=paths.vocoder_path,
                vocoder_config_path=paths.vocoder_config_path,
                progress_bar=False,
                gpu=self._device.type == "cuda",
            )
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailableError(
                f"Could not load IndicTTS for {_language_label(language_code)}: {exc}"
            ) from exc

        self._cache[language_code] = backend
        return backend

    def synthesize(self, text: str, language_code: str) -> Path:
        language_code = _normalize_language_code(language_code)
        backend = self._load_backend(language_code)

        stem = f"tts_{uuid.uuid4().hex}"
        wav_path = settings.temp_dir / f"{stem}.wav"
        settings.temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            backend.tts_to_file(text=text, file_path=str(wav_path))
        except TypeError:
            backend.tts_to_file(text, file_path=str(wav_path))
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailableError(
                f"IndicTTS synthesis failed for {_language_label(language_code)}: {exc}"
            ) from exc

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise TTSUnavailableError("IndicTTS did not create an audio file")

        return wav_path


class PiperTTS:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _load_backend(self, language_code: str) -> Any:
        language_code = _normalize_language_code(language_code)
        if language_code in self._cache:
            return self._cache[language_code]

        paths = _piper_paths(language_code)
        if paths is None:
            raise TTSUnavailableError(
                "Piper is not configured for this language. "
                "Set PIPER_MODEL_PATH_<LANG> or PIPER_MODEL_PATH."
            )

        self._cache[language_code] = paths
        return paths

    def synthesize(self, text: str, language_code: str) -> Path:
        language_code = _normalize_language_code(language_code)
        paths = self._load_backend(language_code)

        stem = f"tts_{uuid.uuid4().hex}"
        wav_path = settings.temp_dir / f"{stem}.wav"
        settings.temp_dir.mkdir(parents=True, exist_ok=True)

        model_path = Path(paths.model_path)
        if not model_path.exists():
            raise TTSUnavailableError(f"Piper model not found: {model_path}")

        binary_path = paths.binary_path
        if binary_path:
            binary = Path(binary_path)
            if not binary.exists():
                raise TTSUnavailableError(f"Piper binary not found: {binary}")
            cmd = [
                str(binary),
                "--model",
                str(model_path),
                "--output_file",
                str(wav_path),
            ]
            logger.info("Running Piper binary for %s: %s", _language_label(language_code), " ".join(cmd))
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        else:
            try:
                from piper import PiperVoice
            except Exception as exc:  # noqa: BLE001
                raise TTSUnavailableError(
                    "The piper-tts package is not installed. Add piper-tts to requirements.txt and reinstall."
                ) from exc

            logger.info("Running Piper python backend for %s", _language_label(language_code))
            voice = PiperVoice.load(str(model_path))
            import wave

            with wave.open(str(wav_path), "wb") as wav_file:
                voice.synthesize(text, wav_file)

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise TTSUnavailableError("Piper did not create an audio file")

        return wav_path


class GoogleTTSTTS:
    def synthesize(self, text: str, language_code: str) -> Path:
        language_code = _normalize_language_code(language_code)

        stem = f"tts_{uuid.uuid4().hex}"
        mp3_path = settings.temp_dir / f"{stem}.mp3"
        settings.temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            from gtts import gTTS
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailableError(
                "The gTTS package is not installed. Add gTTS to requirements.txt and reinstall."
            ) from exc

        try:
            tts = gTTS(text=text, lang=language_code, slow=False)
            tts.save(str(mp3_path))
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailableError(f"gTTS synthesis failed for {_language_label(language_code)}: {exc}") from exc

        if not mp3_path.exists() or mp3_path.stat().st_size == 0:
            raise TTSUnavailableError("gTTS did not create an audio file")

        return mp3_path


@lru_cache(maxsize=1)
def _get_indictts() -> IndicTTSTTS:
    return IndicTTSTTS()


@lru_cache(maxsize=1)
def _get_piper() -> PiperTTS:
    return PiperTTS()


@lru_cache(maxsize=1)
def _get_gtts() -> GoogleTTSTTS:
    return GoogleTTSTTS()


def synthesize_voice_note(text: str, target_language: str) -> Path:
    language_code = _normalize_language_code(target_language)
    if language_code not in SUPPORTED_TARGET_LANGUAGES:
        raise TTSUnavailableError(f"Unsupported target language for TTS: {target_language}")

    provider = settings.tts_provider
    last_error: Exception | None = None

    if provider in {"indictts", "auto"}:
        try:
            return _get_indictts().synthesize(text, language_code)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.exception("IndicTTS failed for %s", _language_label(language_code))

    if provider in {"piper", "auto", "indictts"}:
        try:
            return _get_piper().synthesize(text, language_code)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.exception("Piper failed for %s", _language_label(language_code))

    try:
        return _get_gtts().synthesize(text, language_code)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        logger.exception("gTTS failed for %s", _language_label(language_code))

    if last_error is None:
        raise TTSUnavailableError("No TTS backend is configured")
    raise TTSUnavailableError(str(last_error))
