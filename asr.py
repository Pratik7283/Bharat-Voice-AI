from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import AutoModel

from config import LANGUAGE_NAME_TO_CODE, settings


logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGE_CODES = {"mr", "kn", "ta", "hi", "bn", "gu", "ml", "pa", "or", "as", "ur", "te"}


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is not None:
        return

    try:
        import imageio_ffmpeg
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg is not available on PATH and imageio-ffmpeg is not installed."
        ) from exc

    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = str(ffmpeg_exe.parent)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
    logger.info("Using bundled ffmpeg at %s", ffmpeg_exe)


def _load_audio_waveform(audio_path: str | Path) -> np.ndarray:
    source_path = Path(audio_path)

    try:
        waveform, sample_rate = torchaudio.load(str(source_path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        return waveform.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as torchaudio_exc:
        logger.info("torchaudio could not decode %s: %s", source_path, torchaudio_exc)

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(source_path), dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        waveform = torch.from_numpy(audio.T)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        return waveform.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as soundfile_exc:
        raise RuntimeError(
            "Could not decode the audio file with torchaudio or soundfile. "
            "Try converting the voice note to WAV, or install an ffmpeg-capable audio stack."
        ) from soundfile_exc


class BaseASRService:
    @staticmethod
    def _language_code(language_code: str | None) -> str | None:
        if not language_code:
            return None
        value = language_code.lower().strip()
        mapped = LANGUAGE_NAME_TO_CODE.get(value)
        if mapped:
            return mapped
        if value in SUPPORTED_LANGUAGE_CODES:
            return value
        return None


class FasterWhisperASRService(BaseASRService):
    def __init__(self) -> None:
        self.model_size = settings.local_whisper_model_size
        self.device = "cuda" if settings.asr_device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self.compute_type = settings.local_whisper_compute_type or ("float16" if self.device == "cuda" else "int8")
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "faster-whisper is unavailable on this machine. "
                "If Windows is blocking PyAV, switch LOCAL_WHISPER_BACKEND=openai-whisper."
            ) from exc

        logger.info(
            "Loading Faster-Whisper model size=%s device=%s compute_type=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not load faster-whisper. "
                "Try LOCAL_WHISPER_BACKEND=openai-whisper or use a smaller model."
            ) from exc
        return self._model

    def transcribe(self, audio_path: str | Path, language_code: str | None = None) -> str:
        model = self._load_model()
        language = self._language_code(language_code)

        segments, _info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            beam_size=1,
        )

        parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()


class OpenAIWhisperASRService(BaseASRService):
    def __init__(self) -> None:
        self.model_name = settings.local_whisper_model_size
        self.device = "cuda" if settings.asr_device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import whisper
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not import openai-whisper (module name: whisper). "
                "On Windows this is often a blocked numba/llvmlite binary. "
                "Reinstall the package, then restart the terminal after unblocking native files."
            ) from exc

        logger.info("Loading openai-whisper model=%s device=%s", self.model_name, self.device)
        try:
            self._model = whisper.load_model(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not load the openai-whisper model. "
                "Try a smaller LOCAL_WHISPER_MODEL_SIZE like small or base."
            ) from exc
        return self._model

    def transcribe(self, audio_path: str | Path, language_code: str | None = None) -> str:
        model = self._load_model()
        language = self._language_code(language_code)
        audio = _load_audio_waveform(audio_path)
        result = model.transcribe(
            audio,
            language=language,
            fp16=self.device == "cuda",
        )
        return str(result.get("text", "")).strip()


class IndicConformerASRService(BaseASRService):
    def __init__(self) -> None:
        self.model_id = settings.asr_model_id
        self.device = "cuda" if settings.asr_device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if not self.model_id:
            raise RuntimeError("Set ASR_MODEL_ID to ai4bharat/indic-conformer-600m-multilingual in .env")

        logger.info("Loading IndicConformer model=%s device=%s", self.model_id, self.device)
        try:
            self._model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=settings.hf_token,
            )
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load IndicConformer model {self.model_id}. "
                "Make sure HF_TOKEN is set and the model files are accessible."
            ) from exc

        return self._model

    def transcribe(self, audio_path: str | Path, language_code: str | None = None) -> str:
        model = self._load_model()
        language = self._language_code(language_code) or "hi"

        waveform = _load_audio_waveform(audio_path)
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            transcription = model(wav_tensor, language, "ctc")

        if isinstance(transcription, (list, tuple)):
            transcription = transcription[0]
        return str(transcription).strip()


class OpenAIASRService:
    def __init__(self) -> None:
        self._client: Any | None = None

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not settings.openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The openai package is not installed. Add it to requirements.txt and reinstall dependencies."
            ) from exc

        self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def transcribe(self, audio_path: str | Path, language_code: str | None = None) -> str:
        client = self._load_client()
        transcription_kwargs: dict[str, Any] = {
            "model": settings.openai_transcribe_model,
        }
        if language_code:
            transcription_kwargs["language"] = language_code

        with Path(audio_path).open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(file=audio_file, **transcription_kwargs)

        return str(getattr(transcription, "text", "")).strip()


def _build_local_asr_service() -> Any:
    return IndicConformerASRService()


@lru_cache(maxsize=1)
def get_asr_service() -> Any:
    if settings.asr_provider == "openai":
        return OpenAIASRService()
    return _build_local_asr_service()
