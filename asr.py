from __future__ import annotations

import logging
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from transformers import AutoModel

from config import SUPPORTED_LANGUAGE_CODES, settings


logger = logging.getLogger(__name__)

TARGET_SR = 16_000


class ASRService:
    def __init__(self) -> None:
        self.model_id = settings.asr_model_id
        self.device = 0 if settings.asr_device.lower() == "cuda" and torch.cuda.is_available() else -1
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        logger.info("Loading ASR model %s", self.model_id)
        try:
            self._model = AutoModel.from_pretrained(
                self.model_id,
                token=settings.hf_token,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Hugging Face cannot load the ASR model. "
                "Open the model page, accept access if required, and create a token that has access to public gated repositories."
            ) from exc
        return self._model

    @staticmethod
    def _language_name(language_code: str | None) -> str | None:
        if not language_code:
            return None
        return SUPPORTED_LANGUAGE_CODES.get(language_code.lower().strip())

    @staticmethod
    def _load_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
        """
        Load audio as mono 16 kHz float32.

        librosa can read many common formats; if the file is not supported on a
        given machine, install ffmpeg and/or pydub for WhatsApp OGG voice notes.
        """
        source_path = Path(audio_path)
        try:
            audio, _sample_rate = librosa.load(str(source_path), sr=TARGET_SR, mono=True)
            return audio.astype(np.float32), TARGET_SR
        except Exception:
            wav_path = ASRService._convert_to_wav(source_path)
            try:
                audio, _sample_rate = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
                return audio.astype(np.float32), TARGET_SR
            finally:
                wav_path.unlink(missing_ok=True)

    @staticmethod
    def _convert_to_wav(audio_path: Path) -> Path:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Could not decode this audio file. Install ffmpeg on Windows or upload a .wav file instead."
            )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()

        try:
            segment = AudioSegment.from_file(str(audio_path))
            segment = segment.set_channels(1).set_frame_rate(TARGET_SR)
            segment.export(temp_file.name, format="wav")
        except CouldntDecodeError as exc:
            Path(temp_file.name).unlink(missing_ok=True)
            raise RuntimeError(
                "Unsupported audio format. Please convert the file to .wav or install ffmpeg."
            ) from exc

        return Path(temp_file.name)

    def transcribe(self, audio_path: str | Path, language_code: str | None = None) -> str:
        model = self._load_model()
        audio, sample_rate = self._load_audio(audio_path)
        wav_tensor = torch.from_numpy(audio).unsqueeze(0)

        language_name = self._language_name(language_code) or "hi"
        result = model(wav_tensor, language_name, "ctc")
        return str(result).strip()


@lru_cache(maxsize=1)
def get_asr_service() -> ASRService:
    return ASRService()
