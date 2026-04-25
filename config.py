from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import re

from dotenv import load_dotenv


load_dotenv()


def _clean_url(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", "", value)
    return cleaned or None


DEFAULT_INDICF5_PROMPT_TEXT = (
    "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, "
    "ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)


SUPPORTED_LANGUAGE_CODES: Final[dict[str, str]] = {
    "mr": "marathi",
    "kn": "kannada",
    "ta": "tamil",
    "hi": "hindi",
    "bn": "bengali",
    "gu": "gujarati",
    "ml": "malayalam",
    "pa": "punjabi",
    "or": "odia",
    "as": "assamese",
    "ur": "urdu",
    "te": "telugu",
}

LANGUAGE_NAME_TO_CODE: Final[dict[str, str]] = {
    "marathi": "mr",
    "kannada": "kn",
    "tamil": "ta",
    "hindi": "hi",
    "bengali": "bn",
    "gujarati": "gu",
    "malayalam": "ml",
    "punjabi": "pa",
    "odia": "or",
    "oriya": "or",
    "assamese": "as",
    "urdu": "ur",
    "telugu": "te",
}


@dataclass(frozen=True)
class Settings:
    asr_provider: str = os.getenv("ASR_PROVIDER", "local").lower().strip()
    hf_token: str | None = os.getenv("HF_TOKEN")
    asr_model_id: str = os.getenv("ASR_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual")
    asr_device: str = os.getenv("ASR_DEVICE", "cuda")
    local_whisper_model_size: str = os.getenv("LOCAL_WHISPER_MODEL_SIZE", "small").lower().strip()
    local_whisper_compute_type: str = os.getenv("LOCAL_WHISPER_COMPUTE_TYPE", "int8").lower().strip()
    local_whisper_backend: str = os.getenv("LOCAL_WHISPER_BACKEND", "auto").lower().strip()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_transcribe_model: str = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
    tts_provider: str = os.getenv("TTS_PROVIDER", "indictts").lower().strip()
    tts_device: str = os.getenv("TTS_DEVICE", "cpu").lower().strip()
    indicf5_model_id: str = os.getenv("INDICF5_MODEL_ID", "ai4bharat/IndicF5")
    indicf5_prompt_audio_path: str | None = os.getenv("INDICF5_PROMPT_AUDIO_PATH")
    indicf5_prompt_text: str = os.getenv("INDICF5_PROMPT_TEXT", DEFAULT_INDICF5_PROMPT_TEXT)
    piper_binary_path: str | None = os.getenv("PIPER_BINARY_PATH")
    piper_model_path: str | None = os.getenv("PIPER_MODEL_PATH")
    whatsapp_provider: str = os.getenv("WHATSAPP_PROVIDER", "twilio").lower().strip()
    twilio_account_sid: str | None = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_whatsapp_number: str | None = os.getenv("TWILIO_WHATSAPP_NUMBER")
    public_webhook_url: str | None = _clean_url(os.getenv("PUBLIC_WEBHOOK_URL"))
    translation_ckpt_dir: str | None = os.getenv("INDICTRANS2_CKPT_DIR")
    translation_model_type: str = os.getenv("INDICTRANS2_MODEL_TYPE", "ctranslate2").lower().strip()
    translation_device: str = os.getenv("INDICTRANS2_DEVICE", "cpu").lower().strip()
    database_url: str | None = os.getenv("DATABASE_URL")
    temp_dir: Path = Path(os.getenv("TEMP_DIR", "tmp"))

    def validate(self) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if self.asr_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("Missing OPENAI_API_KEY in .env")

        if self.whatsapp_provider == "twilio":
            missing = [
                name
                for name, value in {
                    "TWILIO_ACCOUNT_SID": self.twilio_account_sid,
                    "TWILIO_AUTH_TOKEN": self.twilio_auth_token,
                    "TWILIO_WHATSAPP_NUMBER": self.twilio_whatsapp_number,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(
                    "Missing Twilio settings in .env: " + ", ".join(missing)
                )

        if self.public_webhook_url and " " in self.public_webhook_url:
            raise ValueError("PUBLIC_WEBHOOK_URL contains spaces. Remove the spaces in .env.")

        if self.tts_provider not in {"indictts", "piper", "auto"}:
            raise ValueError("Unsupported TTS_PROVIDER. Use indictts, piper, or auto.")

        if self.database_url and not self.database_url.startswith("postgresql"):
            raise ValueError("DATABASE_URL should point to a Postgres database.")


settings = Settings()
