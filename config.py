from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


load_dotenv()


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


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = os.getenv("HF_TOKEN")
    asr_model_id: str = os.getenv("ASR_MODEL_ID", "ai4bharat/indicwhisper")
    asr_device: str = os.getenv("ASR_DEVICE", "cuda")
    whatsapp_provider: str = os.getenv("WHATSAPP_PROVIDER", "twilio").lower().strip()
    twilio_account_sid: str | None = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_whatsapp_number: str | None = os.getenv("TWILIO_WHATSAPP_NUMBER")
    public_webhook_url: str | None = os.getenv("PUBLIC_WEBHOOK_URL")
    temp_dir: Path = Path(os.getenv("TEMP_DIR", "tmp"))

    def validate(self) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)

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

        if not self.hf_token:
            raise ValueError("Missing HF_TOKEN in .env")


settings = Settings()
