from __future__ import annotations

import mimetypes
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import requests
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from config import settings


@dataclass(frozen=True)
class MediaItem:
    url: str
    content_type: str | None = None


LANGUAGE_HINTS = {
    "mr": "mr",
    "marathi": "mr",
    "kn": "kn",
    "kannada": "kn",
    "ta": "ta",
    "tamil": "ta",
    "hi": "hi",
    "hindi": "hi",
}


def build_twiml_reply(message: str) -> str:
    response = MessagingResponse()
    response.message(message)
    return str(response)


def parse_language_hint(text: str | None) -> str | None:
    if not text:
        return None

    body = text.strip().lower()
    match = re.search(r"\b(lang|language)\s*[:=]?\s*([a-z]{2,12})\b", body)
    if match:
        return LANGUAGE_HINTS.get(match.group(2))

    match = re.search(r"\b(mr|marathi|kn|kannada|ta|tamil|hi|hindi)\b", body)
    if match:
        return LANGUAGE_HINTS.get(match.group(1))

    return LANGUAGE_HINTS.get(body)


def extract_first_media(form_data: Mapping[str, str]) -> MediaItem | None:
    try:
        num_media = int(form_data.get("NumMedia", "0"))
    except (TypeError, ValueError):
        num_media = 0

    if num_media <= 0:
        return None

    url = form_data.get("MediaUrl0")
    if not url:
        return None

    return MediaItem(
        url=url,
        content_type=form_data.get("MediaContentType0"),
    )


def _suffix_from_content_type(content_type: str | None) -> str:
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    return ".ogg"


def download_media(media: MediaItem, destination_dir: Path | None = None) -> Path:
    destination_dir = destination_dir or settings.temp_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    suffix = _suffix_from_content_type(media.content_type)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(destination_dir))
    temp_file.close()

    auth = None
    if settings.twilio_account_sid and settings.twilio_auth_token:
        auth = (settings.twilio_account_sid, settings.twilio_auth_token)

    response = requests.get(media.url, stream=True, timeout=60, auth=auth)
    response.raise_for_status()

    with open(temp_file.name, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                handle.write(chunk)

    return Path(temp_file.name)


def send_whatsapp_message(to_number: str, body: str) -> str:
    if settings.whatsapp_provider != "twilio":
        raise NotImplementedError("Only Twilio is wired for outbound sending in Day 1.")
    if not settings.twilio_account_sid or not settings.twilio_auth_token or not settings.twilio_whatsapp_number:
        raise ValueError("Twilio settings are missing from .env")

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    message = client.messages.create(
        from_=settings.twilio_whatsapp_number,
        to=to_number,
        body=body,
    )
    return message.sid
