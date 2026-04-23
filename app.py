from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from config import settings
from whatsapp import build_twiml_reply, download_media, extract_first_media, parse_language_hint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bharat Voice AI Day 1", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, object]:
    return {
        "status": "running",
        "service": "Bharat Voice AI Day 1",
        "provider": settings.whatsapp_provider,
        "asr_model": settings.asr_model_id,
        "public_webhook_url": settings.public_webhook_url,
    }


@app.on_event("startup")
def log_webhook_configuration() -> None:
    if settings.public_webhook_url:
        logger.info("Twilio WhatsApp Sandbox inbound URL should point to: %s", settings.public_webhook_url)
    else:
        logger.warning(
            "PUBLIC_WEBHOOK_URL is not set. Twilio WhatsApp Sandbox cannot reach a localhost URL."
        )


@app.post("/asr/transcribe")
async def transcribe_upload(
    file: UploadFile = File(...),
    language_code: str | None = Form(default=None),
) -> JSONResponse:
    from asr import get_asr_service

    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(settings.temp_dir)) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        transcript = get_asr_service().transcribe(temp_path, language_code=language_code)
        return JSONResponse(
            {
                "filename": file.filename,
                "language_code": language_code,
                "transcript": transcript,
            }
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "hint": "Check your Hugging Face token permissions and confirm access to the gated ASR model.",
            },
        )
    finally:
        temp_path.unlink(missing_ok=True)


async def _handle_whatsapp_webhook(request: Request, language_code: str | None = None) -> Response:
    from asr import get_asr_service

    form = await request.form()
    payload = {key: value for key, value in form.items()}

    message_text = (payload.get("Body") or "").strip()
    hinted_language = parse_language_hint(message_text)
    media = extract_first_media(payload)
    active_language = language_code or hinted_language

    if media is None:
        reply = (
            "Send me a voice note and I will transcribe it.\n"
            "You can also test language hints by sending: mr, kn, ta, or 'language=mr'."
        )
        return Response(content=build_twiml_reply(reply), media_type="application/xml")

    audio_path = download_media(media)
    try:
        transcript = get_asr_service().transcribe(audio_path, language_code=active_language)
        reply = f"Transcription: {transcript}"
    except RuntimeError as exc:
        reply = f"Model access problem: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("ASR failed")
        reply = (
            "I could not transcribe that audio yet. "
            f"Error: {exc.__class__.__name__}. "
            "If this was an OGG voice note, make sure ffmpeg is installed."
        )
    finally:
        audio_path.unlink(missing_ok=True)

    return Response(content=build_twiml_reply(reply), media_type="application/xml")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request) -> Response:
    return await _handle_whatsapp_webhook(request)


@app.post("/whatsapp")
async def whatsapp_webhook_short(request: Request) -> Response:
    return await _handle_whatsapp_webhook(request)


@app.post("/reply_whatsapp")
async def whatsapp_webhook_reply(request: Request) -> Response:
    return await _handle_whatsapp_webhook(request)


@app.post("/webhook/whatsapp/{language_code}")
async def whatsapp_webhook_language(language_code: str, request: Request) -> Response:
    return await _handle_whatsapp_webhook(request, language_code=language_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
