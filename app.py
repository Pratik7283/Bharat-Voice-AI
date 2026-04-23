from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from config import settings
from whatsapp import (
    build_twiml_reply,
    download_media,
    extract_first_media,
    parse_language_hint,
    send_whatsapp_message,
)


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


@app.post("/")
async def root_post(request: Request) -> Response:
    # Fallback for cases where the sandbox webhook is configured to the bare domain.
    return await _handle_whatsapp_webhook(request)


@app.on_event("startup")
def log_webhook_configuration() -> None:
    settings.validate()
    if settings.public_webhook_url:
        logger.info("Twilio WhatsApp Sandbox inbound URL should point to: %s", settings.public_webhook_url)
    else:
        logger.warning(
            "PUBLIC_WEBHOOK_URL is not set. Twilio WhatsApp Sandbox cannot reach a localhost URL."
        )


def process_voice_note_audio(media, sender_number: str, active_language: str | None) -> None:
    from asr import get_asr_service

    logger.info("Processing voice note from %s", sender_number)
    try:
        audio_path = download_media(media)
        logger.info(
            "Downloaded voice note media for %s to %s (content_type=%s)",
            sender_number,
            audio_path,
            getattr(media, "content_type", None),
        )
        transcript = get_asr_service().transcribe(audio_path, language_code=active_language)
        reply = f"Transcription: {transcript}"
        logger.info("Transcription complete for %s", sender_number)
    except RuntimeError as exc:
        reply = f"Model access problem: {exc}"
        logger.exception("Model access issue while processing voice note")
    except Exception as exc:  # noqa: BLE001
        logger.exception("ASR failed")
        reply = (
            "I could not transcribe that audio yet. "
            f"Error: {exc.__class__.__name__}. "
            "If this was an OGG voice note, make sure ffmpeg is installed."
        )
    finally:
        if "audio_path" in locals():
            audio_path.unlink(missing_ok=True)

    try:
        message_sid = send_whatsapp_message(sender_number, reply)
        logger.info("Sent async WhatsApp reply to %s sid=%s", sender_number, message_sid)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send async WhatsApp reply")


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
    logger.info(
        "WhatsApp webhook request arrived path=%s content_type=%s",
        request.url.path,
        request.headers.get("content-type"),
    )
    form = await request.form()
    payload = {key: value for key, value in form.items()}
    sender_number = (payload.get("From") or "").strip()
    num_media = payload.get("NumMedia")
    media_url0 = payload.get("MediaUrl0")
    media_type0 = payload.get("MediaContentType0")
    logger.info(
        "Inbound WhatsApp webhook from=%s num_media=%s media_url0=%s media_type0=%s body=%s",
        sender_number,
        num_media,
        bool(media_url0),
        media_type0,
        (payload.get("Body") or "").strip(),
    )

    message_text = (payload.get("Body") or "").strip()
    hinted_language = parse_language_hint(message_text)
    media = extract_first_media(payload)
    active_language = language_code or hinted_language
    if media is not None:
        logger.info(
            "Media detected in inbound WhatsApp webhook from=%s url_present=%s content_type=%s",
            sender_number,
            bool(getattr(media, "url", None)),
            getattr(media, "content_type", None),
        )

    if media is None:
        reply = (
            "Send me a voice note and I will transcribe it.\n"
            "You can also test language hints by sending: mr, kn, ta, or 'language=mr'."
        )
        return Response(content=build_twiml_reply(reply), media_type="application/xml")

    if not sender_number:
        logger.warning("Received voice note without sender number; cannot send async reply.")
        return Response(
            content=build_twiml_reply(
                "I received your voice note, but I could not identify the sender."
            ),
            media_type="application/xml",
        )

    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_voice_note_audio, media, sender_number, active_language)

    return Response(
        content=build_twiml_reply(
            "I got your voice note and I am transcribing it now. "
            "I will send the result here in a moment."
        ),
        media_type="application/xml",
        background=background_tasks,
    )


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
