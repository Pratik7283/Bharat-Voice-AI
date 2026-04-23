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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bharat Voice AI Day 1", version="0.1.0")
LAST_LANGUAGE_BY_SENDER: dict[str, str] = {}


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
    try:
        settings.validate()
    except ValueError as exc:
        logger.error("\n" + "=" * 60)
        logger.error("STARTUP VALIDATION FAILED: %s", exc)
        logger.error("=" * 60)
        return

    logger.info("=" * 60)
    logger.info("Bharat Voice AI — startup complete")
    logger.info("  ASR model  : %s", settings.asr_model_id)
    logger.info("  ASR device : %s", settings.asr_device)
    logger.info("  Provider   : %s", settings.whatsapp_provider)
    logger.info("  Twilio SID : %s", (settings.twilio_account_sid or "")[:8] + "...")
    logger.info("  From number: %s", settings.twilio_whatsapp_number)
    if settings.public_webhook_url:
        logger.info("  Webhook URL: %s", settings.public_webhook_url)
        logger.info("  >>> Set the above URL in Twilio Sandbox 'When a message comes in' <<<")
    else:
        logger.warning("PUBLIC_WEBHOOK_URL is not set in .env — update it with your current ngrok URL!")
    logger.info("=" * 60)


def process_voice_note_audio(media, sender_number: str, active_language: str | None) -> None:
    """Background job: download audio, transcribe it, then send the reply."""
    from asr import get_asr_service

    logger.info("[BG] Starting voice-note processing for %s (lang=%s)", sender_number, active_language)
    audio_path: Path | None = None
    reply: str

    try:
        # ── 1. Download the audio from Twilio ──────────────────────────────
        logger.info("[BG] Downloading media url=%s content_type=%s", media.url, media.content_type)
        audio_path = download_media(media)
        logger.info("[BG] Download complete → %s (%d bytes)", audio_path, audio_path.stat().st_size)

        # ── 2. Transcribe ──────────────────────────────────────────────────
        logger.info("[BG] Starting transcription...")
        transcript = get_asr_service().transcribe(audio_path, language_code=active_language)
        logger.info("[BG] Transcription done: %r", transcript)
        reply = f"✅ Transcription:\n{transcript}"

    except RuntimeError as exc:
        logger.exception("[BG] RuntimeError during ASR (model access / ffmpeg issue)")
        reply = (
            f"⚠️ Model access problem: {exc}\n"
            "Check your HF_TOKEN and confirm you have access to the model."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[BG] Unexpected error during voice-note processing")
        reply = (
            f"❌ Could not transcribe your voice note.\n"
            f"Error: {exc.__class__.__name__}: {exc}\n"
            "Make sure ffmpeg is installed for OGG files."
        )
    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink(missing_ok=True)
            logger.info("[BG] Temp file cleaned up")

    # ── 3. Send reply back via Twilio REST API ─────────────────────────────
    try:
        logger.info("[BG] Sending WhatsApp reply to %s ...", sender_number)
        message_sid = send_whatsapp_message(sender_number, reply)
        logger.info("[BG] ✅ Reply sent to %s  SID=%s", sender_number, message_sid)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[BG] ❌ Failed to send WhatsApp reply to %s: %s", sender_number, exc)


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
        "─── Incoming webhook  path=%s  content-type=%s ───",
        request.url.path,
        request.headers.get("content-type"),
    )

    form = await request.form()
    payload = {key: value for key, value in form.items()}

    # Log the full payload so we can see exactly what Twilio sends
    logger.info("Twilio payload keys: %s", list(payload.keys()))

    sender_number = (payload.get("From") or "").strip()
    num_media = payload.get("NumMedia", "0")
    media_url0 = payload.get("MediaUrl0", "")
    media_type0 = payload.get("MediaContentType0", "")
    body_text = (payload.get("Body") or "").strip()

    logger.info(
        "From=%s  NumMedia=%s  MediaType=%s  Body=%r",
        sender_number, num_media, media_type0, body_text,
    )

    message_text = body_text
    hinted_language = parse_language_hint(message_text)
    media = extract_first_media(payload)
    remembered_language = LAST_LANGUAGE_BY_SENDER.get(sender_number)
    active_language = language_code or hinted_language or remembered_language

    # ── No media: prompt user ──────────────────────────────────────────────
    if media is None:
        if sender_number and hinted_language:
            LAST_LANGUAGE_BY_SENDER[sender_number] = hinted_language
            pretty_language = {
                "mr": "Marathi",
                "kn": "Kannada",
                "ta": "Tamil",
                "hi": "Hindi",
            }.get(hinted_language, hinted_language)
            logger.info("Stored language preference for %s => %s", sender_number, hinted_language)
            reply = (
                f"Language set to {pretty_language}.\n"
                "Now send your voice note and I will transcribe it."
            )
            return Response(content=build_twiml_reply(reply), media_type="application/xml")

        if sender_number and remembered_language:
            pretty_language = {
                "mr": "Marathi",
                "kn": "Kannada",
                "ta": "Tamil",
                "hi": "Hindi",
            }.get(remembered_language, remembered_language)
            reply = (
                f"Current language is {pretty_language}.\n"
                "Send your voice note and I will transcribe it."
            )
            return Response(content=build_twiml_reply(reply), media_type="application/xml")

        logger.info("No media in payload — sending usage hint")
        reply = (
            "🎙️ Send me a voice note and I will transcribe it!\n"
            "Language hints: send 'mr', 'hi', 'kn', 'ta' before your voice note."
        )
        return Response(content=build_twiml_reply(reply), media_type="application/xml")

    logger.info("Media found: url_present=%s  content_type=%s", bool(media.url), media.content_type)

    # ── No sender: can't reply ────────────────────────────────────────────
    if not sender_number:
        logger.warning("Voice note received but 'From' field is empty — cannot send reply")
        return Response(
            content=build_twiml_reply("I received your voice note but could not identify you."),
            media_type="application/xml",
        )

    # ── Fire background transcription task so Twilio gets an immediate reply ───
    logger.info("Scheduling background transcription for %s", sender_number)
    if sender_number and active_language:
        LAST_LANGUAGE_BY_SENDER[sender_number] = active_language
        logger.info("Using language preference for %s => %s", sender_number, active_language)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_voice_note_audio, media, sender_number, active_language)

    ack = (
        "🎙️ Got your voice note! Transcribing now…\n"
        "I'll send the result in a moment."
    )
    return Response(content=build_twiml_reply(ack), media_type="application/xml", background=background_tasks)


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


@app.post("/debug")
async def debug_webhook(request: Request) -> JSONResponse:
    """Dump the raw Twilio POST payload — use this to verify Twilio is reaching your server."""
    form = await request.form()
    payload = {key: value for key, value in form.items()}
    logger.info("[DEBUG] Raw Twilio payload: %s", payload)
    return JSONResponse({
        "received": True,
        "payload_keys": list(payload.keys()),
        "From": payload.get("From"),
        "NumMedia": payload.get("NumMedia"),
        "MediaContentType0": payload.get("MediaContentType0"),
        "MediaUrl0": payload.get("MediaUrl0", "")[:60] + "..." if payload.get("MediaUrl0") else None,
        "Body": payload.get("Body"),
    })

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
