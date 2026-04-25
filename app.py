from __future__ import annotations

import json
import os
import logging
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from config import settings
from db import init_db
from lesson_engine import (
    build_lesson_draft,
    get_template_by_slug,
    list_lesson_templates,
    seed_lesson_templates,
)
from pronunciation import score_pronunciation
from whatsapp import (
    build_twiml_reply,
    download_media,
    extract_first_media,
    parse_language_hint,
    send_whatsapp_media,
    send_whatsapp_message,
)
from translation import translate_hindi
from tts import synthesize_voice_note
from scheduler import start_scheduler, stop_scheduler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bharat Voice AI", version="0.3.0")
LAST_LANGUAGE_BY_SENDER: dict[str, str] = {}
LAST_TRANSCRIPT_BY_SENDER: dict[str, str] = {}
PENDING_TARGET_LANGUAGE_BY_SENDER: dict[str, str] = {}
ONBOARDING_STATE: dict[str, str] = {}  # sender -> "awaiting_target_lang" | "awaiting_time"


def _resolved_asr_model_name() -> str:
    if settings.asr_provider == "openai":
        return settings.openai_transcribe_model
    return settings.asr_model_id


def _public_audio_url(filename: str) -> str:
    if settings.public_webhook_url:
        parsed = urlsplit(settings.public_webhook_url)
        return f"{parsed.scheme}://{parsed.netloc}/audio/{filename}"
    return f"http://127.0.0.1:8000/audio/{filename}"


def _twiml_response(message: str, background: BackgroundTasks | None = None) -> Response:
    return Response(content=build_twiml_reply(message), media_type="text/xml", background=background)


async def audio_file(request: Request) -> FileResponse:
    filename = request.path_params["filename"]
    safe_name = Path(filename).name
    audio_path = settings.temp_dir / safe_name
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    suffix = audio_path.suffix.lower()
    if suffix == ".ogg":
        media_type = "audio/ogg"
    elif suffix == ".mp3":
        media_type = "audio/mpeg"
    else:
        media_type = "audio/wav"
    return FileResponse(audio_path, media_type=media_type, filename=audio_path.name)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def root(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "running",
        "service": "Bharat Voice AI",
        "asr_provider": settings.asr_provider,
        "tts_provider": settings.tts_provider,
        "provider": settings.whatsapp_provider,
        "asr_model": _resolved_asr_model_name(),
        "public_webhook_url": settings.public_webhook_url,
    })


async def root_post(request: Request) -> Response:
    # Fallback for cases where the sandbox webhook is configured to the bare domain.
    return await _handle_whatsapp_webhook(request)

def log_webhook_configuration() -> None:
    try:
        settings.validate()
    except ValueError as exc:
        logger.error("\n" + "=" * 60)
        logger.error("STARTUP VALIDATION FAILED: %s", exc)
        logger.error("=" * 60)
        return

    logger.info("=" * 60)


def init_database_on_startup() -> None:
    if not settings.database_url:
        logger.info("DATABASE_URL is not set; skipping Postgres schema initialization")
        return

    try:
        init_db()
        logger.info("Postgres schema initialized")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not initialize Postgres schema: %s", exc)


def seed_lesson_templates_on_startup() -> None:
    if not settings.database_url:
        logger.info("DATABASE_URL is not set; skipping lesson template seeding")
        return

    from db import session_scope

    try:
        with session_scope() as session:
            result = seed_lesson_templates(session)
        logger.info(
            "Lesson templates seeded: created=%s updated=%s total=%s",
            result["created"],
            result["updated"],
            result["total"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not seed lesson templates: %s", exc)
    logger.info("Bharat Voice AI — startup complete")
    logger.info("  ASR provider: %s", settings.asr_provider)
    logger.info("  ASR model   : %s", _resolved_asr_model_name())
    logger.info("  ASR device : %s", settings.asr_device)
    logger.info("  TTS provider: %s", settings.tts_provider)
    logger.info("  Provider   : %s", settings.whatsapp_provider)
    logger.info("  Twilio SID : %s", (settings.twilio_account_sid or "")[:8] + "...")
    logger.info("  From number: %s", settings.twilio_whatsapp_number)
    if settings.public_webhook_url:
        logger.info("  Webhook URL: %s", settings.public_webhook_url)
        logger.info("  >>> Set the above URL in Twilio Sandbox 'When a message comes in' <<<")
    else:
        logger.warning("PUBLIC_WEBHOOK_URL is not set in .env — update it with your current ngrok URL!")
    logger.info("=" * 60)


def start_scheduler_on_startup() -> None:
    start_scheduler()


def stop_scheduler_on_shutdown() -> None:
    stop_scheduler()


def process_voice_note_audio(media, sender_number: str, active_language: str | None) -> None:
    """Background job: download audio, transcribe it, then send the reply."""
    from asr import get_asr_service

    logger.info("[BG] Starting voice-note processing for %s (lang=%s)", sender_number, active_language)
    downloaded_audio_path: Path | None = None
    generated_audio_path: Path | None = None
    reply: str

    try:
        # ── 1. Download the audio from Twilio ──────────────────────────────
        logger.info("[BG] Downloading media url=%s content_type=%s", media.url, media.content_type)
        downloaded_audio_path = download_media(media)
        logger.info("[BG] Download complete → %s (%d bytes)", downloaded_audio_path, downloaded_audio_path.stat().st_size)

        # ── 2. Transcribe ──────────────────────────────────────────────────
        logger.info("[BG] Starting transcription...")
        transcript = get_asr_service().transcribe(downloaded_audio_path, language_code=active_language)
        logger.info("[BG] Transcription done: %r", transcript)
        LAST_TRANSCRIPT_BY_SENDER[sender_number] = transcript
        target_language = PENDING_TARGET_LANGUAGE_BY_SENDER.pop(sender_number, None)
        if target_language in {"mr", "kn", "ta"}:
            try:
                translated_text = translate_hindi(transcript, target_language)
            except Exception:
                logger.exception("[BG] Translation failed, falling back to the original transcript")
                translated_text = transcript

            generated_audio_path = synthesize_voice_note(translated_text, target_language)
            audio_url = _public_audio_url(generated_audio_path.name)
            pretty_language = {
                "mr": "Marathi",
                "kn": "Kannada",
                "ta": "Tamil",
            }.get(target_language, target_language)
            logger.info(
                "[BG] Sending translated audio to %s via %s (file=%s)",
                sender_number,
                audio_url,
                generated_audio_path,
            )
            message_sid = send_whatsapp_media(
                sender_number,
                audio_url,
                body=f"✅ {pretty_language} voice note is ready.",
            )
            logger.info("[BG] ✅ Translated audio sent to %s SID=%s", sender_number, message_sid)
            return

        reply = f"✅ Transcription:\n{transcript}"

    except RuntimeError as exc:
        logger.exception("[BG] RuntimeError during ASR (model access / ffmpeg issue)")
        reply = (
            f"⚠️ Model access problem: {exc}\n"
            "If you are using OpenAI ASR, check OPENAI_API_KEY. "
            "If you are using local ASR, check ASR_MODEL_ID and HF_TOKEN for IndicConformer."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[BG] Unexpected error during voice-note processing")
        reply = (
            f"❌ Could not transcribe your voice note.\n"
            f"Error: {exc.__class__.__name__}: {exc}\n"
            "Make sure your audio decoder stack can read WhatsApp OGG/Opus files."
        )
    finally:
        if downloaded_audio_path and downloaded_audio_path.exists():
            downloaded_audio_path.unlink(missing_ok=True)
            logger.info("[BG] Downloaded input temp file cleaned up")
        if generated_audio_path:
            logger.info("[BG] Preserving synthesized audio for Twilio fetch: %s", generated_audio_path)

    # ── 3. Send reply back via Twilio REST API ─────────────────────────────
    try:
        logger.info("[BG] Sending WhatsApp reply to %s ...", sender_number)
        message_sid = send_whatsapp_message(sender_number, reply)
        logger.info("[BG] ✅ Reply sent to %s  SID=%s", sender_number, message_sid)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[BG] ❌ Failed to send WhatsApp reply to %s: %s", sender_number, exc)


def process_translation_and_tts(transcript: str, sender_number: str, target_language: str) -> None:
    logger.info("[BG] Starting translation/TTS for %s -> %s", sender_number, target_language)
    try:
        translated_text = translate_hindi(transcript, target_language)
        logger.info("[BG] Translation done: %r", translated_text)
        pretty_language = {
            "mr": "Marathi",
            "kn": "Kannada",
            "ta": "Tamil",
        }.get(target_language, target_language)
        try:
            audio_path = synthesize_voice_note(translated_text, target_language)
            audio_url = _public_audio_url(audio_path.name)
            body = f"✅ {pretty_language} translation:\n{translated_text}"
            logger.info("[BG] Sending translated voice note to %s via %s", sender_number, audio_url)
            message_sid = send_whatsapp_media(sender_number, audio_url, body=body)
            logger.info("[BG] ✅ Voice note sent to %s SID=%s", sender_number, message_sid)
        except Exception as tts_exc:  # noqa: BLE001
            logger.exception("[BG] TTS failed, sending text-only translation for %s", sender_number)
            send_whatsapp_message(
                sender_number,
                (
                    f"✅ {pretty_language} translation:\n{translated_text}\n\n"
                    "I could not create the audio reply on this machine, so I sent the text version instead."
                ),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[BG] ❌ Translation/TTS failed for %s: %s", sender_number, exc)
        try:
            send_whatsapp_message(
                sender_number,
                f"❌ Could not create the translated voice note.\nError: {exc}",
            )
        except Exception:
            logger.exception("[BG] ❌ Could not send failure message to %s", sender_number)


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
                "hint": (
                    "If ASR_PROVIDER=openai, check OPENAI_API_KEY. "
                    "If ASR_PROVIDER=local, check ASR_MODEL_ID and HF_TOKEN for IndicConformer."
                ),
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
    pending_transcript = LAST_TRANSCRIPT_BY_SENDER.get(sender_number)
    active_language = language_code or remembered_language
    normalized_text = message_text.lower().strip()

    # ── No media: handle commands & onboarding ─────────────────────────────
    if media is None:
        # ── Onboarding: "learn" command ────────────────────────────────
        if normalized_text in {"learn", "start lesson", "start lessons", "lesson"}:
            ONBOARDING_STATE[sender_number] = "awaiting_target_lang"
            reply = (
                "📚 *Let's set up your daily lessons!*\n\n"
                "Which language do you want to learn?\n"
                "Reply with: *Marathi*, *Kannada*, *Tamil*, *Hindi*, *Bengali*, *Gujarati*, *Telugu*, etc."
            )
            return _twiml_response(reply)

        # ── Onboarding: awaiting target language ──────────────────────
        if ONBOARDING_STATE.get(sender_number) == "awaiting_target_lang":
            from config import LANGUAGE_NAME_TO_CODE
            lang_code = LANGUAGE_NAME_TO_CODE.get(normalized_text)
            if not lang_code:
                # Try 2-letter code directly
                from config import SUPPORTED_LANGUAGE_CODES
                if normalized_text in SUPPORTED_LANGUAGE_CODES:
                    lang_code = normalized_text
            if lang_code:
                _save_user_target_lang(sender_number, lang_code)
                ONBOARDING_STATE[sender_number] = "awaiting_time"
                from lesson_engine import language_display_name
                reply = (
                    f"✅ Great! You'll learn *{language_display_name(lang_code)}*.\n\n"
                    "⏰ What time should I send your daily lesson?\n"
                    "Reply with a time like: *09:00* or *18:30* (24-hour format)"
                )
            else:
                reply = (
                    "❓ I didn't recognize that language.\n"
                    "Try: *Marathi*, *Kannada*, *Tamil*, *Hindi*, *Bengali*, *Gujarati*, *Telugu*"
                )
            return _twiml_response(reply)

        # ── Onboarding: awaiting lesson time ──────────────────────────
        if ONBOARDING_STATE.get(sender_number) == "awaiting_time":
            import re as _re
            time_match = _re.match(r"^(\d{1,2})[:.]?(\d{2})$", normalized_text.strip())
            if time_match:
                hh = int(time_match.group(1))
                mm = int(time_match.group(2))
                if 0 <= hh <= 23 and 0 <= mm <= 59:
                    lesson_time = f"{hh:02d}:{mm:02d}"
                    _save_user_lesson_time(sender_number, lesson_time)
                    ONBOARDING_STATE.pop(sender_number, None)
                    reply = (
                        f"✅ All set! You'll receive a daily lesson at *{lesson_time}* IST.\n\n"
                        "📚 Your first lesson will arrive at the scheduled time.\n"
                        "Send *lesson status* to check your progress anytime."
                    )
                    return _twiml_response(reply)
            reply = (
                "❓ Please send a valid time in 24-hour format.\n"
                "Examples: *09:00*, *14:30*, *18:00*"
            )
            return _twiml_response(reply)

        # ── Check lesson status ───────────────────────────────────────
        if normalized_text in {"lesson status", "status", "progress", "my lessons"}:
            status_text = _get_lesson_status(sender_number)
            return _twiml_response(status_text)

        # ── Send a lesson now (for testing) ───────────────────────────
        if normalized_text in {"send lesson", "next lesson", "lesson now"}:
            background_tasks = BackgroundTasks()
            background_tasks.add_task(_push_lesson_for_sender, sender_number)
            reply = "📚 Generating your next lesson… I'll send it in a moment!"
            return _twiml_response(reply, background=background_tasks)

        # ── Existing translation logic ────────────────────────────────
        explicit_translate_request = any(
            keyword in normalized_text
            for keyword in {"translate", "translation", "repeat", "last", "send audio", "voice note"}
        )
        if (
            sender_number
            and hinted_language
            and pending_transcript
            and hinted_language in {"mr", "kn", "ta"}
            and explicit_translate_request
        ):
            pretty_language = {
                "mr": "Marathi",
                "kn": "Kannada",
                "ta": "Tamil",
            }.get(hinted_language, hinted_language)
            logger.info(
                "Scheduling translation/TTS for %s -> %s",
                sender_number,
                hinted_language,
            )
            background_tasks = BackgroundTasks()
            background_tasks.add_task(
                process_translation_and_tts,
                pending_transcript,
                sender_number,
                hinted_language,
            )
            reply = (
                f"Got it. I'm translating to {pretty_language} and will send the voice note shortly."
            )
            return _twiml_response(reply, background=background_tasks)

        if sender_number and hinted_language in {"mr", "kn", "ta"}:
            PENDING_TARGET_LANGUAGE_BY_SENDER[sender_number] = hinted_language
            LAST_LANGUAGE_BY_SENDER.pop(sender_number, None)
            pretty_language = {
                "mr": "Marathi",
                "kn": "Kannada",
                "ta": "Tamil",
            }.get(hinted_language, hinted_language)
            logger.info("Stored target language for %s => %s", sender_number, hinted_language)
            reply = (
                f"Language set to {pretty_language}.\n"
                f"Send your voice note and I will send a {pretty_language} voice note."
            )
            return _twiml_response(reply)

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
            return _twiml_response(reply)

        logger.info("No media in payload — sending usage hint")
        reply = (
            "🎙️ Send me a voice note and I will transcribe it!\n"
            "Language hints: send 'mr', 'hi', 'kn', 'ta' before your voice note.\n"
            "To translate the last transcript, send 'translate mr' or 'translate kn'.\n\n"
            "📚 Send *learn* to start daily language lessons!"
        )
        return _twiml_response(reply)

        logger.info("No media in payload — sending usage hint")
        reply = (
            "🎙️ Send me a voice note and I will transcribe it!\n"
            "Language hints: send 'mr', 'hi', 'kn', 'ta' before your voice note.\n"
            "To translate the last transcript, send 'translate mr' or 'translate kn'."
        )
        return _twiml_response(reply)

    logger.info("Media found: url_present=%s  content_type=%s", bool(media.url), media.content_type)

    # ── No sender: can't reply ────────────────────────────────────────────
    if not sender_number:
        logger.warning("Voice note received but 'From' field is empty — cannot send reply")
        return _twiml_response("I received your voice note but could not identify you.")

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
    return _twiml_response(ack, background=background_tasks)


async def whatsapp_webhook(request: Request) -> Response:
    return await _handle_whatsapp_webhook(request)


async def whatsapp_webhook_short(request: Request) -> Response:
    return await _handle_whatsapp_webhook(request)


async def whatsapp_webhook_reply(request: Request) -> Response:
    return await _handle_whatsapp_webhook(request)


async def whatsapp_webhook_language(request: Request) -> Response:
    language_code = request.path_params["language_code"]
    return await _handle_whatsapp_webhook(request, language_code=language_code)


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


async def debug_translate(
    text: str = Form(...),
    target_language: str = Form(...),
) -> JSONResponse:
    translated = translate_hindi(text, target_language)
    return JSONResponse(
        {
            "source_language": "hi",
            "target_language": target_language,
            "input_text": text,
            "translated_text": translated,
        }
    )


async def debug_tts(
    text: str = Form(...),
    target_language: str = Form(...),
) -> JSONResponse:
    audio_path = synthesize_voice_note(text, target_language)
    return JSONResponse(
        {
            "target_language": target_language,
            "input_text": text,
            "audio_file": audio_path.name,
            "audio_url": _public_audio_url(audio_path.name),
            "audio_format": audio_path.suffix.lstrip("."),
        }
    )


async def debug_translate_tts(
    text: str = Form(...),
    target_language: str = Form(...),
) -> JSONResponse:
    translated_text = translate_hindi(text, target_language)
    audio_path = synthesize_voice_note(translated_text, target_language)
    return JSONResponse(
        {
            "source_language": "hi",
            "target_language": target_language,
            "input_text": text,
            "translated_text": translated_text,
            "audio_file": audio_path.name,
            "audio_url": _public_audio_url(audio_path.name),
            "audio_format": audio_path.suffix.lstrip("."),
        }
    )


async def debug_voice_translate_tts(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language_code: str = Form(default="hi"),
) -> JSONResponse:
    from asr import get_asr_service

    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(settings.temp_dir)) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        transcript = get_asr_service().transcribe(temp_path, language_code=source_language_code)
        translated_text = translate_hindi(transcript, target_language)
        audio_path = synthesize_voice_note(translated_text, target_language)
        return JSONResponse(
            {
                "source_language_code": source_language_code,
                "target_language": target_language,
                "transcript": transcript,
                "translated_text": translated_text,
                "audio_file": audio_path.name,
                "audio_url": _public_audio_url(audio_path.name),
                "audio_format": audio_path.suffix.lstrip("."),
            }
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "hint": (
                    "If ASR_PROVIDER=openai, check OPENAI_API_KEY. "
                    "If ASR_PROVIDER=local, check ASR_MODEL_ID and HF_TOKEN for IndicConformer."
                ),
            },
        )
    finally:
        temp_path.unlink(missing_ok=True)


async def debug_pronunciation_score(
    file: UploadFile = File(...),
    target_phrase: str = Form(...),
    language_code: str = Form(...),
) -> JSONResponse:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(settings.temp_dir)) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        result = score_pronunciation(temp_path, target_phrase, language_code)
        return JSONResponse(result.to_dict())
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "hint": (
                    "If ASR_PROVIDER=openai, check OPENAI_API_KEY. "
                    "If ASR_PROVIDER=local, check ASR_MODEL_ID and HF_TOKEN for IndicConformer."
                ),
            },
        )
    finally:
        temp_path.unlink(missing_ok=True)


async def debug_lesson_seed() -> JSONResponse:
    from db import session_scope

    if not settings.database_url:
        return JSONResponse(
            status_code=400,
            content={"error": "DATABASE_URL is not configured"},
        )

    with session_scope() as session:
        result = seed_lesson_templates(session)
    return JSONResponse(result)


async def debug_lesson_templates() -> JSONResponse:
    from db import session_scope

    if not settings.database_url:
        from lesson_engine import default_lesson_templates

        return JSONResponse(
            {
                "source": "defaults",
                "templates": [template.to_dict() for template in default_lesson_templates()],
            }
        )

    with session_scope() as session:
        templates = list_lesson_templates(session)
    return JSONResponse({"source": "database", "templates": templates})


async def debug_lesson_generate(
    template_slug: str = Form(...),
    target_language: str = Form(...),
    native_language: str = Form(default="hi"),
    variables_json: str = Form(default="{}"),
) -> JSONResponse:
    from db import session_scope

    try:
        variables = json.loads(variables_json) if variables_json.strip() else {}
        if not isinstance(variables, dict):
            raise ValueError("variables_json must decode to an object")
    except json.JSONDecodeError as exc:
        return JSONResponse(status_code=400, content={"error": f"Invalid variables_json: {exc}"})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    if settings.database_url:
        with session_scope() as session:
            template = get_template_by_slug(session, template_slug)
            if template is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Lesson template not found: {template_slug}"},
                )
            draft = build_lesson_draft(template, target_language, native_language, variables)
    else:
        from lesson_engine import LESSON_TEMPLATE_DEFINITIONS

        template_def = next((item for item in LESSON_TEMPLATE_DEFINITIONS if item.slug == template_slug), None)
        if template_def is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Lesson template not found: {template_slug}"},
            )
        from models import LessonTemplate

        template = LessonTemplate(
            slug=template_def.slug,
            category=template_def.category,
            template_text=template_def.template_text,
            difficulty=template_def.difficulty,
            default_target_lang=template_def.default_target_lang,
        )
        draft = build_lesson_draft(template, target_language, native_language, variables)

    return JSONResponse(draft.to_dict())


app.add_event_handler("startup", log_webhook_configuration)
app.add_event_handler("startup", init_database_on_startup)
app.add_event_handler("startup", seed_lesson_templates_on_startup)
app.add_event_handler("startup", start_scheduler_on_startup)
app.add_event_handler("shutdown", stop_scheduler_on_shutdown)
app.add_api_route("/audio/{filename}", audio_file, methods=["GET"])
app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/", root, methods=["GET"])
app.add_api_route("/", root_post, methods=["POST"])
app.add_api_route("/webhook/whatsapp", whatsapp_webhook, methods=["POST"])
app.add_api_route("/whatsapp", whatsapp_webhook_short, methods=["POST"])
app.add_api_route("/reply_whatsapp", whatsapp_webhook_reply, methods=["POST"])
app.add_api_route("/webhook/whatsapp/{language_code}", whatsapp_webhook_language, methods=["POST"])
app.add_api_route("/debug", debug_webhook, methods=["POST"])
app.add_api_route("/asr/transcribe", transcribe_upload, methods=["POST"])
app.add_api_route("/debug/translate", debug_translate, methods=["POST"])
app.add_api_route("/debug/tts", debug_tts, methods=["POST"])
app.add_api_route("/debug/translate-tts", debug_translate_tts, methods=["POST"])
app.add_api_route("/debug/voice-translate-tts", debug_voice_translate_tts, methods=["POST"])
app.add_api_route("/debug/pronunciation-score", debug_pronunciation_score, methods=["POST"])
app.add_api_route("/debug/lesson-seed", debug_lesson_seed, methods=["POST"])
app.add_api_route("/debug/lesson-templates", debug_lesson_templates, methods=["GET"])
app.add_api_route("/debug/lesson-generate", debug_lesson_generate, methods=["POST"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
