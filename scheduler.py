"""
Daily lesson scheduler — pushes lessons to users via WhatsApp at their chosen time.

Uses APScheduler to run a check every minute. For each user whose local time
matches their ``lesson_time``, it picks the next pending lesson, generates a
phrase via LLM, creates a TTS audio, and sends it over WhatsApp.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from config import settings
from db import session_scope
from lesson_engine import build_lesson_draft, language_display_name
from llm import generate_phrase
from models import LessonProgress, LessonTemplate, User
from tts import synthesize_voice_note
from whatsapp import send_whatsapp_media, send_whatsapp_message

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_local_time(user: User) -> datetime:
    """Return the current time in the user's timezone (or UTC if not set)."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(user.timezone) if user.timezone else timezone.utc
    except (KeyError, Exception):
        tz = timezone.utc
    return datetime.now(tz)


def _public_audio_url(filename: str) -> str:
    """Build a publicly-reachable URL for a synthesized audio file."""
    from urllib.parse import urlsplit

    if settings.public_webhook_url:
        parsed = urlsplit(settings.public_webhook_url)
        return f"{parsed.scheme}://{parsed.netloc}/audio/{filename}"
    return f"http://127.0.0.1:8000/audio/{filename}"


# ---------------------------------------------------------------------------
# Core: push one lesson to one user
# ---------------------------------------------------------------------------

def push_daily_lesson(user: User) -> None:
    """
    Find the user's next pending lesson, generate a phrase via LLM,
    synthesize audio, and send it over WhatsApp.
    """
    if not user.target_lang:
        logger.warning("User %s has no target_lang set, skipping", user.phone)
        return

    native_lang = user.native_lang or "hi"
    target_lang = user.target_lang

    with session_scope() as session:
        # Re-attach user to this session
        user = session.merge(user)

        # ── 1. Find next pending lesson ──────────────────────────────────
        # Get IDs of lessons already completed or sent today
        done_lesson_ids = (
            session.scalars(
                select(LessonProgress.lesson_id).where(
                    LessonProgress.user_id == user.id,
                    LessonProgress.target_lang == target_lang,
                    LessonProgress.status.in_(["sent", "completed"]),
                )
            ).all()
        )

        # Pick the first template not yet done
        next_template = session.scalar(
            select(LessonTemplate)
            .where(LessonTemplate.id.notin_(done_lesson_ids) if done_lesson_ids else True)
            .order_by(LessonTemplate.difficulty, LessonTemplate.id)
            .limit(1)
        )

        if next_template is None:
            logger.info(
                "User %s has completed all lessons for %s! 🎉",
                user.phone, target_lang,
            )
            try:
                send_whatsapp_message(
                    user.phone,
                    f"🎉 Congratulations! You've completed all {language_display_name(target_lang)} lessons!",
                )
            except Exception:
                logger.exception("Failed to send completion message to %s", user.phone)
            return

        # ── 2. Generate phrase via LLM ───────────────────────────────────
        draft = build_lesson_draft(next_template, target_lang, native_lang)
        try:
            generated_phrase = generate_phrase(draft.prompt, target_lang)
        except Exception:
            logger.exception("LLM generation failed for user %s", user.phone)
            generated_phrase = draft.rendered_template_text

        # ── 3. Build the WhatsApp message ────────────────────────────────
        target_name = language_display_name(target_lang)
        message_body = (
            f"📚 *Daily {target_name} Lesson*\n\n"
            f"*Category:* {next_template.category.title()}\n"
            f"*Intent:* {draft.rendered_template_text}\n\n"
            f"*{target_name} phrase:*\n{generated_phrase}\n\n"
            f"🎙️ Try saying this phrase and send me a voice note! "
            f"I'll score your pronunciation."
        )

        # ── 4. TTS audio ─────────────────────────────────────────────────
        audio_sent = False
        try:
            audio_path = synthesize_voice_note(generated_phrase, target_lang)
            audio_url = _public_audio_url(audio_path.name)
            send_whatsapp_media(user.phone, audio_url, body=message_body)
            audio_sent = True
            logger.info("✅ Lesson sent with audio to %s", user.phone)
        except Exception:
            logger.exception("TTS/media failed for %s, sending text-only", user.phone)

        if not audio_sent:
            try:
                send_whatsapp_message(user.phone, message_body)
                logger.info("✅ Lesson sent (text-only) to %s", user.phone)
            except Exception:
                logger.exception("❌ Failed to send lesson to %s", user.phone)
                return

        # ── 5. Record progress ───────────────────────────────────────────
        progress = LessonProgress(
            user_id=user.id,
            lesson_id=next_template.id,
            target_lang=target_lang,
            status="sent",
            last_sent_at=datetime.now(timezone.utc),
            attempt_count=0,
        )
        session.add(progress)
        logger.info(
            "Recorded lesson progress: user=%s template=%s lang=%s",
            user.phone, next_template.slug, target_lang,
        )


# ---------------------------------------------------------------------------
# Scheduler tick — runs every minute
# ---------------------------------------------------------------------------

def _check_and_push_lessons() -> None:
    """
    Called every minute by APScheduler.
    Finds users whose lesson_time matches the current minute in their timezone.
    """
    if not settings.database_url:
        return

    try:
        with session_scope() as session:
            # Get all users who have a lesson_time set
            users = session.scalars(
                select(User).where(
                    User.lesson_time.isnot(None),
                    User.target_lang.isnot(None),
                )
            ).all()

        for user in users:
            local_now = _user_local_time(user)
            current_hhmm = local_now.strftime("%H:%M")

            if current_hhmm == user.lesson_time:
                logger.info(
                    "⏰ Lesson time match for %s (time=%s tz=%s)",
                    user.phone, user.lesson_time, user.timezone,
                )
                try:
                    push_daily_lesson(user)
                except Exception:
                    logger.exception("Failed to push lesson to %s", user.phone)

    except Exception:
        logger.exception("Error in scheduler tick")


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def start_scheduler() -> None:
    """Start the background scheduler (called once at app startup)."""
    global _scheduler

    if not settings.database_url:
        logger.info("DATABASE_URL not set — lesson scheduler not started")
        return

    if _scheduler is not None:
        logger.warning("Scheduler already running")
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _check_and_push_lessons,
        "interval",
        minutes=1,
        id="daily_lesson_push",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("📅 Lesson scheduler started (checking every 1 minute)")


def stop_scheduler() -> None:
    """Gracefully stop the scheduler (called at app shutdown)."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("📅 Lesson scheduler stopped")
