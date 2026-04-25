from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import LANGUAGE_NAME_TO_CODE
from models import LessonTemplate


@dataclass(frozen=True)
class LessonTemplateDefinition:
    slug: str
    category: str
    template_text: str
    difficulty: int = 1
    default_target_lang: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LessonDraft:
    slug: str
    category: str
    target_language: str
    native_language: str
    template_text: str
    rendered_template_text: str
    prompt: str
    variables: dict[str, Any]
    generated_phrase: str
    llm_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LESSON_TEMPLATE_DEFINITIONS: list[LessonTemplateDefinition] = [
    LessonTemplateDefinition(
        slug="pickup_greet_driver",
        category="pickup",
        template_text="Greet the driver and tell them you are waiting at {pickup_location}.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="pickup_confirm_car",
        category="pickup",
        template_text="Ask if this is the car going to {destination}.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="pickup_share_location",
        category="pickup",
        template_text="Tell the driver your pickup point is near {landmark}.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="pickup_delay",
        category="pickup",
        template_text="Say you will be ready in {minutes} minutes.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="pickup_waiting",
        category="pickup",
        template_text="Tell the driver you are waiting outside the {place}.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="pickup_wrong_spot",
        category="pickup",
        template_text="Ask the driver to come to {alternate_location} instead.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="fare_ask_price",
        category="fare",
        template_text="Ask how much the fare will be to {destination}.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="fare_confirm_meter",
        category="fare",
        template_text="Ask whether the driver will use the meter.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="fare_receipt",
        category="fare",
        template_text="Ask for a receipt after the ride.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="fare_cash_only",
        category="fare",
        template_text="Say you only have cash and ask if that is okay.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="fare_discount",
        category="fare",
        template_text="Politely ask if there is any discount available.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="fare_too_high",
        category="fare",
        template_text="Say the fare seems too high and ask why.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="directions_go_straight",
        category="directions",
        template_text="Ask the driver to go straight for {blocks} blocks.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="directions_turn_left",
        category="directions",
        template_text="Tell them to take the next left turn.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="directions_turn_right",
        category="directions",
        template_text="Tell them to take the next right turn.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="directions_near_landmark",
        category="directions",
        template_text="Say the destination is near {landmark}.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="directions_slow_down",
        category="directions",
        template_text="Ask the driver to slow down near {place}.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="directions_drop_here",
        category="directions",
        template_text="Tell the driver to drop you here.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="polite_thanks",
        category="polite",
        template_text="Say thank you politely.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="polite_please_wait",
        category="polite",
        template_text="Politely ask the person to wait for a moment.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="polite_sorry",
        category="polite",
        template_text="Apologize politely for the delay.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="polite_help",
        category="polite",
        template_text="Ask for help politely.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="polite_repeat",
        category="polite",
        template_text="Politely ask the other person to repeat that.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="polite_speak_slowly",
        category="polite",
        template_text="Politely ask them to speak a little slowly.",
        difficulty=1,
    ),
    LessonTemplateDefinition(
        slug="complaint_aircon",
        category="complaint",
        template_text="Complain that the air conditioning is not working.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="complaint_overcharged",
        category="complaint",
        template_text="Say you were overcharged and want to discuss it.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="complaint_rude_driver",
        category="complaint",
        template_text="Say the driver was rude and you want to complain.",
        difficulty=3,
    ),
    LessonTemplateDefinition(
        slug="complaint_wrong_route",
        category="complaint",
        template_text="Say the driver took the wrong route.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="emergency_stop",
        category="emergency",
        template_text="Tell the driver to stop the vehicle immediately.",
        difficulty=2,
    ),
    LessonTemplateDefinition(
        slug="emergency_hospital",
        category="emergency",
        template_text="Say you need to go to the hospital urgently.",
        difficulty=2,
    ),
]


def _normalize_language(language_code: str) -> str:
    value = language_code.lower().strip()
    return LANGUAGE_NAME_TO_CODE.get(value, value)


def language_display_name(language_code: str) -> str:
    return {
        "hi": "Hindi",
        "mr": "Marathi",
        "kn": "Kannada",
        "ta": "Tamil",
    }.get(language_code, language_code)


def _safe_format(template_text: str, variables: dict[str, Any]) -> str:
    class _SafeDict(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template_text.format_map(_SafeDict(variables))


def default_lesson_templates() -> list[LessonTemplateDefinition]:
    return list(LESSON_TEMPLATE_DEFINITIONS)


def seed_lesson_templates(session: Session) -> dict[str, int]:
    created = 0
    updated = 0

    for definition in LESSON_TEMPLATE_DEFINITIONS:
        existing = session.scalar(select(LessonTemplate).where(LessonTemplate.slug == definition.slug))
        if existing is None:
            session.add(
                LessonTemplate(
                    slug=definition.slug,
                    category=definition.category,
                    template_text=definition.template_text,
                    difficulty=definition.difficulty,
                    default_target_lang=definition.default_target_lang,
                )
            )
            created += 1
            continue

        changed = False
        if existing.category != definition.category:
            existing.category = definition.category
            changed = True
        if existing.template_text != definition.template_text:
            existing.template_text = definition.template_text
            changed = True
        if existing.difficulty != definition.difficulty:
            existing.difficulty = definition.difficulty
            changed = True
        if existing.default_target_lang != definition.default_target_lang:
            existing.default_target_lang = definition.default_target_lang
            changed = True
        if changed:
            updated += 1

    return {"created": created, "updated": updated, "total": len(LESSON_TEMPLATE_DEFINITIONS)}


def list_lesson_templates(session: Session) -> list[dict[str, Any]]:
    rows = session.scalars(select(LessonTemplate).order_by(LessonTemplate.category, LessonTemplate.slug)).all()
    if rows:
        return [
            {
                "slug": row.slug,
                "category": row.category,
                "template_text": row.template_text,
                "difficulty": row.difficulty,
                "default_target_lang": row.default_target_lang,
            }
            for row in rows
        ]

    return [asdict(definition) for definition in LESSON_TEMPLATE_DEFINITIONS]


def build_lesson_prompt(
    template_text: str,
    target_language: str,
    native_language: str = "hi",
    variables: dict[str, Any] | None = None,
) -> str:
    normalized_target = _normalize_language(target_language)
    normalized_native = _normalize_language(native_language)
    variables = variables or {}
    filled_template = _safe_format(template_text, variables)
    target_name = language_display_name(normalized_target)
    native_name = language_display_name(normalized_native)

    return (
        "You are creating a short lesson phrase for a language learning app.\n"
        f"Generate the final phrase in {target_name}.\n"
        f"The learner's native language is {native_name}.\n"
        "Keep it natural, practical, and short.\n"
        "Do not add explanations, transliterations, or markdown.\n"
        "Intent:\n"
        f"{filled_template}\n"
        f"Variables JSON: {json.dumps(variables, ensure_ascii=False)}"
    )


def build_lesson_draft(
    template: LessonTemplate,
    target_language: str,
    native_language: str = "hi",
    variables: dict[str, Any] | None = None,
    use_llm: bool = False,
) -> LessonDraft:
    variables = variables or {}
    rendered = _safe_format(template.template_text, variables)
    prompt = build_lesson_prompt(template.template_text, target_language, native_language, variables)

    generated_phrase = rendered
    llm_output = None

    if use_llm:
        try:
            from llm import generate_phrase

            generated_phrase = generate_phrase(prompt, _normalize_language(target_language))
            llm_output = generated_phrase
        except Exception:
            import logging
            logging.getLogger(__name__).exception("LLM generation failed, using template text")

    return LessonDraft(
        slug=template.slug,
        category=template.category,
        target_language=_normalize_language(target_language),
        native_language=_normalize_language(native_language),
        template_text=template.template_text,
        rendered_template_text=rendered,
        prompt=prompt,
        variables=variables,
        generated_phrase=generated_phrase,
        llm_output=llm_output,
    )


def get_template_by_slug(session: Session, slug: str) -> LessonTemplate | None:
    return session.scalar(select(LessonTemplate).where(LessonTemplate.slug == slug))
