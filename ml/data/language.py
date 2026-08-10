from dataclasses import dataclass
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

_SUPPORTED_LANGUAGES = [
    Language.ENGLISH,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.PORTUGUESE,
    Language.ITALIAN,
    Language.DUTCH,
    Language.JAPANESE,
    Language.ARABIC,
    Language.HINDI,
    Language.TURKISH,
    Language.INDONESIAN,
]

_ISO_639_1 = {lang: lang.iso_code_639_1.name.lower() for lang in _SUPPORTED_LANGUAGES}


@dataclass(frozen=True)
class LangResult:
    lang: str | None
    confidence: float


@lru_cache(maxsize=1)
def _get_detector() -> object:
    return (
        LanguageDetectorBuilder.from_languages(*_SUPPORTED_LANGUAGES)
        .with_low_accuracy_mode()
        .build()
    )


def detect(text: str) -> LangResult:
    if not text or not text.strip():
        return LangResult(lang=None, confidence=0.0)

    detector = _get_detector()
    confidence_values = detector.compute_language_confidence_values(text)  # type: ignore[attr-defined]
    if not confidence_values:
        return LangResult(lang=None, confidence=0.0)

    top = confidence_values[0]
    return LangResult(lang=_ISO_639_1[top.language], confidence=top.value)
