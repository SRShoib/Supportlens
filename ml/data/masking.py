import re
from enum import StrEnum

# Order ids, amounts, and dates are deliberately NOT masked here — they are
# the M4 NER targets, and masking them would destroy the training signal.


class MaskToken(StrEnum):
    URL = "<URL>"
    USER = "<USER>"
    EMAIL = "<EMAIL>"
    PHONE = "<PHONE>"


EMOJI_TOKEN_PREFIX = "<EMOJI:"
EMOJI_TOKEN_SUFFIX = ">"

ALL_MASK_TOKENS: frozenset[str] = frozenset(token.value for token in MaskToken)

_TRAILING_PUNCT = ".,!?;:)]}\"'"

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
_HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{1,15}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")


def _mask_urls(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        url = match.group(0)
        trimmed = url.rstrip(_TRAILING_PUNCT)
        trailing = url[len(trimmed) :]
        return f"{MaskToken.URL}{trailing}"

    return _URL_RE.sub(repl, text)


def mask_entities(text: str) -> str:
    """Mask URLs, emails, @handles, and phone numbers, in that order.

    URLs run first so an @handle or email embedded in a URL's path/query is
    swallowed by the URL token rather than double-masked. Emails run before
    handles so ``a@b.com`` doesn't become ``a<USER>.com``.
    """
    text = _mask_urls(text)
    text = _EMAIL_RE.sub(MaskToken.EMAIL, text)
    text = _HANDLE_RE.sub(MaskToken.USER, text)
    text = _PHONE_RE.sub(MaskToken.PHONE, text)
    return text


def additional_special_tokens() -> list[str]:
    """The exact special-token list M3 passes to a Hugging Face tokenizer."""
    return [token.value for token in MaskToken]
