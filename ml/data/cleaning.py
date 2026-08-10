import html
import re
import unicodedata

import emoji as emoji_lib
import ftfy

from ml.data.masking import ALL_MASK_TOKENS, EMOJI_TOKEN_PREFIX, EMOJI_TOKEN_SUFFIX, mask_entities

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
# Negative lookahead protects our own mask/emoji tokens from being read back
# as HTML tags if clean_text() is ever run on already-cleaned text.
_TAG_RE = re.compile(r"<(?!/?(?:URL|USER|EMAIL|PHONE|EMOJI:))[^>]+>")

_QUOTED_REPLY_RE = re.compile(r"\bOn .{0,80}?wrote:.*$", re.IGNORECASE | re.DOTALL)
_SIGNOFF_BLOCK_RE = re.compile(
    r"\n\s*(?:Thanks|Thank you|Regards|Best|Best regards|Sincerely)[,.]?\s*\n[^\n]*$",
    re.IGNORECASE,
)
_SENT_FROM_RE = re.compile(r"\s*Sent from my \w+\s*$", re.IGNORECASE)
_CARET_SIGNOFF_RE = re.compile(r"\s*\^\s*-?\s*[A-Za-z][\w'-]*\s*$")

_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_EMOTICON_MAP = {
    ":-)": "slightly_smiling_face",
    ":)": "slightly_smiling_face",
    ":-(": "frowning_face",
    ":(": "frowning_face",
    ":-D": "grinning_face",
    ":D": "grinning_face",
    ";-)": "winking_face",
    ";)": "winking_face",
    ":'(": "crying_face",
    "<3": "red_heart",
}
_EMOTICON_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_EMOTICON_MAP, key=len, reverse=True))
)

_REPEAT_RE = re.compile(r"(.)\1{3,}")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

_MASK_TOKEN_STRIP_RE = re.compile(
    "|".join(re.escape(t.lower()) for t in ALL_MASK_TOKENS) + r"|<emoji:[^>]*>"
)


def fix_encoding(text: str) -> str:
    return ftfy.fix_text(text)


def unescape_html(text: str) -> str:
    return html.unescape(text)


def strip_html(text: str) -> str:
    text = _BR_RE.sub("\n", text)
    return _TAG_RE.sub("", text)


def strip_signatures(text: str) -> str:
    text = _QUOTED_REPLY_RE.sub("", text)
    text = _SIGNOFF_BLOCK_RE.sub("", text)
    text = _SENT_FROM_RE.sub("", text)
    text = _CARET_SIGNOFF_RE.sub("", text)
    return text.rstrip()


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return text.replace("\xa0", " ")  # NBSP -> regular space


def map_emoji(text: str) -> str:
    text = emoji_lib.demojize(text, delimiters=(EMOJI_TOKEN_PREFIX, EMOJI_TOKEN_SUFFIX))
    return _EMOTICON_RE.sub(
        lambda m: f"{EMOJI_TOKEN_PREFIX}{_EMOTICON_MAP[m.group(0)]}{EMOJI_TOKEN_SUFFIX}", text
    )


def collapse_repeats(text: str) -> str:
    return _REPEAT_RE.sub(lambda m: m.group(1) * 3, text)


def collapse_whitespace(text: str) -> str:
    text = _INLINE_WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str, *, collapse_repeated_chars: bool = True) -> str:
    text = fix_encoding(text)
    text = unescape_html(text)
    text = strip_html(text)
    text = strip_signatures(text)
    text = normalize_unicode(text)
    text = mask_entities(text)
    text = map_emoji(text)
    if collapse_repeated_chars:
        text = collapse_repeats(text)
    return collapse_whitespace(text)


def normalize_for_hash(text: str) -> str:
    """Lowercase + mask/emoji-token-stripped form used only by dedup, never stored."""
    text = text.lower()
    text = _MASK_TOKEN_STRIP_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
