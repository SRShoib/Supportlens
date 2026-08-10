from ml.data.cleaning import (
    clean_text,
    collapse_repeats,
    collapse_whitespace,
    fix_encoding,
    map_emoji,
    normalize_for_hash,
    normalize_unicode,
    strip_html,
    strip_signatures,
    unescape_html,
)


class TestFixEncoding:
    def test_repairs_mojibake(self) -> None:
        assert fix_encoding("Itâ€™s great") == "It's great"

    def test_leaves_clean_text_unchanged(self) -> None:
        assert fix_encoding("plain ascii text") == "plain ascii text"


class TestUnescapeHtml:
    def test_unescapes_entities(self) -> None:
        assert unescape_html("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"

    def test_double_escaped_entity(self) -> None:
        assert unescape_html("&amp;amp;") == "&amp;"


class TestStripHtml:
    def test_br_becomes_newline(self) -> None:
        assert strip_html("Hello<br>World<br/>Foo<br />Bar") == "Hello\nWorld\nFoo\nBar"

    def test_strips_other_tags(self) -> None:
        assert strip_html("<b>bold</b> and <i>italic</i>") == "bold and italic"


class TestStripSignatures:
    def test_caret_agent_initials(self) -> None:
        assert strip_signatures("Please DM us for help ^AB") == "Please DM us for help"

    def test_caret_dash_name(self) -> None:
        assert strip_signatures("Please DM us for help ^ -Amy") == "Please DM us for help"

    def test_sent_from_iphone(self) -> None:
        assert strip_signatures("ok thanks bye Sent from my iPhone") == "ok thanks bye"

    def test_signoff_block(self) -> None:
        assert strip_signatures("Please reply soon\n\nThanks,\nJohn") == "Please reply soon"

    def test_quoted_reply_block(self) -> None:
        text = "My reply here\nOn Jan 5, 2020, Amy wrote:\n> old message"
        assert strip_signatures(text) == "My reply here"

    def test_no_signature_unchanged(self) -> None:
        assert strip_signatures("just a normal message") == "just a normal message"


class TestNormalizeUnicode:
    def test_nfkc_normalizes_compatibility_chars(self) -> None:
        assert normalize_unicode("café") == "café"

    def test_strips_zero_width_chars(self) -> None:
        assert normalize_unicode("a​b⁠c﻿") == "abc"

    def test_nbsp_becomes_space(self) -> None:
        assert normalize_unicode("a\xa0b") == "a b"

    def test_strips_control_chars(self) -> None:
        assert normalize_unicode("a\x00b\x1fc") == "abc"


class TestMapEmoji:
    def test_unicode_emoji_demojized(self) -> None:
        assert map_emoji("great job \U0001f44d") == "great job <EMOJI:thumbs_up>"

    def test_skin_tone_modifier_handled(self) -> None:
        result = map_emoji("nice \U0001f44d\U0001f3fd")
        assert result == "nice <EMOJI:thumbs_up_medium_skin_tone>"

    def test_ascii_emoticons(self) -> None:
        assert map_emoji("happy :-) sad :( love <3") == (
            "happy <EMOJI:slightly_smiling_face> sad <EMOJI:frowning_face> love <EMOJI:red_heart>"
        )

    def test_longer_emoticon_wins_over_prefix(self) -> None:
        assert map_emoji(":-)") == "<EMOJI:slightly_smiling_face>"


class TestCollapseRepeats:
    def test_caps_letter_run_at_three(self) -> None:
        assert collapse_repeats("soooooo") == "sooo"

    def test_caps_punctuation_run(self) -> None:
        assert collapse_repeats("wow!!!!!") == "wow!!!"

    def test_short_run_untouched(self) -> None:
        assert collapse_repeats("aaa") == "aaa"


class TestCollapseWhitespace:
    def test_collapses_inline_whitespace(self) -> None:
        assert collapse_whitespace("a   b\tc") == "a b c"

    def test_caps_blank_lines(self) -> None:
        assert collapse_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_strips_leading_trailing(self) -> None:
        assert collapse_whitespace("  hello  ") == "hello"


class TestCleanTextPipeline:
    def test_full_pipeline_order(self) -> None:
        text = "Hey &amp; @AppleSupport <br> check https://x.co/a!! soooo goooood ^AB"
        result = clean_text(text)
        assert result == "Hey & <USER> \n check <URL>!! sooo goood"

    def test_url_containing_escaped_ampersand(self) -> None:
        result = clean_text("see https://x.co/a?b=1&amp;c=2")
        assert result == "see <URL>"

    def test_emoji_inside_url_not_separately_masked(self) -> None:
        # the URL token swallows everything, so no stray emoji token appears
        result = clean_text("https://x.co/\U0001f600")
        assert result == "<URL>"

    def test_handle_at_string_start(self) -> None:
        assert clean_text("@AppleSupport help me") == "<USER> help me"

    def test_idempotent(self) -> None:
        text = "Hey @user check https://x.co/a soooo goooood!!!"
        once = clean_text(text)
        twice = clean_text(once)
        assert once == twice

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_whitespace_only(self) -> None:
        assert clean_text("   \n\t  ") == ""

    def test_never_leaves_bare_url_or_handle(self) -> None:
        result = clean_text("visit https://example.com or dm @support or email a@b.com")
        assert "http" not in result
        assert "@support" not in result
        assert "@b.com" not in result

    def test_collapse_repeats_disabled(self) -> None:
        result = clean_text("soooooo good", collapse_repeated_chars=False)
        assert result == "soooooo good"


class TestNormalizeForHash:
    def test_strips_mask_tokens(self) -> None:
        assert normalize_for_hash("Check <URL> now <EMOJI:red_heart>") == "check now"

    def test_case_insensitive(self) -> None:
        assert normalize_for_hash("HELLO world") == normalize_for_hash("hello WORLD")

    def test_mask_variant_texts_collide(self) -> None:
        a = clean_text("@amy thanks so much!")
        b = clean_text("@bob thanks so much!")
        assert normalize_for_hash(a) == normalize_for_hash(b)

    def test_whitespace_collapsed(self) -> None:
        assert normalize_for_hash("a   b") == normalize_for_hash("a b")
