"""Tests for vocabulary decoding and precomputed lookups."""

from src.vocab import (
    Vocabulary,
    bytes_to_unicode,
    decode_token,
    is_json_string_fragment,
    split_at_closing_quote,
)


def test_bytes_to_unicode_is_a_bijection() -> None:
    """Every byte gets its own character."""
    table = bytes_to_unicode()
    assert len(table) == 256
    assert len(set(table.values())) == 256
    assert table[ord("a")] == "a"
    assert table[ord(" ")] == "Ġ"
    assert table[ord("\n")] == "Ċ"


def test_decode_token() -> None:
    """Special characters are turned back into real text."""
    decoder = {char: byte for byte, char in bytes_to_unicode().items()}
    assert decode_token("Ġhello", decoder) == " hello"
    assert decode_token("ĊĊ", decoder) == "\n\n"
    assert decode_token("Ã©", decoder) == "é"
    assert decode_token("Ã", decoder) is None  # half of a UTF-8 char


def test_json_fragments() -> None:
    """Only text that is valid inside a JSON string is a fragment."""
    assert is_json_string_fragment("hello world")
    assert is_json_string_fragment("\\n")
    assert is_json_string_fragment('\\"')
    assert is_json_string_fragment("")
    assert not is_json_string_fragment('"')
    assert not is_json_string_fragment("\\")
    assert not is_json_string_fragment("\n")


def test_split_at_closing_quote() -> None:
    """The first unescaped quote closes the string."""
    assert split_at_closing_quote('",') == ("", ",")
    assert split_at_closing_quote('ab"}}') == ("ab", "}}")
    assert split_at_closing_quote('\\""') == ('\\"', "")
    assert split_at_closing_quote("\\") is None


def test_vocabulary_lookups(vocab: Vocabulary) -> None:
    """Precomputed categories contain the right tokens."""
    assert vocab.text_of(vocab.text_to_id[" world"]) == " world"
    assert vocab.text_of(len(vocab.texts)) == ""  # out of range
    assert vocab.texts[-1] is None  # undecodable byte
    numbers = {vocab.text_of(i) for i in vocab.number_ids}
    assert {"0", "12", "-", " -", ".", " "} <= numbers
    assert "a" not in numbers
    body = {vocab.text_of(int(i)) for i in vocab.string_body_ids}
    assert "hello" in body and "\\\\" in body
    assert '"' not in body and "\n" not in body and "\\" not in body
    opening = {vocab.text_of(int(i)) for i in vocab.opening_ids}
    assert {' "', '"', ' "hi'} <= opening


def test_ids_for_prefixes(vocab: Vocabulary) -> None:
    """Every token that is a prefix of the text is found."""
    texts = [vocab.text_of(i) for i in vocab.ids_for_prefixes('_add"')]
    assert texts == ["_a", "_add"]
