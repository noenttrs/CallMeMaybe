"""Shared fixtures: a tiny fake vocabulary written in vocab.json format."""

import pytest

from src.vocab import Vocabulary, bytes_to_unicode

# Texts of the fake tokens; the token id is the position in this list.
FAKE_TOKENS = [
    "fn", "_add", "_a", "_numbers", "_greet", '"', '",', '"}}', "}}", "}",
    ",", ", ", " ", " -", "-", ".", "0", "1", "2", "3", "5", "12", "true",
    " true", "false", " false", "tr", " \"", '"', ' ""', ' "hi', "hi",
    "hello", " world", "\\\\", "\\n", "\\", "a\"b", "\n", "'", "é",
]


def to_vocab_entry(text: str) -> str:
    """Encode text the way vocab.json stores it (byte-level BPE).

    Args:
        text: Real token text.

    Returns:
        The vocab.json spelling, e.g. ``"Ġhello"`` for ``" hello"``.
    """
    table = bytes_to_unicode()
    return "".join(table[byte] for byte in text.encode("utf-8"))


@pytest.fixture(scope="session")
def vocab() -> Vocabulary:
    """Build a Vocabulary from FAKE_TOKENS.

    Returns:
        The fake vocabulary (duplicates keep their first id).
    """
    token_map: dict[str, int] = {}
    for token_id, text in enumerate(FAKE_TOKENS):
        token_map.setdefault(to_vocab_entry(text), token_id)
    # A lone UTF-8 continuation byte: not decodable on its own.
    token_map[bytes_to_unicode()[0xA9]] = len(FAKE_TOKENS)
    return Vocabulary.from_token_map(token_map)
