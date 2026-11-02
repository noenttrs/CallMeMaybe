"""Token vocabulary: real text of every token, plus precomputed lookups.

Qwen uses byte-level BPE (like GPT-2). Each token in vocab.json is written
with printable stand-in characters: every byte 0-255 is mapped to one
unicode character (for example a space becomes ``Ġ`` and a newline ``Ċ``).
To know what text a token really produces we undo that mapping to get the
bytes, then decode the bytes as UTF-8.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

NUMBER_CHARS = frozenset("0123456789.- ")


def bytes_to_unicode() -> dict[int, str]:
    """Build GPT-2's byte -> printable character table.

    Printable bytes keep their own character; the others (control
    characters, space, ...) are shifted to characters from 256 upward.

    Returns:
        A mapping from every byte value (0-255) to one character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: dict[int, str] = {byte: chr(byte) for byte in printable}
    shift = 0
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(256 + shift)
            shift += 1
    return table


def decode_token(token: str, byte_decoder: dict[str, int]) -> str | None:
    """Convert one vocab.json token into the text it stands for.

    Args:
        token: Token as written in vocab.json, e.g. ``"Ġhello"``.
        byte_decoder: Inverse of :func:`bytes_to_unicode`.

    Returns:
        The decoded text (``" hello"``), or None when the token is only a
        piece of a multi-byte UTF-8 character and has no text on its own.
    """
    try:
        raw = bytes(byte_decoder[char] for char in token)
        return raw.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def is_json_string_fragment(text: str) -> bool:
    """Tell whether text can appear as-is inside a JSON string literal.

    Raw quotes, lone backslashes and raw control characters are refused;
    complete escape sequences like ``\\n`` or ``\\\\`` are accepted.

    Args:
        text: Candidate piece of a JSON string body.

    Returns:
        True when ``'"' + text + '"'`` is a valid JSON string.
    """
    try:
        return isinstance(json.loads('"' + text + '"'), str)
    except json.JSONDecodeError:
        return False


class Vocabulary(BaseModel):
    """Decoded vocabulary with the lookups needed by the constraints.

    Attributes:
        texts: ``texts[token_id]`` is the token's text, or None if unusable.
        text_to_id: Reverse lookup from text to token id.
        number_ids: Tokens made only of digits, ``.``, ``-`` and spaces.
        string_body_ids: Tokens that are valid inside a JSON string.
        quote_splits: Tokens holding a closing quote, as
            ``(token_id, text before the quote, text after the quote)``.
        opening_ids: Tokens that open a string and leave it open, like
            ``' "'`` or ``'"hel'``.
        opening_quote_splits: Tokens that open and close a string, as
            ``(token_id, text after the closing quote)``, like ``' ""'``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    texts: list[str | None]
    text_to_id: dict[str, int]
    number_ids: list[int]
    string_body_ids: np.ndarray
    quote_splits: list[tuple[int, str, str]]
    opening_ids: np.ndarray
    opening_quote_splits: list[tuple[int, str]]

    @classmethod
    def from_token_map(cls, token_map: dict[str, int]) -> "Vocabulary":
        """Decode a vocab.json mapping and precompute every lookup.

        This loops over the ~150k tokens once, at start-up, so that no
        generation step ever needs to scan the full vocabulary again.

        Args:
            token_map: Content of vocab.json (token string -> id).

        Returns:
            The ready-to-use vocabulary.
        """
        table = bytes_to_unicode()
        byte_decoder = {char: byte for byte, char in table.items()}
        texts: list[str | None] = [None] * (max(token_map.values()) + 1)
        for token, token_id in token_map.items():
            texts[token_id] = decode_token(token, byte_decoder)

        text_to_id: dict[str, int] = {}
        number_ids: list[int] = []
        body_ids: list[int] = []
        quote_splits: list[tuple[int, str, str]] = []
        opening_ids: list[int] = []
        opening_quote_splits: list[tuple[int, str]] = []
        for token_id, text in enumerate(texts):
            if not text:
                continue
            text_to_id.setdefault(text, token_id)
            if set(text) <= NUMBER_CHARS:
                number_ids.append(token_id)
            if is_json_string_fragment(text):
                body_ids.append(token_id)
            elif '"' in text:
                split = split_at_closing_quote(text)
                if split is not None:
                    quote_splits.append((token_id, *split))
            inner = after_opening_quote(text)
            if inner is not None:
                if is_json_string_fragment(inner):
                    opening_ids.append(token_id)
                else:
                    split = split_at_closing_quote(inner)
                    if split is not None:
                        opening_quote_splits.append((token_id, split[1]))
        return cls(
            texts=texts,
            text_to_id=text_to_id,
            number_ids=number_ids,
            string_body_ids=np.array(body_ids, dtype=np.int64),
            quote_splits=quote_splits,
            opening_ids=np.array(opening_ids, dtype=np.int64),
            opening_quote_splits=opening_quote_splits,
        )

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        """Load vocab.json from disk.

        Args:
            path: Path given by ``get_path_to_vocab_file()``.

        Returns:
            The decoded vocabulary.

        Raises:
            ValueError: If the file cannot be read or is malformed.
        """
        try:
            with path.open("r", encoding="utf-8") as file:
                token_map = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot load vocabulary {path}: {error}"
            ) from None
        if not isinstance(token_map, dict) or not token_map:
            raise ValueError(f"vocabulary {path} is not a token -> id mapping")
        return cls.from_token_map(token_map)

    def text_of(self, token_id: int) -> str:
        """Return a token's text (empty string for unusable tokens).

        Args:
            token_id: Id chosen by the decoder.

        Returns:
            The token's decoded text.
        """
        if 0 <= token_id < len(self.texts):
            return self.texts[token_id] or ""
        return ""

    def ids_for_prefixes(self, text: str) -> list[int]:
        """Find every token whose text is a non-empty prefix of ``text``.

        Only ``len(text)`` dictionary lookups: no scan of the vocabulary.

        Args:
            text: The text that must be continued.

        Returns:
            The matching token ids, shortest prefix first.
        """
        ids = []
        for end in range(1, len(text) + 1):
            token_id = self.text_to_id.get(text[:end])
            if token_id is not None:
                ids.append(token_id)
        return ids


def after_opening_quote(text: str) -> str | None:
    """Return what follows an opening quote at the start of a token.

    Args:
        text: Token text.

    Returns:
        The text after ``"`` when the token starts with ``"`` or ``' "'``,
        otherwise None.
    """
    for opening in ('"', ' "'):
        if text.startswith(opening):
            return text[len(opening):]
    return None


def split_at_closing_quote(text: str) -> tuple[str, str] | None:
    """Split a token at the quote that would close a JSON string.

    Args:
        text: Token text containing at least one ``"``.

    Returns:
        ``(body, rest)`` where ``body`` is a valid string fragment and
        ``rest`` is what follows the closing quote, or None if impossible.
    """
    position = text.find('"')
    while position != -1:
        body = text[:position]
        if is_json_string_fragment(body):
            return body, text[position + 1:]
        position = text.find('"', position + 1)
    return None
