"""Constraints: which tokens may come next while generating one value.

Every constraint works on ``buffer``, the text generated so far for the
current value, and answers two questions:

* ``allowed_ids(buffer)``: the token ids the model may choose now;
* ``advance(buffer, token_text)``: the new buffer, and whether the value
  is finished.

``follow`` is the fixed JSON text written right after the value (for
example ``", "`` or ``"}}"``). A value ends when the model picks a token
that starts this text: that is how the model decides where a number ends
or where a string closes. Allowing tokens such as ``",`` (quote + comma)
matters: it is what the model naturally wants to write.
"""

import json
import re

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.models import ArgumentValue
from src.vocab import (
    Vocabulary,
    is_json_string_fragment,
    split_at_closing_quote,
)

# Number values may start with one space: the tokenizer writes ": -2" as
# ":", " -", "2", so the model expects to produce that space itself.
# Leading zeros ("05") are accepted: the text is parsed with int()/float().
NUMBER_RE = re.compile(r" ?-?[0-9]+(\.[0-9]+)?")
NUMBER_PREFIX_RE = re.compile(r" ?-?[0-9]*(\.[0-9]*)?")


class Constraint(BaseModel):
    """Base class for all value constraints.

    Attributes:
        vocab: Decoded vocabulary.
        follow: Fixed text that comes right after the value.
        max_tokens: Maximum number of tokens generated for this value.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab: Vocabulary
    follow: str = ""
    max_tokens: int = 32

    def allowed_ids(self, buffer: str) -> np.ndarray:
        """Return the token ids allowed after ``buffer``.

        Args:
            buffer: Text generated so far for this value.

        Returns:
            Allowed token ids (never empty).
        """
        raise NotImplementedError

    def advance(self, buffer: str, token_text: str) -> tuple[str, bool]:
        """Apply the chosen token.

        Args:
            buffer: Text generated so far.
            token_text: Text of the chosen token.

        Returns:
            The new buffer (the value text only) and True if it is complete.
        """
        raise NotImplementedError

    def force_close(self, buffer: str) -> str:
        """Turn a buffer into a complete value when ``max_tokens`` is hit.

        Args:
            buffer: Unfinished text.

        Returns:
            A complete value text.
        """
        raise NotImplementedError

    def to_value(self, buffer: str) -> ArgumentValue:
        """Convert the finished buffer into a Python value.

        Args:
            buffer: Complete value text.

        Returns:
            The typed value.
        """
        raise NotImplementedError


class ChoiceConstraint(Constraint):
    """Pick exactly one string from a fixed list (function names, booleans).

    A token is allowed if ``buffer + token`` is still the beginning of
    ``option + follow`` for at least one option. The value is complete as
    soon as the buffer starts with a whole option. Options must not be
    prefixes of each other: function names get a closing ``"`` so that
    ``fn_a"`` and ``fn_ab"`` are told apart.

    Attributes:
        options: The possible complete values.
    """

    options: list[str]

    def allowed_ids(self, buffer: str) -> np.ndarray:
        """Return tokens that keep the buffer a prefix of an option.

        Args:
            buffer: Text generated so far.

        Returns:
            Allowed token ids.
        """
        allowed: set[int] = set()
        for option in self.options:
            target = option + self.follow
            if target.startswith(buffer):
                remaining = target[len(buffer):]
                allowed.update(self.vocab.ids_for_prefixes(remaining))
        return np.array(sorted(allowed), dtype=np.int64)

    def advance(self, buffer: str, token_text: str) -> tuple[str, bool]:
        """Append the token; finished once a whole option is written.

        Args:
            buffer: Text generated so far.
            token_text: Text of the chosen token.

        Returns:
            The new buffer (cut to the option when complete) and whether
            an option is complete.
        """
        buffer += token_text
        for option in self.options:
            if buffer.startswith(option):
                return option, True
        return buffer, False

    def force_close(self, buffer: str) -> str:
        """Complete the buffer with the first option it starts.

        Args:
            buffer: Unfinished text.

        Returns:
            A complete option.
        """
        for option in self.options:
            if option.startswith(buffer):
                return option
        return self.options[0]

    def to_value(self, buffer: str) -> ArgumentValue:
        """Return the chosen option (``true``/``false`` become booleans).

        Args:
            buffer: A complete option.

        Returns:
            The option text, or a bool for ``true``/``false``.
        """
        if buffer.strip() in ("true", "false"):
            return buffer.strip() == "true"
        return buffer


class NumberConstraint(Constraint):
    """Generate a number (``integer`` forbids the decimal point).

    Attributes:
        integer: True for the ``integer`` type, False for ``number``.
    """

    integer: bool = False
    max_tokens: int = 24

    def is_prefix(self, text: str) -> bool:
        """Tell whether text can still grow into a valid number.

        Args:
            text: Candidate number text.

        Returns:
            True if text is a valid number prefix.
        """
        if self.integer and "." in text:
            return False
        return NUMBER_PREFIX_RE.fullmatch(text) is not None

    def allowed_ids(self, buffer: str) -> np.ndarray:
        """Allow digits/sign/point that keep a valid number, then the end.

        Args:
            buffer: Number text generated so far.

        Returns:
            Allowed token ids.
        """
        allowed = [
            token_id
            for token_id in self.vocab.number_ids
            if self.is_prefix(buffer + self.vocab.text_of(token_id))
        ]
        if NUMBER_RE.fullmatch(buffer):
            allowed += self.vocab.ids_for_prefixes(self.follow)
        return np.array(allowed, dtype=np.int64)

    def advance(self, buffer: str, token_text: str) -> tuple[str, bool]:
        """Append a digit token, or finish on a token from ``follow``.

        Args:
            buffer: Number text generated so far.
            token_text: Text of the chosen token.

        Returns:
            The new buffer and whether the number is complete.
        """
        if self.is_prefix(buffer + token_text):
            return buffer + token_text, False
        return buffer, True

    def force_close(self, buffer: str) -> str:
        """Repair an unfinished number such as ``""``, ``"-"`` or ``"3."``.

        Args:
            buffer: Unfinished number text.

        Returns:
            A complete number text.
        """
        if NUMBER_RE.fullmatch(buffer):
            return buffer
        return buffer + "0"

    def to_value(self, buffer: str) -> ArgumentValue:
        """Parse the number (``number`` is always a float).

        Args:
            buffer: Complete number text.

        Returns:
            An int for ``integer``, a float for ``number``.
        """
        return int(buffer) if self.integer else float(buffer)


class StringConstraint(Constraint):
    """Generate a JSON string, from its opening quote to its closing quote.

    The buffer holds the raw text: an optional space, the opening quote and
    the escaped body (``' "hel'``). The model writes the opening quote
    itself because the tokenizer often merges it with what follows
    (``' ""'`` for an empty string, ``' "*'``...). Every allowed token is a
    valid string fragment on its own, so the body is always valid JSON,
    whatever the model chooses.
    """

    max_tokens: int = 64

    def allowed_ids(self, buffer: str) -> np.ndarray:
        """Allow an opening token first, then fragments or a closing token.

        Closing tokens hold the closing quote and possibly the start of
        ``follow``: ``"``, ``",`` or ``"}}`` when follow is ``"}}"``.

        Args:
            buffer: String text generated so far.

        Returns:
            Allowed token ids.
        """
        if not buffer:
            open_and_close = [
                token_id
                for token_id, rest in self.vocab.opening_quote_splits
                if self.follow.startswith(rest)
            ]
            return np.concatenate([
                self.vocab.opening_ids,
                np.array(open_and_close, dtype=np.int64),
            ])
        closing = [
            token_id
            for token_id, _body, rest in self.vocab.quote_splits
            if self.follow.startswith(rest)
        ]
        return np.concatenate(
            [self.vocab.string_body_ids, np.array(closing, dtype=np.int64)]
        )

    def advance(self, buffer: str, token_text: str) -> tuple[str, bool]:
        """Append the token, or finish if it holds the closing quote.

        Args:
            buffer: String text generated so far.
            token_text: Text of the chosen token.

        Returns:
            The new buffer (without the closing quote) and whether the
            string is closed.
        """
        if not buffer:
            # Move the opening quote (and its optional space) to the buffer.
            opening = token_text.index('"') + 1
            buffer, token_text = token_text[:opening], token_text[opening:]
        if is_json_string_fragment(token_text):
            return buffer + token_text, False
        split = split_at_closing_quote(token_text)
        body = split[0] if split is not None else ""
        return buffer + body, True

    def force_close(self, buffer: str) -> str:
        """Close the string as it is (the body is always valid).

        Args:
            buffer: String text generated so far.

        Returns:
            The buffer, with an opening quote if it had none.
        """
        return buffer or '"'

    def to_value(self, buffer: str) -> ArgumentValue:
        """Unescape the JSON body into a Python string.

        Args:
            buffer: Opening quote and escaped body.

        Returns:
            The decoded string.
        """
        body = buffer[buffer.index('"') + 1:]
        value: str = json.loads('"' + body + '"')
        return value
