"""Tests for the value constraints, driven by a fake vocabulary."""

from src.constraints import (
    ChoiceConstraint,
    Constraint,
    NumberConstraint,
    StringConstraint,
)
from src.vocab import Vocabulary


def texts(vocab: Vocabulary, constraint: Constraint, buffer: str) -> set[str]:
    """Return the texts of the tokens allowed after buffer.

    Args:
        vocab: Fake vocabulary.
        constraint: Constraint under test.
        buffer: Text generated so far.

    Returns:
        Allowed token texts.
    """
    return {vocab.text_of(int(i)) for i in constraint.allowed_ids(buffer)}


def run(constraint: Constraint, tokens: list[str]) -> tuple[str, bool]:
    """Feed tokens to a constraint.

    Args:
        constraint: Constraint under test.
        tokens: Token texts, in order.

    Returns:
        The final buffer and whether the value is complete.
    """
    buffer, done = "", False
    for token in tokens:
        buffer, done = constraint.advance(buffer, token)
    return buffer, done


def test_choice_handles_names_that_are_prefixes(vocab: Vocabulary) -> None:
    """fn_add and fn_add_numbers are both reachable and distinct."""
    names = ChoiceConstraint(
        vocab=vocab, options=['fn_add"', 'fn_add_numbers"'], follow=", "
    )
    assert texts(vocab, names, "") == {"fn"}
    assert texts(vocab, names, "fn") == {"_a", "_add"}
    assert texts(vocab, names, "fn_add") == {'"', '",', "_numbers"}
    assert run(names, ["fn", "_add", '",']) == ('fn_add"', True)
    assert run(names, ["fn", "_add", "_numbers", '"']) == (
        'fn_add_numbers"', True
    )
    assert run(names, ["fn", "_add"]) == ("fn_add", False)
    assert names.force_close("fn_add_n") == 'fn_add_numbers"'


def test_choice_booleans(vocab: Vocabulary) -> None:
    """Booleans accept an optional leading space."""
    boolean = ChoiceConstraint(
        vocab=vocab, options=[" true", " false", "true", "false"], follow="}}"
    )
    assert texts(vocab, boolean, "") == {
        " ", " true", " false", "true", "false", "tr"
    }
    buffer, done = run(boolean, [" true"])
    assert done and boolean.to_value(buffer) is True
    buffer, done = run(boolean, ["false"])
    assert done and boolean.to_value(buffer) is False


def test_number_masks(vocab: Vocabulary) -> None:
    """Only valid number prefixes, and the end only once complete."""
    number = NumberConstraint(vocab=vocab, follow=", ")
    start = texts(vocab, number, "")
    assert {" -", "-", " ", "1", "12", "."} <= start
    assert "," not in start  # empty number cannot end
    assert "-" not in texts(vocab, number, "-")
    assert "," in texts(vocab, number, "12")
    assert ", " in texts(vocab, number, "12")
    assert "." not in texts(vocab, number, "1.5")
    assert "," not in texts(vocab, number, "1.")


def test_number_values(vocab: Vocabulary) -> None:
    """Numbers become floats, integers become ints."""
    number = NumberConstraint(vocab=vocab, follow="}}")
    buffer, done = run(number, [" -", "12", ".", "5", "}}"])
    assert done and number.to_value(buffer) == -12.5
    assert isinstance(number.to_value(" 3"), float)
    integer = NumberConstraint(vocab=vocab, follow="}}", integer=True)
    assert "." not in texts(vocab, integer, "1")
    assert integer.to_value("05") == 5
    assert isinstance(integer.to_value("5"), int)


def test_number_force_close(vocab: Vocabulary) -> None:
    """Unfinished numbers are repaired."""
    number = NumberConstraint(vocab=vocab)
    assert number.force_close("") == "0"
    assert number.force_close("-") == "-0"
    assert number.force_close("3.") == "3.0"
    assert number.force_close("42") == "42"


def test_string_flow(vocab: Vocabulary) -> None:
    """Open, write fragments, close with a follow-compatible token."""
    string = StringConstraint(vocab=vocab, follow="}}")
    start = texts(vocab, string, "")
    assert {' "', '"', ' "hi', ' ""'} <= start
    assert "hello" not in start  # the quote must come first
    inside = texts(vocab, string, ' "')
    assert {"hello", " world", "\\\\", "\\n", '"', '"}}'} <= inside
    assert '",' not in inside  # "," does not start "}}"
    assert "\\" not in inside and "\n" not in inside and 'a"b' not in inside
    buffer, done = run(string, [' "', "hello", "\\n", '"}}'])
    assert done and string.to_value(buffer) == "hello\n"


def test_string_edge_cases(vocab: Vocabulary) -> None:
    """Empty strings, merged tokens and escapes are decoded correctly."""
    string = StringConstraint(vocab=vocab, follow=", ")
    buffer, done = run(string, [' ""'])
    assert done and string.to_value(buffer) == ""
    buffer, done = run(string, [' "hi', " world", '",'])
    assert done and string.to_value(buffer) == "hi world"
    buffer, done = run(string, ['"', "\\\\", "é", '"'])
    assert done and string.to_value(buffer) == "\\é"
    assert string.to_value(string.force_close("")) == ""
