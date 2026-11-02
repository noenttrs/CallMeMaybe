"""Tests for the command-line error handling (no model needed)."""

from pathlib import Path

import pytest

from src.cli import main, parse_args


def test_defaults() -> None:
    """Default paths match the subject."""
    settings = parse_args([])
    assert settings.functions_definition == Path(
        "data/input/functions_definition.json"
    )
    assert settings.input == Path("data/input/function_calling_tests.json")
    assert settings.output == Path("data/output/function_calling_results.json")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "file not found"),
        ("not json", "invalid JSON"),
        ("[]", "empty array"),
        ('[{"wrong": "schema"}]', "invalid entry"),
    ],
)
def test_input_errors_are_reported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str | None,
    message: str,
) -> None:
    """Bad input files end with exit code 1 and a clear message."""
    prompts = tmp_path / "prompts.json"
    if content is not None:
        prompts.write_text(content, encoding="utf-8")
    code = main(["--input", str(prompts),
                 "--output", str(tmp_path / "out.json")])
    assert code == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
