"""Tests for the pydantic models and the input/output helpers."""

import json
from pathlib import Path

import pytest

from src.io_utils import load_functions, load_prompts, write_results
from src.models import FunctionCall, FunctionDefinition


def write(path: Path, content: str) -> Path:
    """Write a text file.

    Args:
        path: Destination.
        content: Text to write.

    Returns:
        The path.
    """
    path.write_text(content, encoding="utf-8")
    return path


ADD = {
    "name": "fn_add",
    "description": "Add",
    "parameters": {"a": {"type": "number"}, "n": {"type": "integer"},
                   "s": {"type": "string"}, "b": {"type": "boolean"}},
    "returns": {"type": "number"},
}


def test_load_valid_files(tmp_path: Path) -> None:
    """Valid files load into models, keeping parameter order."""
    functions = load_functions(write(tmp_path / "f.json", json.dumps([ADD])))
    assert list(functions[0].parameters) == ["a", "n", "s", "b"]
    prompts = load_prompts(
        write(tmp_path / "p.json", '[{"prompt": "hi"}, {"prompt": ""}]')
    )
    assert [p.prompt for p in prompts] == ["hi", ""]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[{", "invalid JSON"),
        ("[]", "empty array"),
        ('{"prompt": "x"}', "JSON array"),
        ('[{"text": "x"}]', "invalid entry #0"),
        ('[{"prompt": 3}]', "invalid entry #0"),
    ],
)
def test_bad_prompt_files(tmp_path: Path, content: str, message: str) -> None:
    """Broken prompt files give a clear ValueError."""
    with pytest.raises(ValueError, match=message):
        load_prompts(write(tmp_path / "p.json", content))


def test_missing_file(tmp_path: Path) -> None:
    """A missing file is reported, not raised as OSError."""
    with pytest.raises(ValueError, match="file not found"):
        load_prompts(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="directory"):
        load_prompts(tmp_path)


@pytest.mark.parametrize(
    ("functions", "message"),
    [
        ([{**ADD, "parameters": {"x": {"type": "array"}}}], "unsupported"),
        ([{**ADD, "name": ""}], "must not be empty"),
        ([{"name": "f"}], "parameters"),
        ([ADD, ADD], "duplicate"),
    ],
)
def test_bad_function_files(
    tmp_path: Path, functions: list[object], message: str
) -> None:
    """Invalid definitions are rejected with a clear message."""
    with pytest.raises(ValueError, match=message):
        load_functions(write(tmp_path / "f.json", json.dumps(functions)))


def test_check_arguments() -> None:
    """Types must match the definition exactly."""
    definition = FunctionDefinition.model_validate(ADD)
    definition.check_arguments({"a": 1.0, "n": 2, "s": "x", "b": True})
    with pytest.raises(ValueError, match="should be number"):
        definition.check_arguments({"a": 1, "n": 2, "s": "x", "b": True})
    with pytest.raises(ValueError, match="should be integer"):
        definition.check_arguments({"a": 1.0, "n": True, "s": "x", "b": True})
    with pytest.raises(ValueError, match="do not match"):
        definition.check_arguments({"a": 1.0})


def test_write_results(tmp_path: Path) -> None:
    """Output has exactly prompt/name/parameters and keeps types."""
    call = FunctionCall(
        prompt='Say "hé"', name="fn", parameters={"a": 2.0, "n": 3, "b": False}
    )
    output = tmp_path / "new_dir" / "out.json"
    write_results(output, [call])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data == [{"prompt": 'Say "hé"', "name": "fn",
                     "parameters": {"a": 2.0, "n": 3, "b": False}}]
    assert "2.0" in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        FunctionCall.model_validate({"prompt": "", "name": "f",
                                     "parameters": {}, "extra": 1})
