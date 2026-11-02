"""Reading the input files and writing the results, with clear errors."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from src.models import FunctionCall, FunctionDefinition, PromptItem


def read_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: File to read.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If the file is missing, unreadable or not valid JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}") from None
    except IsADirectoryError:
        raise ValueError(f"expected a file, got a directory: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in {path} (line {error.lineno}, "
            f"column {error.colno}): {error.msg}"
        ) from None
    except UnicodeDecodeError:
        raise ValueError(f"file is not valid UTF-8 text: {path}") from None
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error.strerror}") from None


def _format_validation_error(error: ValidationError) -> str:
    """Turn a pydantic error into a short, readable message.

    Args:
        error: The validation error raised by pydantic.

    Returns:
        One line per problem, like ``[0].parameters.a.type: unsupported ...``.
    """
    lines = []
    for problem in error.errors():
        location = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in problem["loc"]
        )
        lines.append(f"  {location or '.'}: {problem['msg']}")
    return "\n".join(lines)


def _parse_list(data: Any, model: type[BaseModel], path: Path) -> list[Any]:
    """Validate that data is a non-empty JSON array of ``model`` objects.

    Args:
        data: Parsed JSON content.
        model: Pydantic model each element must match.
        path: Source file, used in error messages.

    Returns:
        The validated model instances.

    Raises:
        ValueError: If the data is not a non-empty list of valid objects.
    """
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    if not data:
        raise ValueError(f"{path} contains an empty array")
    items = []
    for index, raw in enumerate(data):
        try:
            items.append(model.model_validate(raw))
        except ValidationError as error:
            raise ValueError(
                f"invalid entry #{index} in {path}:\n"
                f"{_format_validation_error(error)}"
            ) from None
    return items


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        The function definitions, in file order.

    Raises:
        ValueError: On any I/O, JSON or schema problem.
    """
    functions: list[FunctionDefinition] = _parse_list(
        read_json(path), FunctionDefinition, path
    )
    names = [function.name for function in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate function names in {path}: {', '.join(duplicates)}"
        )
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the test prompts file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        The prompts, in file order.

    Raises:
        ValueError: On any I/O, JSON or schema problem.
    """
    prompts: list[PromptItem] = _parse_list(read_json(path), PromptItem, path)
    return prompts


def write_results(path: Path, calls: list[FunctionCall]) -> None:
    """Write the function calls as a JSON array, creating parent folders.

    Args:
        path: Output file.
        calls: Validated function calls.

    Raises:
        ValueError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                [call.model_dump() for call in calls],
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")
    except OSError as error:
        raise ValueError(f"cannot write {path}: {error.strerror}") from None
