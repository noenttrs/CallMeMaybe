"""Command-line entry point: read inputs, run the model, write results."""

import argparse
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from src.io_utils import load_functions, load_prompts, write_results
from src.models import FunctionCall

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


class Settings(BaseModel):
    """Paths given on the command line.

    Attributes:
        functions_definition: Function definitions file.
        input: Test prompts file.
        output: Results file to write.
    """

    functions_definition: Path = DEFAULT_FUNCTIONS
    input: Path = DEFAULT_INPUT
    output: Path = DEFAULT_OUTPUT


def parse_args(argv: list[str] | None = None) -> Settings:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name (None: use sys.argv).

    Returns:
        The validated settings.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument("--functions_definition", type=Path,
                        default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return Settings(**vars(parser.parse_args(argv)))


def error(message: str) -> None:
    """Print an error message on stderr.

    Args:
        message: Text to display.
    """
    print(f"Error: {message}", file=sys.stderr)


def run(settings: Settings) -> int:
    """Run the whole pipeline.

    Args:
        settings: Input and output paths.

    Returns:
        The process exit code (0 when every prompt succeeded).
    """
    # Check the inputs before loading the model, which takes a while.
    try:
        functions = load_functions(settings.functions_definition)
        prompts = load_prompts(settings.input)
    except ValueError as exc:
        error(str(exc))
        return 1

    start = time.perf_counter()
    print("Loading model...", flush=True)
    try:
        # Imported here so that input errors are reported instantly.
        from llm_sdk import Small_LLM_Model

        from src.decoder import FunctionCaller
        from src.vocab import Vocabulary

        llm = Small_LLM_Model()
        vocab = Vocabulary.from_file(Path(llm.get_path_to_vocab_file()))
    except Exception as exc:
        error(f"cannot load the model: {exc}")
        return 1
    caller = FunctionCaller(llm=llm, vocab=vocab, functions=functions)
    print(f"Model ready in {time.perf_counter() - start:.1f}s", flush=True)

    calls: list[FunctionCall] = []
    failures = 0
    generation_start = time.perf_counter()
    for number, item in enumerate(prompts, start=1):
        prompt_start = time.perf_counter()
        try:
            call = caller.call(item.prompt)
        except Exception as exc:
            failures += 1
            error(f"prompt {number} ({item.prompt!r}) failed: {exc}")
            continue
        calls.append(call)
        print(
            f"[{number}/{len(prompts)}] "
            f"{time.perf_counter() - prompt_start:.2f}s  "
            f"{call.name}({call.parameters})",
            flush=True,
        )

    try:
        write_results(settings.output, calls)
    except ValueError as exc:
        error(str(exc))
        return 1

    elapsed = time.perf_counter() - generation_start
    print(
        f"\nWrote {len(calls)} result(s) to {settings.output}\n"
        f"Generation: {elapsed:.1f}s for {len(prompts)} prompt(s) "
        f"({elapsed / len(prompts):.2f}s/prompt, "
        f"{caller.model_calls} model calls)\n"
        f"Total time: {time.perf_counter() - start:.1f}s"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Program entry point; never lets an exception escape.

    Args:
        argv: Arguments without the program name (None: use sys.argv).

    Returns:
        The process exit code.
    """
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        error("interrupted by user")
        return 130
    except Exception as exc:
        error(f"unexpected failure: {exc}")
        return 1
