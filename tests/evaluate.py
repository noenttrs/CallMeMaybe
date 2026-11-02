"""Measure accuracy: run the real CLI on a test set and compare results.

Usage:
    uv run python tests/evaluate.py FUNCTIONS_FILE EXPECTED_FILE

EXPECTED_FILE is a JSON array of ``{"prompt", "expected": {"name",
"parameters"}}``. Only the prompts are given to the program.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def same_value(key: str, got: Any, want: Any, params: dict[str, Any]) -> bool:
    """Compare one argument, judging regexes by what they do.

    Args:
        key: Argument name.
        got: Value produced by the program.
        want: Expected value.
        params: All produced arguments (for the regex check).

    Returns:
        True if the argument is considered correct.
    """
    if key == "regex" and "source_string" in params:
        try:
            source = params["source_string"]
            return bool(
                re.sub(got, "#", source) == re.sub(want, "#", source)
            )
        except (re.error, TypeError):
            return False
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        return isinstance(got, (int, float)) and float(got) == float(want)
    return bool(got == want)


def main() -> int:
    """Run the evaluation and print a report.

    Returns:
        Exit code (0 on success).
    """
    functions, expected_file = Path(sys.argv[1]), Path(sys.argv[2])
    cases = json.loads(expected_file.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        prompts = Path(tmp) / "prompts.json"
        output = Path(tmp) / "out.json"
        prompts.write_text(
            json.dumps([{"prompt": case["prompt"]} for case in cases])
        )
        run = subprocess.run(
            [sys.executable, "-m", "src", "--functions_definition",
             str(functions), "--input", str(prompts), "--output", str(output)],
            capture_output=True, text=True,
        )
        print(run.stdout)
        if run.returncode != 0:
            print(run.stderr)
        results = json.loads(output.read_text(encoding="utf-8"))

    by_prompt = {r["prompt"]: r for r in results}
    correct_names = correct = 0
    for case in cases:
        want = case["expected"]
        got = by_prompt.get(case["prompt"])
        name_ok = got is not None and got["name"] == want["name"]
        args_ok = (
            got is not None
            and name_ok
            and set(got["parameters"]) == set(want["parameters"])
            and all(
                same_value(key, got["parameters"][key], value,
                           got["parameters"])
                for key, value in want["parameters"].items()
            )
        )
        correct_names += name_ok
        correct += args_ok
        if not args_ok:
            print(f"MISS {case['prompt']!r}\n  got  {got}\n  want {want}")
    total = len(cases)
    print(f"\nFunction choice: {correct_names}/{total} "
          f"({100 * correct_names / total:.0f}%)")
    print(f"Fully correct:   {correct}/{total} ({100 * correct / total:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
