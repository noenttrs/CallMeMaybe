"""Build the text prompt given to the model.

We reuse the tool-calling layout Qwen3 was trained on: the functions are
listed as JSON inside ``<tools>`` tags, and the answer is a JSON object
inside ``<tool_call>`` tags with the keys ``name`` and ``arguments``.
Speaking the model's "native" format makes it choose better. Each function
is written in a compact form to keep the prompt short: the model reads the
whole prompt again at every step, so prompt length drives the speed.

The prompt only helps the model choose well (accuracy). It never
guarantees the structure: that is the job of the constraints.
"""

import json

from src.models import FunctionDefinition

# Key holding the arguments in the text the model sees (Qwen's own key).
# The output file uses "parameters", as the subject requires.
ARGUMENTS_KEY = "arguments"

SYSTEM_INSTRUCTIONS = (
    "You are a function calling assistant. Choose the ONE function that "
    "does what the user asks and fill in its arguments with values taken "
    "from the request. Copy text arguments exactly, without the quotes "
    "around them. When an argument describes a pattern or a symbol (for "
    "example a regular expression or a character), write the actual "
    "pattern or symbol."
)


def function_schema(function: FunctionDefinition) -> str:
    """Write one function as a compact JSON tool description.

    Args:
        function: The function to describe.

    Returns:
        One JSON line like ``{"name": ..., "description": ...,
        "parameters": {"a": "number"}}``.
    """
    schema = {
        "name": function.name,
        "description": function.description,
        "parameters": {
            name: spec.type for name, spec in function.parameters.items()
        },
    }
    return json.dumps(schema, ensure_ascii=False)


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    """Build the full chat prompt, up to the start of the tool call.

    The assistant turn starts with an empty ``<think></think>`` block so
    that Qwen3 answers directly instead of spending tokens on reasoning.

    Args:
        functions: Available functions.
        request: The user's natural-language request.

    Returns:
        The prompt text, ending where the JSON answer must start.
    """
    tools = "\n".join(function_schema(function) for function in functions)
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "# Tools\n\n"
        "You are provided with function signatures within <tools></tools> "
        f"XML tags:\n<tools>\n{tools}\n</tools>\n\n"
        "Return a json object with the function name and arguments within "
        "<tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        f'{{"name": <function-name>, "{ARGUMENTS_KEY}": <args-json-object>}}\n'
        "</tool_call><|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n"
    )
