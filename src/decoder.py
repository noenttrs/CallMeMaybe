"""Constrained decoding: turn one prompt into one valid function call.

The answer is always built in the same order, and only the variable parts
are asked to the model::

    {"name": "<NAME>", "arguments": {"a":<VALUE>, "b":<VALUE>}}

(``arguments`` is the key Qwen was trained with; the output file uses
``parameters``.) Everything outside ``<...>`` is fixed text that we add
ourselves, without calling the model. Each ``<...>`` is generated token
by token under a constraint that masks every token which would break the
schema.
"""

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.constraints import (
    ChoiceConstraint,
    Constraint,
    NumberConstraint,
    StringConstraint,
)
from src.models import (
    ArgumentValue,
    FunctionCall,
    FunctionDefinition,
    ParameterSpec,
)
from src.prompting import ARGUMENTS_KEY, build_prompt
from src.vocab import Vocabulary


class FunctionCaller(BaseModel):
    """Generate function calls with a model and the constraints.

    Attributes:
        llm: The ``Small_LLM_Model`` from llm_sdk (public methods only).
        vocab: Decoded vocabulary of the same model.
        functions: The functions the model may call.
        model_calls: Number of forward passes done so far (statistics).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    vocab: Vocabulary
    functions: list[FunctionDefinition]
    model_calls: int = 0

    def encode(self, text: str) -> list[int]:
        """Tokenize text with the model's tokenizer.

        Args:
            text: Text to tokenize.

        Returns:
            The token ids as plain Python ints.
        """
        rows = self.llm.encode(text).tolist()
        return [int(token_id) for token_id in rows[0]]

    def next_token(self, ids: list[int], allowed: np.ndarray) -> int:
        """Choose the best allowed token after ``ids``.

        If only one token is allowed the model is not called at all.

        Args:
            ids: Full context (prompt + answer so far).
            allowed: Token ids allowed by the constraint.

        Returns:
            The chosen token id.
        """
        if len(allowed) == 1:
            return int(allowed[0])
        logits = np.asarray(
            self.llm.get_logits_from_input_ids(ids), dtype=np.float64
        )
        self.model_calls += 1
        masked = np.full(logits.shape, -np.inf)
        masked[allowed] = logits[allowed]
        return int(np.argmax(masked))

    def generate(self, ids: list[int], constraint: Constraint) -> str:
        """Generate one value token by token under a constraint.

        Args:
            ids: Context ending right where the value starts.
            constraint: Rules for this value.

        Returns:
            The complete value text.
        """
        context = list(ids)
        buffer = ""
        for _ in range(constraint.max_tokens):
            token_id = self.next_token(context, constraint.allowed_ids(buffer))
            buffer, done = constraint.advance(
                buffer, self.vocab.text_of(token_id)
            )
            if done:
                return buffer
            context.append(token_id)
        return constraint.force_close(buffer)

    def constraint_for(self, spec: ParameterSpec, follow: str) -> Constraint:
        """Create the constraint matching a parameter type.

        Args:
            spec: Parameter type.
            follow: Fixed text written after the value.

        Returns:
            The constraint to use.
        """
        if spec.type == "string":
            return StringConstraint(vocab=self.vocab, follow=follow)
        if spec.type == "boolean":
            return ChoiceConstraint(
                vocab=self.vocab,
                follow=follow,
                options=[" true", " false", "true", "false"],
            )
        return NumberConstraint(
            vocab=self.vocab, follow=follow, integer=spec.type == "integer"
        )

    def call(self, request: str) -> FunctionCall:
        """Produce a validated function call for one request.

        Args:
            request: Natural-language prompt.

        Returns:
            The function call, checked against its definition.
        """
        prompt_ids = self.encode(build_prompt(self.functions, request))
        arguments_start = f', "{ARGUMENTS_KEY}": {{'

        # 1. Function name: one of the known names, followed by a quote.
        answer = '{"name": "'
        name_constraint = ChoiceConstraint(
            vocab=self.vocab,
            options=[function.name + '"' for function in self.functions],
            follow=arguments_start,
            max_tokens=max(len(f.name) for f in self.functions) + 20,
        )
        name_text = self.generate(
            prompt_ids + self.encode(answer), name_constraint
        )
        function = next(f for f in self.functions if f.name + '"' == name_text)
        answer += name_text + arguments_start

        # 2. Arguments, in the order of the definition. The fixed text
        # stops right after the colon: the model writes the value with its
        # own leading space (" -2", " true", ' "hello'), as it was trained.
        arguments: dict[str, ArgumentValue] = {}
        parameters = list(function.parameters.items())
        for index, (key, spec) in enumerate(parameters):
            is_last = index == len(parameters) - 1
            follow = "}}" if is_last else ", "
            answer += json.dumps(key) + ":"
            constraint = self.constraint_for(spec, follow)
            value_text = self.generate(
                prompt_ids + self.encode(answer), constraint
            )
            arguments[key] = constraint.to_value(value_text)
            closing_quote = '"' if spec.type == "string" else ""
            answer += value_text + closing_quote + follow
        if not parameters:
            answer += "}}"

        call = FunctionCall(
            prompt=request, name=function.name, parameters=arguments
        )
        function.check_arguments(call.parameters)
        return call
