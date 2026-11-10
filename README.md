*This activity has been created as part of the 42 curriculum by nonottar.*

# call me maybe

Function calling with a small LLM (Qwen3-0.6B) and **constrained decoding**.

## Description

The program reads natural-language requests such as *"What is the sum of
2 and 3?"* and, instead of answering them, produces the function call that
would answer them:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

A 0.6B-parameter model asked to "answer in JSON" often produces invalid
or incomplete JSON. Here the model never gets the chance to break the
format: at every generation step the program removes (masks) every token
that would break the JSON or the function's schema, so the model can only
choose among valid continuations. The **model** chooses the function and
the argument values; the **program** guarantees the structure.

The only libraries used by `src/` are the standard library, `numpy`,
`pydantic` and the provided `llm_sdk` (public methods only:
`get_logits_from_input_ids`, `get_path_to_vocab_file`, `encode`).

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/). The
first run downloads the model (~1.2 GB) from Hugging Face.

```bash
make install        # uv sync: creates .venv, installs numpy, pydantic, llm_sdk
make run            # uv run python -m src (default paths)
make debug          # same, under pdb
make test           # pytest unit tests
make lint           # flake8 + mypy with the subject's flags
make lint-strict    # flake8 + mypy --strict
make clean          # remove caches and data/output
```

Command line:

```bash
uv run python -m src [--functions_definition FILE] [--input FILE] [--output FILE]
```

Defaults: `data/input/functions_definition.json`,
`data/input/function_calling_tests.json` and
`data/output/function_calling_results.json` (the output folder is created
if needed).

### Project layout

```
src/
  __main__.py     entry point for `python -m src`
  cli.py          arguments, main loop, timing, error reporting
  models.py       pydantic models: definitions, prompts, output calls
  io_utils.py     reading/validating input files, writing the output
  vocab.py        token id -> real text, precomputed token groups
  constraints.py  which tokens are allowed for a name/number/string/bool
  decoder.py      the generation state machine (FunctionCaller)
  prompting.py    the text prompt given to the model
tests/            pytest unit tests + evaluate.py and evaluation sets
llm_sdk/          provided package (installed by uv as a path dependency)
data/input/       example input files
```

## Algorithm explanation

### 1. From token ids to text (`vocab.py`)

Qwen uses byte-level BPE. In `vocab.json` every byte is written as a
printable character (`Ġ` = space, `Ċ` = newline, `Ã©` = the two bytes of
`é`...). We rebuild GPT-2's `bytes_to_unicode` table, invert it, turn each
token back into bytes and decode them as UTF-8. Tokens that are only half
of a multi-byte character have no text on their own and are never
allowed. The result was checked against `llm_sdk.decode()` on 3,000
random tokens: 0 mismatches.

The vocabulary has ~150k tokens, so we never loop over all of them during
generation. At start-up (0.35 s) we precompute:

* `text_to_id`: text -> token id, used to find "all tokens that are a
  prefix of this text" with only `len(text)` dictionary lookups;
* `number_ids`: the ~50 tokens made only of digits, `.`, `-`, spaces;
* `string_body_ids`: the ~145k tokens that are valid *inside* a JSON
  string (no raw `"`, no lone `\`, no raw control character);
* `quote_splits` / `opening_*`: the ~1.3k tokens that close (or open) a
  string, split into "text before the quote" and "text after it".

### 2. The answer as a state machine (`decoder.py`)

The answer always has the same shape:

```
{"name": "<NAME>", "arguments": {"a":<VALUE>, "b":<VALUE>}}
```

Everything outside `<...>` is **fixed text**: we add its tokens ourselves
with `encode()`, without asking the model. Each `<...>` is generated
token by token under a constraint:

1. ask the model for the logits of the next token;
2. ask the constraint for the allowed token ids;
3. set all other logits to `-inf` with numpy and take the `argmax`;
4. give the token to the constraint, which updates the value text and
   says whether the value is finished.

If only one token is allowed, the model is not called at all.

The parameters come from the chosen function's definition, in order.
When everything is generated, each value is converted (`float`, `int`,
`json.loads` for strings), the call is validated with pydantic and checked
against the definition, then written with `json.dump`.

### 3. The constraints (`constraints.py`)

A value ends when the model picks a token that starts the next fixed
text (called `follow`, e.g. `", "` or `"}}"`). This is how the *model*
decides where a number ends or where a string closes.

* **Function name** (`ChoiceConstraint`): options are the names followed
  by a closing quote (`fn_add"`, `fn_add_numbers"`), so a name that is a
  prefix of another one is still handled. A token is allowed if
  `generated + token` is still the start of `option + follow` for some
  option. Allowed ids are found with `ids_for_prefixes` on the remaining
  text: a few dictionary lookups per option.
* **number / integer** (`NumberConstraint`): tokens are allowed if the
  text still matches ` ?-?[0-9]*(\.[0-9]*)?` (no `.` for integers). The
  `follow` tokens are allowed only once the number is complete
  (` ?-?[0-9]+(\.[0-9]+)?`). `number` values are output as floats (`2.0`,
  as in the subject's example), `integer` values as ints.
* **boolean**: a `ChoiceConstraint` over `true` / `false`.
* **string** (`StringConstraint`): first a token that opens the string
  (`"`, ` "`, ` "hi`, ` ""`...), then any token that is a valid string
  fragment by itself (so escapes like `\n` or `\\` are always complete),
  until a token holding the closing quote whose remaining text matches
  `follow` (`"`, `",`, `"}}`...).
* Every constraint has a length cap (24 tokens for numbers, 64 for
  strings). If it is reached the value is closed cleanly (`"3."` becomes
  `"3.0"`, a string is closed as it is).

## Design decisions

* **The model writes its own leading space and quotes.** The tokenizer
  merges a space with what follows (`": -2"` is `:`, ` -`, `2`;
  `": ""` is `:`, ` ""`). Our first version forced `": "` (with the
  space) and the opening quote; the model then never produced negative
  numbers or empty strings, because it had never seen a lone `-` after a
  space. Stopping the fixed text right after `:` fixed this.
* **Closing tokens can include the next fixed text** (`",`, `"}}`).
  Allowing only a bare `"` pushed the model to continue a name
  (`fn_reverse` -> `fn_reverse_words`) because the token it wanted
  (`",`) was forbidden.
* **Qwen's own tool-calling layout** for the prompt (`<tools>` and
  `<tool_call>`, key `arguments` inside the model's answer; the output
  file uses `parameters` as required). It raised function choice from
  92-96% to 100% on our sets. Each tool is written in a compact form
  (`{"name", "description", "parameters": {"a": "number"}}`), which cut
  the prompt from 491 to 322 tokens and generation time by ~40% with the
  same accuracy.
* **Empty `<think></think>` block**: Qwen3 answers directly instead of
  reasoning first.
* **Leading zeros accepted** inside the generated number (`6:05` gives
  `05`, parsed as 5): the raw text is parsed with `int()`/`float()`, the
  file is written by `json.dump`, so the output is always valid JSON.
* **Every class is a pydantic model** (input files, output calls,
  vocabulary, constraints, decoder). Input errors are turned into one
  readable message per problem; the program never shows a traceback.
* **Inputs are validated before loading the model**, so input errors
  are reported instantly.

## Performance analysis

Measured on an Apple M3 Pro (MPS GPU, float16), model already
downloaded. Accuracy is measured by `tests/evaluate.py`, which runs the
real CLI and compares with expected answers (regex arguments are judged
by applying them to the source string).

| Set | Prompts | Function choice | Fully correct | Time / prompt |
|---|---|---|---|---|
| Subject examples (`official_tests.json`) | 11 | 100% | 100% | 2.1 s |
| Own set: 11 functions, all 4 types, negatives, floats, escapes, unicode, empty string | 24 | 100% | 100% | 2.6 s |
| Hard set: no-parameter function, near-duplicate names, quotes/backslashes/emoji, 12-digit number | 12 | 92% | 75% | 2.3 s |
| **Total** | **47** | **98%** | **94%** | |

* **Valid JSON: 100%** of outputs in every run (47/47), including emoji,
  accented letters, backslashes, empty strings and huge numbers.
* **Speed**: full default run 31.5 s (model load 9 s, 11 prompts in
  22.6 s, 149 model calls). On CPU one model call takes 0.69 s instead of
  0.14 s, so the same run would take about 2 minutes: still under 5.
* `llm_sdk` has no KV cache: each call reads the whole prompt again, so
  speed depends on prompt length and on the number of model calls. Fixed
  text and forced tokens cost no model call.
* **Misses on the hard set** are model choices, not format errors: it
  picked `fn_translate(language="en")` over `fn_translate_to_english`,
  wrote `'hi'` instead of `"hi"` inside a string, and reused a text as a
  file path.
* On the GPU, float16 can make results differ slightly between runs when
  two tokens are almost tied (we saw `34|233` and `([0-9]+)` for the same
  prompt; both give the same result).

## Challenges faced

* **Token boundaries.** The tokenizer merges spaces and quotes with
  their neighbours, so forcing text that splits those merges puts the
  model in situations it never saw. Negative numbers, empty strings and
  `"*"` failed until values were allowed to start with their own space
  and quote.
* **Names that are prefixes of other names** (`fn_reverse` /
  `fn_reverse_words`): solved with the closing quote in the options and by
  allowing tokens like `",` that end the name.
* **Speed without scanning 150k tokens per step**: all token groups are
  precomputed once; per step we only filter small lists (≤1.3k tokens)
  or do dictionary lookups.
* **mypy and the `llm_sdk/` folder**: mypy resolved `import llm_sdk` to
  the project folder instead of the installed package; a mypy override in
  `pyproject.toml` treats it as an external untyped module.

## Testing strategy

* `make test`: 31 pytest unit tests that run in under a second with a
  fake vocabulary (no model needed): byte-level decoding, JSON fragments,
  each constraint's masks and values (prefix names, negative/float/
  integer numbers, escapes, empty and unicode strings, forced closing),
  model validation, input errors (missing file, directory, invalid JSON,
  empty array, wrong schema, unsupported type, duplicate names), output
  format and CLI exit codes.
* `tests/evaluate.py` on three evaluation sets in `tests/eval/` (two
  written for this project) to measure accuracy and speed end to end.
* Manual checks of every error case with the real CLI, and `make lint` /
  `make lint-strict`.

```bash
uv run python tests/evaluate.py tests/eval/extra_functions.json tests/eval/extra_tests.json
```

## Example usage

```bash
$ make run
Loading model...
Model ready in 8.9s
[1/11] 1.58s  fn_add_numbers({'a': 2.0, 'b': 3.0})
...
[11/11] 3.73s  fn_substitute_string_with_regex({'source_string': 'The cat sat on the mat with another cat', 'regex': 'cat', 'replacement': 'dog'})

Wrote 11 result(s) to data/output/function_calling_results.json
Generation: 22.6s for 11 prompt(s) (2.06s/prompt, 149 model calls)

$ uv run python -m src --input missing.json
Error: file not found: missing.json

$ uv run python -m src --functions_definition tests/eval/stress_functions.json \
    --input my_prompts.json --output data/output/stress.json
```

## Resources

* Subject: *call me maybe*, 42 (version 1.2).
* Qwen3 model card and chat/tool-calling template:
  https://huggingface.co/Qwen/Qwen3-0.6B
* GPT-2 byte-level BPE (`bytes_to_unicode`):
  https://github.com/openai/gpt-2/blob/master/src/encoder.py
* Willard & Louf, *Efficient Guided Generation for Large Language
  Models* (2023), the idea behind constrained decoding:
  https://arxiv.org/abs/2307.09702
* JSON grammar: https://www.json.org/
* pydantic v2 documentation: https://docs.pydantic.dev/
* NumPy documentation: https://numpy.org/doc/

### Use of AI

An AI coding assistant was used during this project to:
read the subject and propose a module layout; write versions of tests and this README; and debug accuracy problems (it ran the
experiments that found the token-boundary issues described above). Every
part was then reviewed, tested and explained so that it can be modified
by hand during the evaluation. Design choices and final measurements were
checked by running the program.
