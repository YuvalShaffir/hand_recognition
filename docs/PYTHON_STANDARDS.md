# Python Standards

Conventions already established in this codebase. Follow them for consistency;
they're enforced/checked by the tools in `pyproject.toml`'s `dev`
dependency group (black, flake8, mypy, pytest) plus the patterns below.

## Language & typing

- Target Python 3.13. Use modern typing syntax: `list[int]`, `dict[str, X]`,
  `X | None` — never `typing.List`, `typing.Optional`, etc.
- Type-hint function signatures (params and return type). Prefer `np.ndarray`
  for numpy arrays; a shape/dtype comment is fine where it isn't obvious
  (e.g. `# (T, 15) quantized angle vectors`).
- Dataclasses for structured records instead of dicts or tuples (see
  `GestureTemplate` in `domain.py`).

## Comments & docstrings

- No comments by default. Add one only when the *why* isn't obvious from the
  code itself — a non-obvious constraint, a workaround, or rationale that
  would surprise a reader (e.g. why hysteresis exists in
  `gestures/movement.py`, why `hand_world_landmarks` is used instead of
  `hand_landmarks`). Never restate
  what the code visibly does.
- Class/function docstrings: at most a short paragraph explaining intent,
  only when the name+signature don't already make it obvious. No
  multi-paragraph docstrings.

## Structure

- One concern per module (feature extraction, quantization, recording,
  matching, actions are separate files) rather than one large script.
- Expose read-only internal state through `@property` rather than reaching
  into `_private` attributes from other modules (see `GestureRecorder`).
- Module-level constants in `SCREAMING_SNAKE_CASE` (`FINGER_CHAINS`,
  `DEFAULT_BIN_SIZE_DEG`).

## Numpy

- Vectorize; avoid Python-level loops over array elements when a numpy
  expression does the same thing (see `quantize_angles`, the per-row cost in
  `dtw_distance`). A loop is fine where numpy already returns a scalar per
  iteration and vectorizing would hurt clarity for little gain.

## Tooling

Run before considering a change done:

```
uv run black .
uv run flake8
uv run mypy .
```

The `pytest` suite lives in `tests/`, mirroring the source tree, and runs
with no display, webcam, model file or network:

```
uv run pytest
```

All four commands run on every pull request - see
`.github/workflows/ci.yml`, which checks formatting, lint and types once and
runs the suite on both Linux and Windows.

It is specified by `docs/TEST_PLAN.md`; the coverage gate is on by default
(`--cov-fail-under=90`, with `apps/` and `model/` omitted), so add `--no-cov`
when running a single file.
