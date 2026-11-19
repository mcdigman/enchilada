# Block-Owned Initial Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The `Block` protocol gains a required `initial_sample(conditional_residual, covariance, *, rng) -> BlockState`; `Wheel.add` calls it once at registration and adopts the result like a `sample` return; `initial_block_state` disappears from `Wheel.add` and `check_block`; every block, test, example, notebook and document follows.

**Architecture:** One mechanism in three source files (`block.py`, `wheel.py`, `testing.py`), landed first with its own inverted test file, then four independent migrations (the two big Wheel test files; the six smaller test files plus the mypy fixtures; the example scripts, benchmark and their tests; the four notebooks) that can run in parallel, then the README, CHANGELOG and a wording sweep. `Wheel.add` builds a new block's inputs with the same helper `run` uses, generalised to take the block's grid and to start from the working residual, calls `initial_sample` with a candidate copy of the campaign RNG, canonicalises the returned state exactly as `run` does, and commits through the existing `_adopt` with `source_method_name="initial_sample"`.

**Tech Stack:** Python 3.12+, NumPy, the existing `Wheel`/`BlockState`/`transform` machinery; pytest, ruff, mypy; notebooks edited as JSON by cell id (`demo.ipynb` by text substitution).

**User decisions (already made):**
- `initial_sample(self, conditional_residual: L1Data, covariance: Covariance | None, *, rng: np.random.Generator) -> BlockState`, i.e. `sample` minus `current_block_state` (user, 2026-11-18).
- No override: `Wheel.add` always calls `initial_sample`; `initial_block_state` is removed from `Wheel.add` and `check_block`; a chosen start (an injection, a checkpoint) is block configuration, rendered by `initial_sample` (user, 2026-11-18).
- Defaults recorded in the spec, not open: the campaign RNG through the candidate/commit path (registration consumes draws); the inputs are the working residual on the block's grid plus the covariance, which must be on that grid or `add` refuses it; adoption through the existing `_adopt`; the explicit pre-translation `check_on_grid` of the old `add` is dropped in favour of the `run` path's treatment.
- This reverses the decision of commit 5b6ff3b ("Registration ... never calls model initialization or consumes random draws"). Reviewers must not restore it.

**Spec:** `docs/superpowers/specs/2026-11-18-block-initial-sample-design.md`. **Branch:** `block-initial-sample`, cut from `main` at 1e2872b (already created; the spec is committed on it as 46ba987). Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

**Tooling rules (binding for every task):**
- Run checks as `.venv/bin/python -m pytest ...`, `.venv/bin/python -m ruff check src tests examples benchmarks`, `.venv/bin/python -m ruff format --check src tests examples benchmarks` (formatting itself: `.venv/bin/python -m ruff format <the .py files you changed>`; **never format a notebook**), `.venv/bin/python -m mypy`. Never `uv run`, `uvx`, `uv sync`, `uv lock`, prek. No dependency changes anywhere in this plan, so `uv.lock` is untouched.
- Full suite: `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q --cov=enchilada` (571 tests, 4 expected skips, coverage gate 95 %).
- Temporary files under `$TMPDIR`. Stage named paths only (never `git add -A`, `.` or `-u`).
- Notebooks: edit `create_block.ipynb`, `domain_translation.ipynb` and `gb_block_eryn.ipynb` by cell id with a JSON script (`json.load`, edit `cell["source"]` as a list of lines ending in `\n` except the last, `json.dump(nb, f, indent=1, ensure_ascii=False)` plus a trailing newline; all three round-trip byte for byte, verify with a no-op dump before editing). `demo.ipynb` does **not** round-trip: edit it by exact text substitution on the raw file.

**Task map and parallelism.** Task 1 first. Tasks 2, 3, 4 and 5 depend only on Task 1 and touch disjoint files, so the controller may dispatch them in parallel (two at a time is comfortable). Task 6 depends on all of them. The suite is red between Task 1 and the end of Task 5 by construction (every old registration passes the removed keyword); each task's **Verify** names the files that must be green at its commit, and Task 6 restores the whole suite. Do not push before Task 6.

| Task | Files | Blocked by |
|---|---|---|
| 1 Protocol, `Wheel.add`, testing helpers, inverted test file | `src/enchilada/{block,wheel,testing}.py`, `tests/conftest.py`, `tests/test_initial_sample.py` (renamed from `tests/test_initial_block_state.py`), `tests/test_testing.py` | — |
| 2 Wheel test migration | `tests/test_wheel.py`, `tests/test_wheel_translation.py` | 1 |
| 3 Remaining test migration and typing fixtures | `tests/test_state.py`, `tests/test_block_state.py`, `tests/test_block_state_contract.py`, `tests/test_ledger.py`, `tests/test_translation_review.py`, `tests/test_typing.py` | 1 |
| 4 Examples, benchmark and their tests | `examples/{toy_fit,gb_model,domain_translation,demo,mojito_wdm_wheel}.py`, `benchmarks/benchmark_wheel.py`, `tests/{test_examples,test_toy_translation,test_gb_example_contract}.py` | 1 |
| 5 Notebooks | `examples/{create_block,demo,domain_translation,gb_block_eryn}.ipynb` | 1 |
| 6 README, CHANGELOG, wording sweep, full gates | `README.md`, `CHANGELOG.md`, anything the sweep finds | 2, 3, 4, 5 |

---

### Task 1: Protocol, `Wheel.add`, testing helpers, inverted test file

**Goal:** `Block` requires `initial_sample`; `Wheel.add(block, *, data_domain=None, num_frequency_divisions=None, num_time_divisions=None)` calls it with the block's inputs and the candidate RNG and adopts the result through `_adopt`; `check_block` and `EchoBlock` follow; `tests/test_initial_sample.py` replaces `tests/test_initial_block_state.py` with the inverted tests, and `tests/conftest.py` gains the `starting_at` shorthand the migrations use.

**Files:**
- Modify: `src/enchilada/block.py` (docstring, new protocol method)
- Modify: `src/enchilada/wheel.py:40-56` (class docstring), `:152-245` (`add`), `:247-266` (`_block_inputs`), `:332` (the call in `run`)
- Modify: `src/enchilada/testing.py` (`EchoBlock.initial_sample`, `check_block` signature and docstring)
- Modify: `tests/conftest.py` (add `starting_at`)
- Rename and rewrite: `tests/test_initial_block_state.py` → `tests/test_initial_sample.py` (`git mv`, then replace the contents)
- Modify: `tests/test_testing.py`

**Acceptance Criteria:**
- [ ] `isinstance(x, Block)` is False for an object with `name` and `sample` but no `initial_sample`, and `Wheel.add` refuses it with `TypeError` matching `does not implement initial_sample`, leaving the ledger, block list and RNG unchanged.
- [ ] `Wheel.add` calls `initial_sample` exactly once, with the working residual on the block's grid (an independent copy), the covariance when the campaign has one (`None` otherwise), and a `np.random.Generator`; `run` does not call it again.
- [ ] Two Wheels seeded alike produce identical starts and identical first samples; a start that draws from `rng` advances the campaign stream (the first `sample` no longer sees the seed's first draw), a start that draws nothing leaves it in place.
- [ ] A raising `initial_sample`, a non-`BlockState` return (`TypeError` `"<name>.initial_sample must return a BlockState"`), an off-grid template, a non-finite template, a covariance on another grid, and a state that cannot be deep-copied each leave `ledger`, `_blocks`, `_block_domains`, the residual and `_rng` unchanged, on the observation grid and on a translated block grid.
- [ ] A covariance on another grid than the block's is refused at `add` with the existing `does not translate between grids` message naming the block and both grids, before `initial_sample` is called.
- [ ] Block B registered after block A receives the observations minus A's adopted start, on B's grid; a noise block's start covariance is handed to the next block's `initial_sample`.
- [ ] `check_block(block, observed, num_cycles=2, *, initial_covariance=None, random_seed=0)` registers through `initial_sample`; `EchoBlock.initial_sample` returns `BlockState()`.
- [ ] `tests/test_initial_sample.py` and `tests/test_testing.py` pass; `ruff check`, `ruff format --check` and `mypy` are clean for `src` and these two test files.

**Verify:** `.venv/bin/python -m pytest -q tests/test_initial_sample.py tests/test_testing.py` → all passed, 0 failed; `.venv/bin/python -m mypy` → `Success: no issues found`.

**Steps:**

- [ ] **Step 1: Rename the test file and write the inverted tests**

```bash
git mv tests/test_initial_block_state.py tests/test_initial_sample.py
```

Replace the whole contents of `tests/test_initial_sample.py` with:

```python
"""Wheel.add draws each block's start from its initial_sample and adopts it like a sample."""

import warnings
from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import (
    Block,
    BlockState,
    Covariance,
    L1Data,
    NoiseOverwrittenWarning,
    Wheel,
    transform,
)
from enchilada.testing import check_block


@pytest.fixture
def observed():
    return L1Data(
        channel_data={"A": np.full(64, 5.0)},
        channel_names=("A",),
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )


class StartingBlock:
    """Starts at a configured amplitude on whatever grid it is handed.

    `state.starts` records every `initial_sample` call's inputs; `state.counter`
    and `state.draw` record `sample` calls, as the old SeededBlock did.
    """

    name = "starting"

    @dataclass
    class State:
        counter: int = 0
        draw: float = 0.0
        starts: list = field(default_factory=list)

    def __init__(self, amplitude=2.0):
        self.amplitude = amplitude
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self.state.starts.append((conditional_residual, covariance, rng))
        time_reference = transform(conditional_residual, "time")
        start = time_reference.block_state(
            {"A": np.full(64, self.amplitude)},
            parameters={"amplitude": self.amplitude},
        )
        grid = conditional_residual.wdm_grid
        grid_options = (
            {} if grid is None else {"num_frequency_divisions": grid.num_frequency_divisions}
        )
        return transform(
            start,
            conditional_residual.data_domain,
            reference_data=time_reference,
            **grid_options,
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        assert current_block_state.parameters["amplitude"] == self.amplitude
        assert (
            current_block_state.template["A"].shape
            == conditional_residual.channel_data["A"].shape
        )
        self.state.counter += 1
        self.state.draw = rng.normal()
        return current_block_state


def noise_start(level):
    """A StartingBlock whose start publishes only white noise of `level`."""

    class NoiseStart(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            self.state.starts.append((conditional_residual, covariance, rng))
            return BlockState(covariance=white_covariance(conditional_residual, level))

    return NoiseStart()


def test_block_without_initial_sample_is_refused(observed):
    class SampleOnly:
        name = "sample_only"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    assert not isinstance(SampleOnly(), Block)
    assert isinstance(StartingBlock(), Block)
    wheel = Wheel(observed, random_seed=1)
    before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises(TypeError, match="does not implement initial_sample"):
        wheel.add(SampleOnly())
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._rng.bit_generator.state == before


def test_registration_calls_initial_sample_once_with_the_block_inputs(observed):
    wheel = Wheel(observed, random_seed=5)
    block = StartingBlock()
    wheel.add(block)
    assert len(block.state.starts) == 1
    residual, covariance, rng = block.state.starts[0]
    np.testing.assert_array_equal(residual.channel_data["A"], 5.0)
    assert covariance is None
    assert isinstance(rng, np.random.Generator)
    np.testing.assert_array_equal(wheel.contribution("starting")["A"], 2.0)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 3.0)
    # The block received its own copy of the residual.
    residual.channel_data["A"][:] = 0.0
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 3.0)
    wheel.run(1)
    assert len(block.state.starts) == 1
    assert block.state.counter == 1


def test_initial_sample_receives_the_campaign_covariance(observed):
    wheel = Wheel(observed, initial_covariance=white_covariance(observed, 0.25))
    block = StartingBlock()
    wheel.add(block)
    _, covariance, _ = block.state.starts[0]
    assert isinstance(covariance, Covariance)
    np.testing.assert_array_equal(covariance.A, 0.25)
    assert covariance is not wheel._covariance


def test_initial_sample_consumes_the_campaign_rng_reproducibly(observed):
    class PriorStart(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            self.amplitude = float(rng.normal())
            return super().initial_sample(conditional_residual, covariance, rng=rng)

    def start(seed, block):
        wheel = Wheel(observed, random_seed=seed)
        wheel.add(block)
        wheel.run(1)
        return wheel.ledger["starting"].parameters["amplitude"], block.state.draw

    same_a, same_b, other = (start(seed, PriorStart()) for seed in (11, 11, 12))
    assert same_a == same_b
    assert same_a != other
    first_draw = np.random.default_rng(11).normal()
    # The start took the seed's first draw, so the first sample sees the second.
    assert same_a[0] == first_draw
    assert same_a[1] != first_draw
    # A start that draws nothing leaves the stream where it was.
    _, draw = start(11, StartingBlock())
    assert draw == first_draw


@pytest.mark.parametrize(
    ("canonical_domain", "block_domain"),
    [
        ("time", None),
        ("frequency", "time"),
        ("time", "frequency"),
        ("time", "wdm"),
        ("wdm", "wdm"),
    ],
)
def test_initial_sample_receives_inputs_on_the_selected_grid_and_round_trips(
    observed, canonical_domain, block_domain
):
    canonical_options = (
        {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    )
    canonical = transform(observed, canonical_domain, **canonical_options)
    block_options = {"num_time_divisions": 4} if block_domain == "wdm" else {}
    wheel = Wheel(canonical, random_seed=23)
    block = StartingBlock()
    wheel.add(block, data_domain=block_domain, **block_options)
    residual, _, _ = block.state.starts[0]
    assert residual.data_domain == (block_domain or canonical_domain)
    if block_domain == "wdm":
        assert residual.wdm_grid.num_time_divisions == 4
    np.testing.assert_allclose(transform(residual, "time").channel_data["A"], 5.0)
    assert block.state.counter == 0
    np.testing.assert_allclose(transform(wheel.residual(), "time").channel_data["A"], 3)
    np.testing.assert_array_equal(
        wheel.observed_data.channel_data["A"], canonical.channel_data["A"]
    )
    assert wheel.contribution("starting")["A"].shape == canonical.channel_data["A"].shape
    wheel.run(1)
    assert block.state.counter == 1
    assert block.state.draw == np.random.default_rng(23).normal()


def test_adopted_start_is_an_independent_snapshot(observed):
    returned = {}

    class NoisyStart(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            start = replace(
                super().initial_sample(conditional_residual, covariance, rng=rng),
                covariance=white_covariance(conditional_residual, 0.25),
                parameters={"amplitude": 2.0, "positions": [np.array([1, 2])]},
            )
            returned["start"] = start
            return start

    wheel = Wheel(observed)
    wheel.add(NoisyStart())
    start = returned["start"]
    start.template["A"][:] = 100
    start.parameters["amplitude"] = 100
    start.parameters["positions"][0][:] = 100
    start.covariance.blocks.setflags(write=True)
    start.covariance.A[:] = 100
    accepted = wheel.ledger["starting"]
    assert accepted.parameters["amplitude"] == 2.0
    np.testing.assert_array_equal(accepted.parameters["positions"][0], [1, 2])
    np.testing.assert_array_equal(wheel.contribution("starting")["A"], 2)
    np.testing.assert_array_equal(wheel.covariance.A, 0.25)


@pytest.mark.parametrize("block_domain", [None, "frequency"])
@pytest.mark.parametrize(
    "failure", ["raises", "type", "shape", "nonfinite", "covariance", "copy"]
)
def test_failed_initial_sample_leaves_registration_and_rng_unchanged(
    observed, failure, block_domain
):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("start state cannot be copied")

    class FailingStart(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            rng.normal()  # a discarded draw must never reach the campaign RNG
            good = super().initial_sample(conditional_residual, covariance, rng=rng)
            if failure == "raises":
                raise RuntimeError("the prior is improper here")
            if failure == "type":
                return {"amplitude": 2.0}
            if failure == "shape":
                return replace(good, template={"A": np.zeros(3)})
            if failure == "nonfinite":
                good.template["A"][1] = np.nan
                return good
            if failure == "covariance":
                shorter = replace(
                    observed, channel_data={"A": np.full(32, 5.0)}, num_time_samples=32
                )
                return replace(good, covariance=white_covariance(shorter, 1.0))
            return replace(good, parameters={"amplitude": CannotCopy()})

    # Each kind has its own error, so one failure cannot pass for another.
    expected, message = {
        "raises": (RuntimeError, "the prior is improper here"),
        "type": (TypeError, "starting.initial_sample must return a BlockState"),
        "shape": (ValueError, "length 3, expected"),
        "nonfinite": (ValueError, "finite"),
        "covariance": (ValueError, "grid"),
        "copy": (RuntimeError, "start state cannot be copied"),
    }[failure]
    wheel = Wheel(observed, random_seed=7)
    before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises(expected, match=message):
        wheel.add(FailingStart(), data_domain=block_domain)
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._block_domains == {}
    assert wheel._rng.bit_generator.state == before
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
    # The same name can be retried, and the discarded draw never reached the RNG.
    block = StartingBlock()
    wheel.add(block, data_domain=block_domain)
    wheel.run(1)
    assert block.state.draw == np.random.default_rng(7).normal()


@pytest.mark.parametrize("with_noise", [False, True])
def test_start_without_signal_can_sample_in_selected_domain(observed, with_noise):
    class NoiseOrNothing(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            assert conditional_residual.data_domain == "frequency"
            # White noise with variance 0.25 on the block's frequency grid:
            # C = (Tobs / 2) * (2 sigma^2 / fs) = N sigma^2 / fs^2 = 64 * 0.25 / 0.4**2.
            published = (
                Covariance.like(
                    conditional_residual,
                    {ch: np.full(33, 100.0) for ch in conditional_residual.channel_names},
                )
                if with_noise
                else None
            )
            return BlockState(covariance=published, parameters={"variance": 0.25})

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            assert current_block_state.template is None
            assert current_block_state.parameters == {"variance": 0.25}
            if with_noise:
                current_block_state.covariance.check_compatible(conditional_residual)
                covariance.check_compatible(conditional_residual)
                np.testing.assert_allclose(covariance.A, 100.0)
            self.state.counter += 1
            return current_block_state

    wheel = Wheel(observed)
    block = NoiseOrNothing()
    wheel.add(block, data_domain="frequency")
    wheel.run(1)
    assert block.state.counter == 1
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
    if with_noise:
        np.testing.assert_allclose(wheel.covariance.A, 100.0)
        assert wheel.covariance.grid == observed.to_frequency().grid


def test_noise_only_start_establishes_covariance_owner(observed):
    class DropsNoise(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            return BlockState(
                covariance=white_covariance(conditional_residual, 0.25),
                parameters={"variance": 0.25},
            )

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return BlockState(parameters={"variance": 0.25})

    wheel = Wheel(observed)
    wheel.add(DropsNoise())
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
    np.testing.assert_array_equal(wheel.covariance.A, 0.25)
    with pytest.raises(ValueError, match="owns the current covariance"):
        wheel.run(1)
    assert wheel.ledger["starting"].covariance is not None


def test_start_noise_takeover_warning_can_fail_atomically(observed):
    wheel = Wheel(observed, random_seed=3)
    first, second = noise_start(1.0), noise_start(2.0)
    first.name, second.name = "first", "second"
    wheel.add(first)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseOverwrittenWarning)
        with pytest.raises(NoiseOverwrittenWarning, match="second.initial_sample"):
            wheel.add(second)
    assert list(wheel.ledger) == ["first"]
    assert "second" not in wheel._block_domains
    np.testing.assert_array_equal(wheel.covariance.A, 1.0)


def test_check_block_can_exercise_a_configured_start(observed):
    block = StartingBlock(amplitude=4.0)
    check_block(block, observed, num_cycles=2)
    assert len(block.state.starts) == 1
    assert block.state.counter == 2
    np.testing.assert_array_equal(block.state.starts[0][0].channel_data["A"], 5.0)


def test_later_blocks_start_from_the_residual_of_earlier_starts(observed):
    wheel = Wheel(observed)
    first, second = StartingBlock(amplitude=2.0), StartingBlock(amplitude=1.0)
    first.name, second.name = "first", "second"
    wheel.add(first)
    wheel.add(second, data_domain="frequency")
    first_residual = first.state.starts[0][0]
    second_residual = second.state.starts[0][0]
    np.testing.assert_array_equal(first_residual.channel_data["A"], 5.0)
    assert second_residual.data_domain == "frequency"
    np.testing.assert_allclose(transform(second_residual, "time").channel_data["A"], 3)
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 2.0)


def test_covariance_on_another_grid_is_refused_at_add(observed):
    wheel = Wheel(observed, white_covariance(observed, 0.25), random_seed=3)
    block = StartingBlock()
    before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises(ValueError, match="does not translate between grids") as info:
        wheel.add(block, data_domain="frequency")
    message = str(info.value)
    assert "'starting'" in message
    assert "data_domain='frequency'" in message and "data_domain='time'" in message
    assert "Register the block on the covariance's grid" in message
    assert block.state.starts == []
    assert len(wheel.ledger) == 0
    assert wheel._rng.bit_generator.state == before
    assert wheel.covariance.grid == observed.grid


def test_start_that_publishes_a_covariance_is_handed_to_the_next_block(observed):
    wheel = Wheel(observed)
    noise, signal = noise_start(0.5), StartingBlock()
    noise.name = "noise"
    wheel.add(noise)
    wheel.add(signal)
    _, covariance, _ = signal.state.starts[0]
    np.testing.assert_array_equal(covariance.A, 0.5)
    assert covariance.grid == observed.grid
```

- [ ] **Step 2: Run the new file to see it fail on the old API**

Run: `.venv/bin/python -m pytest -q tests/test_initial_sample.py 2>&1 | tail -3`
Expected: errors such as `TypeError: Wheel.add() missing 1 required keyword-only argument: 'initial_block_state'` and the "does not implement initial_sample" match failing.

- [ ] **Step 3: The protocol, `src/enchilada/block.py`**

Replace the class docstring's second paragraph (the four lines from "Keep sampler continuation and diagnostics" through "returns a complete new state.") with:

```python
    """A model that publishes physical estimates while owning its sampler state.

    Keep sampler continuation and diagnostics in a block-owned State object.
    BlockState contains only template, covariance, and physical parameters.
    `Wheel.add` calls `initial_sample` once with the block's starting inputs and
    adopts the returned state; `sample` then advances the latest accepted
    `current_block_state` and returns a complete new state. Both receive the
    conditional residual on the block's registered grid, the covariance when
    the campaign has one, and the campaign RNG. A block that starts at a known
    point takes that point in its constructor and returns it from
    `initial_sample`; a block that starts from its prior draws from `rng`.
```

Change the later sentence "Each call receives independent snapshots of the conditional residual, covariance, and current state." to "Each call, `initial_sample` included, receives independent snapshots of the conditional residual, covariance, and current state." Add the method before `sample`:

```python
    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw or construct this block's starting physical state on the input grid."""
        ...
```

- [ ] **Step 4: Registration, `src/enchilada/wheel.py`**

In the class docstring replace the two sentences "Registration requires and adopts an explicit initial_block_state. It never calls model initialization or consumes random draws." with:

```
    Registration calls the block's `initial_sample` with the residual of the
    blocks registered so far on the block's grid, the covariance when the
    campaign has one, and the campaign RNG, then adopts the returned state
    like a `sample` return.
```

and, in the last paragraph, "a block on another grid is refused at run with ValueError naming both grids" with "a block on another grid is refused at add or run with ValueError naming both grids".

Replace `add` entirely:

```python
    def add(
        self,
        block_to_register: Block,
        *,
        data_domain: str | None = None,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
    ) -> None:
        """Register a block and adopt the start its initial_sample returns.

        block_to_register must have a unique name and implement initial_sample
        and sample. Failed registration leaves the campaign unchanged: an
        exception from initial_sample, a return that is not a BlockState, an
        off-grid or non-finite template, or a covariance on another grid than
        the block's leaves the ledger, the block list, the residual and the
        Wheel RNG as they were.

        initial_sample receives the current working residual (the observations
        minus the adopted starts of every block registered so far) on this
        block's grid, the current covariance (None until a block supplies one;
        a covariance on another grid than this block's is refused here with
        ValueError) and a copy of the campaign RNG that is committed with the
        state. Blocks therefore start in registration order: register a noise
        block before the blocks whose starts need its covariance. The returned
        state is validated and adopted exactly like a sample return: its
        template must be on this block's grid, and Wheel translates it to the
        observation grid for storage. A covariance in the start makes this
        block the covariance owner. A block that must start at a chosen point
        takes that point in its constructor and returns it from initial_sample.

        data_domain selects time, frequency or wdm for this block's residual,
        current state and published covariance. None keeps the observation grid.
        For WDM, specify either division count; the other follows from the
        observation length. Both counts must be even. An existing observation
        WDM grid is reused if neither count is supplied.
        """
        observed = self._observed_data
        if data_domain is None:
            if num_frequency_divisions is not None or num_time_divisions is not None:
                raise ValueError("WDM division counts require data_domain='wdm'")
            block_grid = observed.grid
        else:
            block_grid = Grid(
                data_domain,
                observed.num_time_samples,
                observed.sample_rate_hz,
                _target_grid(
                    data_domain,
                    observed.num_time_samples,
                    observed.wdm_grid,
                    num_frequency_divisions,
                    num_time_divisions,
                ),
            )
        block_name = getattr(block_to_register, "name", None)
        if not isinstance(block_name, str) or not block_name:
            raise ValueError(
                f"block name must be a non-empty string, got {block_name!r}"
            )
        if block_name in self._ledger:
            raise ValueError(f"block name {block_name!r} already registered")
        if not callable(getattr(block_to_register, "sample", None)):
            raise TypeError(
                f"block {block_name!r} does not implement sample; "
                "a Block needs name, initial_sample(conditional_residual, "
                "covariance, *, rng) and sample(conditional_residual, "
                "covariance, current_block_state, *, rng)."
            )
        if not callable(getattr(block_to_register, "initial_sample", None)):
            raise TypeError(
                f"block {block_name!r} does not implement initial_sample; "
                "a Block needs name, initial_sample(conditional_residual, "
                "covariance, *, rng) and sample(conditional_residual, "
                "covariance, current_block_state, *, rng)."
            )
        conditional_residual, covariance = self._block_inputs(
            block_name, block_grid, registered=False
        )
        candidate_rng = deepcopy(self._rng)
        new_block_state: object = block_to_register.initial_sample(
            conditional_residual=conditional_residual,
            covariance=covariance,
            rng=candidate_rng,
        )
        if block_grid != observed.grid:
            new_block_state = self._canonical_state(
                new_block_state, conditional_residual
            )
        self._adopt(
            block_name,
            new_block_state,
            "initial_sample",
            candidate_rng,
            block_to_register=block_to_register,
            block_grid=block_grid,
        )
```

Replace `_block_inputs` with the grid-taking form (the error message is unchanged):

```python
    def _block_inputs(
        self, block_name: str, block_grid: Grid, *, registered: bool
    ) -> tuple[L1Data, Covariance | None]:
        """Conditional residual on the block's grid, and the covariance when on it.

        A registered block's residual leaves its own signal in; a block being
        registered has no signal yet, so it receives the working residual.
        """
        if self._covariance is not None and self._covariance.grid != block_grid:
            raise ValueError(
                f"block {block_name!r} is registered on grid {block_grid} but the "
                f"current covariance is on grid {self._covariance.grid}; a "
                "covariance does not translate between grids. Register the block "
                "on the covariance's grid, or have the noise block publish on the "
                "block's grid."
            )
        conditional_residual = (
            self.residual(exclude_block_name=block_name)
            if registered
            else self.working_residual
        )
        if block_grid != self._observed_data.grid:
            conditional_residual = transform(
                conditional_residual,
                block_grid.data_domain,
                num_frequency_divisions=self._frequency_divisions(block_grid),
            )
        covariance = None if self._covariance is None else deepcopy(self._covariance)
        return conditional_residual, covariance
```

In `run`, the line `conditional_residual, covariance = self._block_inputs(block_name)` becomes:

```python
                conditional_residual, covariance = self._block_inputs(
                    block_name, block_grid, registered=True
                )
```

(`block_grid` is already bound on the line above it.) Nothing else in `run` or `_adopt` changes; `_adopt`'s docstring sentence "block_to_register is supplied only when adopting a new block's initial state." stays true.

- [ ] **Step 5: `src/enchilada/testing.py`**

`EchoBlock` docstring becomes:

```python
    """Zero-signal diagnostic block that prints residual RMS and advances a counter.

    Register with ``wheel.add(block)``: ``initial_sample`` publishes an empty
    state, and every ``sample`` call does the same after printing. Inspect
    ``block.state.num_sample_calls`` for the counter; diagnostics belong to
    ``block.state.metadata``. Use a separate instance per campaign.
    """
```

Add, between `__init__` and `sample`:

```python
    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Publish an empty physical state; the counter and the RNG are untouched."""
        return BlockState()
```

`check_block` becomes:

```python
def check_block(
    block_under_test: Block,
    observed_data: L1Data,
    num_cycles: int = 2,
    *,
    initial_covariance: Covariance | None = None,
    random_seed: int | None = 0,
) -> None:
    """Run a block on a scratch Wheel and validate each returned BlockState.

    Registers the block, which calls its ``initial_sample`` on the observation
    grid, then calls ``sample`` for ``num_cycles``, passing accepted state into
    each call. Wheel validates the start and every sample alike: the signal
    grid, finiteness and any published Covariance.

    Args:
        block_under_test: Configured block whose exchange contract is checked.
            A block that must start at a chosen point is constructed with it.
        observed_data: L1 observations and shared run settings for the scratch run.
        num_cycles: Number of sampling cycles after registration; defaults to 2.
        initial_covariance: Starting Covariance for the scratch Wheel. None
            leaves it unset until a block supplies it.
        random_seed: Seed for the scratch Wheel's NumPy generator, which also
            drives ``initial_sample``. Defaults to 0 for reproducibility; None
            requests nondeterministic initialization.

    This advances the supplied block's local state. It checks the exchange
    contract, not statistical correctness or block-local failure recovery.
    Test posterior recovery and independent campaigns separately;
    ``examples/toy_fit.py`` provides a pattern. ``check_block`` does not select
    a separate block domain: ``initial_sample`` and ``sample`` receive
    ``observed_data``'s grid.
    """
    from enchilada.wheel import Wheel

    wheel = Wheel(
        observed_data=observed_data,
        initial_covariance=initial_covariance,
        random_seed=random_seed,
    )
    wheel.add(block_to_register=block_under_test)
    wheel.run(num_cycles=num_cycles)
```

- [ ] **Step 6: `tests/conftest.py`: the `starting_at` shorthand**

Add after `const_block_state`:

```python
def starting_at(block, state):
    """Give `block` an `initial_sample` that returns `state` as it is.

    Test shorthand for a block whose start is a constant built by the test. The
    block keeps its identity, name and local state; `state` must be on the grid
    the block registers on, as a real `initial_sample` return would be.
    """

    def initial_sample(conditional_residual, covariance, *, rng):
        return state

    # object.__setattr__ also works on a frozen dataclass instance.
    object.__setattr__(block, "initial_sample", initial_sample)
    return block
```

- [ ] **Step 7: `tests/test_testing.py`**

Drop every `initial_block_state=BlockState()` argument (`check_block(EchoBlock(name="echo"), obs)` and so on; the `Bad`, `Drifter`, `FlatNoiseBlock` and `BadNoise` subclasses inherit `initial_sample` from `EchoBlock`). Replace `test_initial_block_state_is_required` and `test_initial_block_state_must_be_a_block_state` with:

```python
    def test_block_without_initial_sample_is_refused(self, observed):
        class SampleOnly:
            name = "sample_only"

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return current_block_state

        with pytest.raises(TypeError, match="does not implement initial_sample"):
            check_block(SampleOnly(), observed)

    def test_failing_initial_sample_is_caught(self, observed):
        class BadStart(EchoBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return {"A": np.zeros(1)}  # not a BlockState

        with pytest.raises(TypeError, match="initial_sample must return a BlockState"):
            check_block(BadStart(name="bad"), observed)
```

In `TestEchoBlock`, `wheel.add(echo)` and `wheel.add(EchoBlock(name="echo"))`. In `test_echo_continuation_belongs_to_each_block_instance`, replace the final `assert not hasattr(echo, "draw_prior")` with:

```python
    start = echo.initial_sample(observed, None, rng=rng)
    assert start.template is None
    assert start.covariance is None
    assert start.parameters == {}
    assert echo.state.num_sample_calls == 2
```

If `BlockState` is no longer used in the file after these edits, drop it from the import.

- [ ] **Step 8: Gates for this task**

Run: `.venv/bin/python -m pytest -q tests/test_initial_sample.py tests/test_testing.py`
Expected: all passed (the new file is about 30 cases counting parametrizations), 0 failed.

Run: `.venv/bin/python -m ruff format src/enchilada/block.py src/enchilada/wheel.py src/enchilada/testing.py tests/conftest.py tests/test_initial_sample.py tests/test_testing.py && .venv/bin/python -m ruff check src tests/conftest.py tests/test_initial_sample.py tests/test_testing.py && .venv/bin/python -m mypy`
Expected: `All checks passed!`, `Success: no issues found in 15 source files`.

- [ ] **Step 9: Commit**

```bash
git add src/enchilada/block.py src/enchilada/wheel.py src/enchilada/testing.py tests/conftest.py tests/test_initial_sample.py tests/test_testing.py
git status --short   # the rename shows as R tests/test_initial_block_state.py -> tests/test_initial_sample.py
git commit -m "Blocks draw their own start: Block.initial_sample replaces Wheel.add's initial_block_state

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

The rest of the suite is red at this commit by construction (every other registration still passes the removed keyword); Tasks 2 to 5 restore it.

---

### Task 2: Wheel test migration

**Goal:** `tests/test_wheel.py` and `tests/test_wheel_translation.py` register every block through `initial_sample`, the tests that pinned "registration never calls the block" are inverted, and the two files pass again.

**Files:**
- Modify: `tests/test_wheel.py` (71 registration sites; `ConstBlock`, `LevelNoiseBlock`, `_noise_block`; `TestAtomicRegistration`; `TestBoundaryGuards`; `TestNoiseOwnership`; `TestDocumentedContracts`)
- Modify: `tests/test_wheel_translation.py` (17 sites; `ConstantSignal`; the five seed/grid tests)
- Read only: `tests/conftest.py` (`starting_at`, `const_block_state`, `white_covariance`), `src/enchilada/wheel.py` (Task 1's `add`)

**Acceptance Criteria:**
- [ ] No `initial_block_state` remains in either file (`grep -c initial_block_state tests/test_wheel.py tests/test_wheel_translation.py` prints `0` twice).
- [ ] `ConstBlock`, `LevelNoiseBlock`, `_noise_block`'s `NoiseWriter` and `ConstantSignal` define `initial_sample`; every other block either defines one or is wrapped with `starting_at(block, state)`.
- [ ] `TestAtomicRegistration` asserts that registration calls `initial_sample` exactly once and nothing else, that a start which is not a `BlockState` is refused with `bad.initial_sample must return a BlockState`, and the noise-takeover message names `b.initial_sample`.
- [ ] The translation tests assert: a block on another grid than the covariance is refused at `add` before `initial_sample` (message naming the block and both grids, RNG unchanged, nothing registered); a WDM block on a time covariance is refused at `add` before any WDM transform; a start on the wrong WDM grid is refused with `ValueError` matching `shape`; a mutated start array is refused with `TypeError` matching `numpy array`; a noise-only start on the block grid is held there; a noise-only start on another grid is refused with `noise.initial_sample covariance is on grid`.
- [ ] `tests/test_wheel.py`, `tests/test_wheel_translation.py`, `tests/test_initial_sample.py` and `tests/test_testing.py` pass; ruff is clean on the two files.

**Verify:** `.venv/bin/python -m pytest -q tests/test_wheel.py tests/test_wheel_translation.py tests/test_initial_sample.py tests/test_testing.py` → all passed, 0 failed.

**Steps:**

- [ ] **Step 1: Shared blocks in `tests/test_wheel.py`**

Import `starting_at` from `conftest`. Give the shared classes their starts:

```python
class ConstBlock:
    ...  # docstring and State unchanged

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return replace(
            self._block_state(conditional_residual), parameters={"value": self.value}
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        ...  # unchanged

    def _block_state(self, residual):
        return const_block_state(residual, self.value)


class LevelNoiseBlock:
    ...

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.block_state().with_covariance(
            white_covariance(conditional_residual, 1.0)
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        ...  # unchanged
```

`NarrowSignal(ConstBlock)` overrides `_block_state`, so its inherited `initial_sample` produces the float32 start the old `constant_initial_result(..., dtype=np.float32)` built. In `TestNoiseOwnership._noise_block`, `NoiseWriter` gains:

```python
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return conditional_residual.block_state().with_covariance(
                    white_covariance(conditional_residual, level)
                )
```

- [ ] **Step 2: The per-site rule for `tests/test_wheel.py`**

Work through `grep -n "initial_block_state" tests/test_wheel.py` top to bottom:

1. `wheel.add(X, initial_block_state=constant_initial_result(obs, v))` where `X` is a `ConstBlock` (or subclass) constructed with the same `v` → `wheel.add(X)`. Every `ConstBlock` site in the file has matching values (`"a", 1.0`; `"rj", 1.0`; `"positive", 1e16`; `Changing(str(index), float(index + 1))`; `"overflow_guard", np.finfo(float).max`; and so on).
2. `wheel.add(EchoBlock(...), initial_block_state=BlockState())` → `wheel.add(EchoBlock(...))`.
3. `wheel.add(LevelNoiseBlock(...), initial_block_state=noise_initial_result(observed))` and `wheel.add(self._noise_block(n, level), initial_block_state=noise_initial_result(observed, level))` → drop the keyword (the methods above publish the same level).
4. A test-local class registered with `BlockState()` or another constant (`InPlace`, `Mutator`, `Recorder` if it is not a `ConstBlock`, `PartlyZero`, `PublishesUnchecked`, `PublishesCompatible`, ...) → `wheel.add(starting_at(Local(...), <the same state expression>))`. Where the class is the subject of the test, adding a two-line `initial_sample` method that returns that state is equally acceptable; use whichever reads better.
5. Delete `constant_initial_result` and `noise_initial_result` once nothing uses them (ruff flags unused helpers only if imported; delete them anyway so the file does not keep dead seed builders).

Before/after example (line 105):

```python
        wheel.add(
            ConstBlock("a", 1.0), initial_block_state=constant_initial_result(obs, 1.0)
        )
```
becomes
```python
        wheel.add(ConstBlock("a", 1.0))
```

- [ ] **Step 3: The inverted tests in `tests/test_wheel.py`**

`TestAtomicRegistration.test_registration_adopts_state_without_invoking_block` becomes:

```python
    def test_registration_calls_initial_sample_and_nothing_else(self, observed):
        calls = []

        class Recorder(ConstBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                calls.append("initial_sample")
                return super().initial_sample(conditional_residual, covariance, rng=rng)

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                calls.append("sample")
                return super().sample(
                    conditional_residual,
                    covariance,
                    current_block_state,
                    rng=rng,
                )

        wheel = Wheel(observed)
        block = Recorder("prior", 2.0)
        wheel.add(block)
        assert calls == ["initial_sample"]
        assert block.state.updates == 0
        np.testing.assert_array_equal(wheel.contribution("prior")["A"], 2.0)
        wheel.run(2)
        assert calls == ["initial_sample", "sample", "sample"]
```

`test_invalid_initial_result_leaves_wheel_untouched` becomes `test_invalid_start_leaves_wheel_untouched`:

```python
    def test_invalid_start_leaves_wheel_untouched(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("ok", 1.0))
        with pytest.raises(TypeError, match="bad.initial_sample must return a BlockState"):
            wheel.add(starting_at(ConstBlock("bad", 0.0), "not a result"))
        with pytest.raises(ValueError, match="unknown block"):
            wheel.contribution("bad")  # not registered
        wheel.contribution("ok")  # the good one still is
        wheel.run(1)  # still healthy
```

`TestBoundaryGuards.test_block_with_only_name_and_sample_can_register` becomes `test_block_with_name_initial_sample_and_sample_can_register`, its local class renamed `Minimal` with

```python
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return BlockState()
```

added above `sample`, registered as `wheel.add(Minimal())`; `test_missing_sample_is_caught_at_registration` stays as it is (the `sample` check still comes first) with `Wheel(observed).add(NoSample())`.

`TestNoiseOwnership.test_the_message_names_both_blocks_and_the_fix`: `assert "b.initial_sample" in msg and "'a'" in msg`.

`TestDocumentedContracts.test_explicit_initialization_can_read_current_residual` is replaced by:

```python
    def test_initial_sample_reads_the_current_residual(self, rng):
        """A new block starts from the observations minus the blocks registered before it."""
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstBlock("a", 1.0))
        seen = {}

        class Reader(ConstBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                seen["residual"] = conditional_residual
                return super().initial_sample(conditional_residual, covariance, rng=rng)

        wheel.add(Reader("b", 2.0))
        np.testing.assert_allclose(
            seen["residual"].channel_data["A"], obs.channel_data["A"] - 1.0
        )
        np.testing.assert_allclose(
            wheel.residual().channel_data["A"], obs.channel_data["A"] - 3.0
        )
```

- [ ] **Step 4: `tests/test_wheel_translation.py`**

Import `starting_at` from `conftest`. `ConstantSignal` gains, above `sample`:

```python
    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self._render(conditional_residual)
```

(`_render` already asserts the residual is on the expected domain and renders the amplitude there.) The loop at line 104 becomes `wheel.add(block, data_domain=domain, **options)` and `block_data` is no longer needed.

Sites with a constant seed (`PixelNoise` with `initial`; `RandomSignal`, `TimeSignal`, `FrequencyNoise`, `TimeNoiseFromFrequency` with `BlockState(parameters=...)`; the three `EchoBlock("noise")` seed tests) use `starting_at(Block(...), <same state>)`. `NeverCalled` gains an `initial_sample` that raises `AssertionError` (registration must fail on the domain before reaching it) and is registered as `wheel.add(NeverCalled(), data_domain="unknown")` / `wheel.add(NeverCalled(), num_time_divisions=8)`.

Rewrite the five existing tests whose premise changes (`test_initial_wdm_signal_must_match_selected_grid_before_the_transform`, `test_explicit_domain_revalidates_mutated_initial_signal_arrays`, `test_block_on_another_grid_than_the_covariance_is_refused_at_run`, `test_wdm_block_on_a_time_covariance_is_refused_before_the_transform`, `test_noise_only_seed_on_another_grid_than_the_block_is_refused_at_registration`), in place, as:

```python
def test_a_start_on_the_wrong_wdm_grid_is_refused():
    # The old test also monkeypatched _wdm_transform to prove the bad template
    # never reached domania. add now legitimately transforms the residual to
    # WDM before initial_sample runs, so that guard would fire for the wrong
    # reason; the wrong template is still caught by the grid check inside
    # transform, before any WDM conversion of the template itself.
    data = observations()
    wheel = Wheel(data)
    # Both grids represent 64 samples, but the selected (Nf, Nt) = (4, 16)
    # requires shape (4, 16), not the (8, 8) coefficients of an (8, 8) grid.
    wrong = BlockState(
        template={name: np.zeros((8, 8)) for name in data.channel_names}
    )
    with pytest.raises(ValueError, match="shape"):
        wheel.add(
            starting_at(ConstantSignal("wrong_grid", 1.0, "wdm"), wrong),
            data_domain="wdm",
            num_frequency_divisions=4,
        )
    assert len(wheel.ledger) == 0
    for name in data.channel_names:
        np.testing.assert_array_equal(
            wheel.residual().channel_data[name], data.channel_data[name]
        )


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_explicit_domain_revalidates_mutated_start_arrays(domain):
    data = observations()
    wheel = Wheel(data)
    start = transform(data, domain).zero_block_state()
    start.template["A"] = start.template["A"].tolist()
    with pytest.raises(TypeError, match="numpy array"):
        wheel.add(
            starting_at(ConstantSignal("invalid_array", 1.0, domain), start),
            data_domain=domain,
        )
    assert len(wheel.ledger) == 0


def test_block_on_another_grid_than_the_covariance_is_refused_at_add():
    """The Wheel does not translate a covariance: a block registered on another
    grid than the current covariance is refused before initial_sample is called."""

    class NeverSampled:
        name = "frequency_block"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            raise AssertionError("the Wheel must refuse before initial_sample is called")

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError("the Wheel must refuse before sample is called")

    data = observations()
    wheel = Wheel(data, white_covariance(data, 0.25), random_seed=3)
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(ValueError, match="does not translate between grids") as info:
        wheel.add(NeverSampled(), data_domain="frequency")
    message = str(info.value)
    assert "'frequency_block'" in message
    assert "data_domain='frequency'" in message and "data_domain='time'" in message
    assert "Register the block on the covariance's grid" in message
    assert wheel._rng.bit_generator.state == rng_before
    assert len(wheel.ledger) == 0
    assert wheel.covariance.grid == data.grid
    np.testing.assert_array_equal(wheel.covariance.A, 0.25)


def test_wdm_block_on_a_time_covariance_is_refused_before_the_transform(monkeypatch):
    # Formerly refused at run; the covariance-grid check in _block_inputs now
    # runs at add, before the residual is transformed and before initial_sample.
    def unexpected_transform(*args, **kwargs):
        raise AssertionError("the grid check must come before any WDM transform")

    monkeypatch.setattr("enchilada._transforms._wdm_transform", unexpected_transform)
    data = observations()
    wheel = Wheel(data, white_covariance(data, 1.0))
    with pytest.raises(ValueError, match="does not translate between grids"):
        wheel.add(EchoBlock("wdm_block"), data_domain="wdm", num_frequency_divisions=8)
    assert len(wheel.ledger) == 0
    assert wheel.covariance.grid == data.grid


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_start_on_another_grid_than_the_block_is_refused_at_registration(
    domain,
):
    data = observations()
    _, options = block_reference(data, domain)
    start = BlockState(covariance=white_covariance(data, 1.0))  # on the time grid
    wheel = Wheel(data)
    with pytest.raises(ValueError, match="noise.initial_sample covariance is on grid"):
        wheel.add(starting_at(EchoBlock("noise"), start), data_domain=domain, **options)
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None
```

`test_noise_only_seed_on_the_block_grid_is_adopted_there` and `test_seed_covariance_with_reordered_channel_names_is_refused_at_registration` keep their bodies with `starting_at(EchoBlock("noise"), seed)` and are renamed `test_noise_only_start_on_the_block_grid_is_adopted_there` and `test_start_covariance_with_reordered_channel_names_is_refused_at_registration`. `test_default_block_is_refused_once_a_frequency_noise_owner_has_published` keeps its sequence (the frequency noise block still publishes its covariance in `sample`, so the time block is refused at the second `run`).

- [ ] **Step 5: Gates and commit**

Run: `grep -c initial_block_state tests/test_wheel.py tests/test_wheel_translation.py` → `0` and `0`.
Run: `.venv/bin/python -m ruff format tests/test_wheel.py tests/test_wheel_translation.py && .venv/bin/python -m ruff check tests/test_wheel.py tests/test_wheel_translation.py && .venv/bin/python -m pytest -q tests/test_wheel.py tests/test_wheel_translation.py tests/test_initial_sample.py tests/test_testing.py`
Expected: `All checks passed!`; all tests passed.

```bash
git add tests/test_wheel.py tests/test_wheel_translation.py
git commit -m "Tests: the Wheel suites register through initial_sample

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Remaining test migration and the typing fixtures

**Goal:** The six smaller test files register through `initial_sample`, and the mypy consumer fixture shows that a class without `initial_sample` is not a `Block` while conforming blocks still type-check.

**Files:**
- Modify: `tests/test_state.py`, `tests/test_block_state.py`, `tests/test_block_state_contract.py`, `tests/test_ledger.py`, `tests/test_translation_review.py`, `tests/test_typing.py`
- Read only: `tests/conftest.py` (`starting_at`)

**Acceptance Criteria:**
- [ ] `grep -c initial_block_state` prints `0` for each of the six files.
- [ ] `Counter`, `NoiseOnly`, `StatefulSignal`, `AmplitudeUpdate` and `RecordInputs` define `initial_sample`; the other local blocks define one or use `starting_at`.
- [ ] `tests/test_typing.py::test_consumer_typos_are_static_errors` asserts three mypy errors: the two attribute typos and `Incompatible types in assignment (expression has type "SampleOnly", variable has type "Block")`; every conforming fixture block defines `initial_sample`.
- [ ] `test_block_protocol_is_runtime_checkable` also asserts that a sample-only class is not a `Block`; `test_frozen_dataclass_can_explicitly_inherit_block` registers `ExplicitBlock` whose `initial_sample` returns `BlockState(parameters={"value": 1})`.
- [ ] The six files pass; ruff is clean on them.

**Verify:** `.venv/bin/python -m pytest -q tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_typing.py` → all passed, 0 failed.

**Steps:**

- [ ] **Step 1: `tests/test_state.py`**

```python
class Counter:
    name = "counter"

    @dataclass
    class State:
        n: int = 0

    def __init__(self):
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.zero_block_state(parameters={"a": 0.0})

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        ...  # unchanged
```

Sites: `wheel.add(block)` / `actual.add(block); expected.add(Counter())` / `one.add(first); two.add(second)` wherever the seed was `counter_initial_result(observed)` (the method returns the same state). Two tests keep a hand-built start: `test_block_state_parameters_are_copied_on_adoption_and_read` (it mutates `initial` after registration) uses `wheel.add(starting_at(Counter(), initial))`, and the non-finite test at the end uses

```python
    with pytest.raises(ValueError, match="non-finite"):
        wheel.add(
            starting_at(
                Counter(),
                data.block_state({ch: -x for ch, x in data.channel_data.items()}),
            )
        )
```

Delete `counter_initial_result` when nothing uses it; import `starting_at` from `conftest`.

- [ ] **Step 2: `tests/test_block_state.py`**

`NoiseOnly` gains:

```python
    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.block_state(
            covariance=white_covariance(conditional_residual, 1),
            parameters={"variance": 1},
        )
```

The `Signal` block in the rfft-endpoint test is registered as `wheel.add(starting_at(Signal(), observed.block_state(template=signal)))`. The two paired tests that seed both an `InvalidOnce`/`FailsOnce` block and a plain `NoiseOnly` with `noise_initial_result(observed, signal_value=0.5)` use `starting_at(actual_block, initial)` and `starting_at(expected_block, initial)`. `Population` gains `initial_sample` returning `population_initial_result(conditional_residual)`; `wheel.add(NoiseOnly())` and `wheel.add(population)`. Keep `noise_initial_result` only if a `starting_at` site still calls it.

- [ ] **Step 3: `tests/test_block_state_contract.py`, `tests/test_ledger.py`, `tests/test_translation_review.py`**

`StatefulSignal` gains `initial_sample` returning `conditional_residual.zero_block_state(parameters={"amplitude": 0.0})`; `wheel.add(block)` and the local `initial` goes. `AmplitudeUpdate` gains `initial_sample` returning `BlockState(parameters={"amplitudes": np.array([1.0])})`; both registrations become `wheel.add(AmplitudeUpdate())`. In `test_translation_review.py`, `RecordInputs` gains

```python
        def initial_sample(self, conditional_residual, covariance, *, rng):
            covariance.check_compatible(conditional_residual)
            return BlockState()
```

(so `received == []` after both registrations still holds, and registration is shown to deliver the covariance on the block's grid); `RecordCovariance` is registered as `wheel.add(starting_at(RecordCovariance(), BlockState()), data_domain="time")`; `MutateInputs` as `wheel.add(starting_at(MutateInputs(), BlockState(parameters={"amplitude": 0.0, "coordinates": np.array([3.0])})), data_domain="frequency")` for both wheels (adoption deep-copies, so one shared state object is fine).

- [ ] **Step 4: `tests/test_typing.py`**

In `CONSUMER`, replace the registration lines and the three block classes:

```python
wheel = Wheel(r, noise)
wheel.add(EchoBlock("wdm"), data_domain="wdm", num_time_divisions=2)
wheel.add(EchoBlock("seeded"))
check_block(EchoBlock("checked"), r)

class SampleOnly:
    name = "sample_only"

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

sample_only: Block = SampleOnly()   # no initial_sample: not a Block

@dataclass(frozen=True)
class FrozenBlock:
    name: str = "frozen"

    def initial_sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None, *, rng: np.random.Generator,
    ) -> BlockState:
        return BlockState()

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

frozen: Block = FrozenBlock()
wheel.add(FrozenBlock())
check_block(FrozenBlock(), r)

@dataclass(frozen=True)
class ExplicitBlock(Block):
    name: str

    def initial_sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None, *, rng: np.random.Generator,
    ) -> BlockState:
        return BlockState()

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

explicit: Block = ExplicitBlock("explicit")
wheel.add(explicit)
```

(`initial = BlockState()` and the `wheel.add(sample_only, ...)` / `check_block(sample_only, ...)` lines go.) The assertions in `test_consumer_typos_are_static_errors` become:

```python
    out = proc.stdout + proc.stderr
    assert 'has no attribute "Tobbs"' in out, out
    assert 'has no attribute "T_obs"' in out, out
    # exactly the two typos -- the correct `r.Tobs` on the preceding line must
    # not be flagged (an over-broad ban would break legitimate access)
    assert out.count("[attr-defined]") == 2, out
    # ... and a class without initial_sample is not a Block
    assert (
        'Incompatible types in assignment (expression has type "SampleOnly", '
        'variable has type "Block")'
    ) in out, out
    assert out.count("error:") == 3, out
```

`test_block_protocol_is_runtime_checkable` gains:

```python
    class SampleOnly:
        name = "s"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    assert not isinstance(SampleOnly(), Block)
```

`test_frozen_dataclass_can_explicitly_inherit_block`: `ExplicitBlock` gains

```python
        def initial_sample(self, conditional_residual, covariance, *, rng):
            return BlockState(parameters={"value": 1})
```

and registers with `wheel.add(block)`; the ledger assertion is unchanged.

- [ ] **Step 5: Gates and commit**

Run: `for f in tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_typing.py; do grep -c initial_block_state $f; done` → six zeros.
Run: `.venv/bin/python -m ruff format tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_typing.py && .venv/bin/python -m ruff check tests && .venv/bin/python -m pytest -q tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_typing.py`
Expected: `All checks passed!` (ruff on `tests` may still report nothing: Tasks 2 and 4 own the other files); all tests passed, including the mypy subprocess test (it takes about 10 s).

```bash
git add tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_typing.py
git commit -m "Tests: state, ledger, translation-review and typing fixtures register through initial_sample

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Examples, benchmark and their tests

**Goal:** The example blocks (`SineBlock`, `WhiteNoiseBlock`, `GBBlock`, `DomainEcho`, the benchmark's `FixedSignal`) draw or render their own starts in `initial_sample`; the caller-side prior helpers are gone; the example scripts and benchmark register with the new `add`; their tests pass.

**Files:**
- Modify: `examples/toy_fit.py` (docstring, both blocks, delete `draw_sine_prior` and `draw_white_noise_prior`, `run_toy_fit`)
- Modify: `examples/gb_model.py:119-140` (docstring), `:305-308` (error text), `:334-360` (`draw_gb_prior` becomes `GBBlock.initial_sample`)
- Modify: `examples/domain_translation.py` (`DomainEcho.initial_sample`, the loop)
- Modify: `examples/demo.py:14,39-40`, `examples/mojito_wdm_wheel.py:20,84`
- Modify: `benchmarks/benchmark_wheel.py`
- Modify: `tests/test_examples.py`, `tests/test_toy_translation.py`, `tests/test_gb_example_contract.py`
- Not touched (Task 5): the notebooks; `tests/test_translation_example.py` and `tests/test_block_tutorial.py` need no edit but their notebook cases pass only after Task 5.

**Acceptance Criteria:**
- [ ] `grep -rn "initial_block_state\|draw_sine_prior\|draw_white_noise_prior\|draw_gb_prior" examples/*.py benchmarks tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py` prints nothing.
- [ ] `SineBlock.initial_sample` returns a rendered amplitude drawn from `N(0, prior_std²)` with `rng`; `WhiteNoiseBlock.initial_sample` returns `scale / rng.gamma(shape)` rendered as a covariance; both refuse non-time input with the existing "time-domain" `ValueError`s.
- [ ] `GBBlock.initial_sample(conditional_residual, covariance, *, rng)` resets `self.state` with uniform-box walkers and an Eryn `RandomState`, and returns the first walker's rendered state; `sample` on an uninitialised block says `Initialize GBBlock.state with initial_sample or a checkpoint`.
- [ ] `run_toy_fit` seeds one Wheel from a child of `SeedSequence(seed)` and registers the three blocks with `data_domain="time"` and no start; `python examples/toy_fit.py` still prints the three posterior lines and `test_toy_fit_converges_to_truth` passes.
- [ ] `test_toy_prior_is_random_and_separate_campaigns_are_reproducible` builds its three campaigns with `Wheel(observed, random_seed=seed)` and plain `add` calls and keeps its assertions (seeds 42 and 42 agree, 42 and 43 differ, the prior is not 0).
- [ ] `test_toy_time_blocks_preserve_draws_on_translated_observations` still matches the time and translated chains at `rtol=1e-12` with no separate initialization RNG.
- [ ] The GB contract test's `ContextProbe` delegates `initial_sample` to the real block and the registration test passes for both storage domains.
- [ ] `examples/demo.py`, `examples/mojito_wdm_wheel.py` and `benchmarks/benchmark_wheel.py` no longer import `BlockState`; the benchmark's residual assertion (`0.5`) still holds.
- [ ] ruff and mypy clean on the files changed.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_translation_example.py --deselect tests/test_examples.py::test_notebook_code_cells_execute --deselect tests/test_translation_example.py::test_translation_notebook_runs_all_cells_with_domania` → all passed, 0 failed (skips for optional Eryn/h5py are allowed, as today).

**Steps:**

- [ ] **Step 1: `examples/toy_fit.py`**

Module docstring, lines 3-9, becomes:

```python
"""Two conjugate Gibbs source blocks plus sampled white noise.

Blocks retain fixed model configuration. Wheel owns the current parameters,
BlockState snapshots, covariance, and sampling random stream. Each block draws
its own start from its prior in initial_sample, with the RNG Wheel passes at
registration; an on_cycle_complete callback collects posterior chains outside
the blocks. These blocks request time-domain inputs; Wheel also runs them on
Fourier or WDM observations with data_domain="time", because the noise block
publishes its covariance on that time grid. Their likelihood requires
time-domain noise with independent channels.
Run with:

    uv run --no-sync python examples/toy_fit.py
"""
```

Add to `SineBlock`, between `_render` and `sample`:

```python
    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw the amplitude from its prior and render it on the time grid."""
        return self._render(conditional_residual, float(rng.normal(0.0, self.prior_std)))
```

Add to `WhiteNoiseBlock`, between `_render` and `sample`:

```python
    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw the variance from its inverse-gamma prior and publish it."""
        return self._render(
            conditional_residual, float(self.scale / rng.gamma(self.shape))
        )
```

Delete `draw_sine_prior` and `draw_white_noise_prior` (lines 129-140). `run_toy_fit` lines 169-191 become:

```python
    observed = make_observed(seed)
    # A child stream keeps the campaign (prior draws at registration, then
    # sampling) from replaying the observation-noise draws.
    campaign_seed = np.random.SeedSequence(seed).spawn(1)[0]
    wheel = Wheel(observed, random_seed=int(campaign_seed.generate_state(1)[0]))
    for name, frequency in (("slow", 0.004), ("fast", 0.011)):
        wheel.add(SineBlock(name=name, freq=frequency), data_domain="time")
    wheel.add(WhiteNoiseBlock(name="noise"), data_domain="time")
```

(The sine starts need no covariance, so the signal blocks may still register before the noise block; the first `run` cycle finds the noise block's start covariance in place.)

- [ ] **Step 2: `examples/gb_model.py`**

Class docstring: replace "The caller-side ``draw_gb_prior`` helper initializes all walkers before registration." with "``initial_sample`` draws every walker from the uniform box, resets ``self.state`` and publishes the first walker; Wheel calls it once at registration." Replace the last paragraph's first sentence with "Register with ``wheel.add(block, data_domain=\"frequency\")``." The error at line 307 becomes `"Initialize GBBlock.state with initial_sample or a checkpoint"`.

Move `draw_gb_prior` into the class as a method placed before `sample` (delete the module-level function):

```python
    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Initialize block-owned walkers and publish the first walker's signal.

        Wheel calls this at registration with frequency-domain data and a
        frequency-domain covariance on the data grid. It resets ``self.state``;
        a fresh block instance belongs to each independent campaign.
        """
        gb, _ = self._context(conditional_residual, covariance)
        bounds = np.asarray(self.bounds)
        coords = rng.uniform(bounds[:, 0], bounds[:, 1], size=(1, self.nw, 1, 4))
        # Eryn consumes RandomState tuples, not Generator.bit_generator.state.
        random_state = np.random.RandomState(int(rng.integers(2**32))).get_state()
        result = self._render(gb, coords[0, 0, 0], conditional_residual)
        self.state = self.State(
            coords={"model_0": coords},
            inds={"model_0": np.ones((1, self.nw, 1), dtype=bool)},
            random_state=random_state,
        )
        return result
```

- [ ] **Step 3: `examples/domain_translation.py`, `examples/demo.py`, `examples/mojito_wdm_wheel.py`, `benchmarks/benchmark_wheel.py`**

`DomainEcho` gains, above `sample`:

```python
    def initial_sample(self, conditional_residual, covariance, *, rng):
        self._check_input(conditional_residual, covariance)
        return BlockState()
```

and the loop in `run_example` becomes:

```python
    for domain in ("time", "frequency", "wdm"):
        block = DomainEcho(domain, domain)
        blocks[domain] = block
        options = {"num_frequency_divisions": 8} if domain == "wdm" else {}
        wheel.add(block, data_domain=domain, **options)
```

`examples/demo.py`: `from enchilada import L1Data, Wheel`; `wheel.add(ucb)` and `wheel.add(mbhb)`. `examples/mojito_wdm_wheel.py`: `from enchilada import Wheel, decimate, load_mojito_l1, transform`; `wheel.add(EchoBlock(name="echo"))`.

`benchmarks/benchmark_wheel.py`: `from enchilada import Covariance, L1Data, Wheel` and

```python
class FixedSignal:
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.block_state(
            {
                name: np.full_like(values, self.level)
                for name, values in conditional_residual.channel_data.items()
            }
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return current_block_state
```

with the registration loop reduced to:

```python
    for index in range(num_blocks):
        wheel.add(FixedSignal(str(index), 0.5 / num_blocks))
```

- [ ] **Step 4: `tests/test_examples.py`**

`test_toy_prior_is_random_and_separate_campaigns_are_reproducible` lines 77-94 become:

```python
    campaigns = [Wheel(observed, random_seed=seed) for seed in (42, 42, 43)]
    for wheel in campaigns:
        wheel.add(toy.SineBlock(name="slow", freq=0.004))
        wheel.add(toy.WhiteNoiseBlock(name="noise"))
```

(the `priors` assertions and the two `run(3)` comparisons stay). Every `toy.draw_sine_prior(block, data, rng=R)` becomes `block.initial_sample(data, None, rng=R)` (lines 123, 154, 328); `gb.draw_gb_prior(block, data, noise, rng=R)` becomes `block.initial_sample(data, noise, rng=R)` (lines 270, 310). `test_white_noise_prior_requires_time_data` becomes:

```python
def test_white_noise_start_requires_time_data():
    toy = load_example("toy_fit")
    block = toy.WhiteNoiseBlock("noise")
    with pytest.raises(ValueError, match="time-domain"):
        block.initial_sample(
            toy.make_observed().to_frequency(), None, rng=np.random.default_rng(0)
        )
```

- [ ] **Step 5: `tests/test_toy_translation.py`**

Lines 30-48 become:

```python
    for observed in (data, translated_data):
        wheel = Wheel(observed, random_seed=42)
        for name, frequency in (("slow", 0.004), ("fast", 0.011)):
            wheel.add(toy.SineBlock(name, frequency), data_domain="time")
        noise_block = toy.WhiteNoiseBlock("noise")
        wheel.add(noise_block, data_domain="time")
```

The prior draws depend only on the RNG and the time grid, not on the residual values, so the two campaigns still agree to `1e-12`.

- [ ] **Step 6: `tests/test_gb_example_contract.py`**

`gb_model.draw_gb_prior(block, data, noise, rng=R)` becomes `block.initial_sample(data, noise, rng=R)` at lines 63, 83-85, 104-106 (`make_block(gb_model).initial_sample(data, time_noise, rng=np.random.default_rng(0))` and the `noise` variant). The registration test's probe becomes:

```python
    block = make_block(gb_model)

    class ContextProbe:
        """Check the GB likelihood inputs without requiring optional Eryn."""

        name = block.name

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return block.initial_sample(conditional_residual, covariance, rng=rng)

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            block._context(conditional_residual, covariance)
            return current_block_state

    wheel.add(ContextProbe(), data_domain="frequency")
```

(`initial` is no longer built outside.) Rename `test_gb_prior_keeps_continuation_on_its_block` to `test_gb_start_keeps_continuation_on_its_block`.

- [ ] **Step 7: Gates and commit**

Run: `.venv/bin/python -m ruff format examples/toy_fit.py examples/gb_model.py examples/domain_translation.py examples/demo.py examples/mojito_wdm_wheel.py benchmarks/benchmark_wheel.py tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py && .venv/bin/python -m ruff check examples benchmarks tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py && .venv/bin/python -m mypy`
Expected: `All checks passed!`; `Success: no issues found`.

Run the **Verify** command. Then:

```bash
git add examples/toy_fit.py examples/gb_model.py examples/domain_translation.py examples/demo.py examples/mojito_wdm_wheel.py benchmarks/benchmark_wheel.py tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py
git commit -m "Examples: blocks draw their own starts in initial_sample

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Notebooks

**Goal:** The four notebooks teach and use `initial_sample`: the tutorial blocks take `start_amplitude` / `start_variance` and draw from their priors otherwise, the translation probe publishes its zero start itself, the GB notebook registers the block directly, and the demo prose and calls follow; the notebook-executing tests pass.

**Files:**
- Modify (by cell id, JSON round trip): `examples/create_block.ipynb`, `examples/domain_translation.ipynb`, `examples/gb_block_eryn.ipynb`
- Modify (exact text substitution on the raw file): `examples/demo.ipynb`
- Read only: `tests/test_block_tutorial.py` (asserts the five printed "checks passed" lines and execs the `definitions`-tagged cells; its unit tests construct `SineAmplitudeBlock("first", 0.125)` positionally and call `block._render(data, 0.0)`), `tests/test_translation_example.py`, `tests/test_examples.py::test_notebook_code_cells_execute`

**Acceptance Criteria:**
- [ ] `grep -c "initial_block_state\|draw_gb_prior\|initial_at_injection\|initial_noise_result" examples/*.ipynb` prints `0` for all four.
- [ ] `SineAmplitudeBlock` has a trailing field `start_amplitude: float | None = None`, validated finite, and `initial_sample` rendering it or `rng.normal(prior_mean, prior_std)`; `WhiteNoiseVarianceBlock` has `start_variance: float | None = None`, validated positive, and `initial_sample` rendering it or `prior_scale / rng.gamma(prior_shape)`; existing positional constructions and `_render` / `_result` keep working.
- [ ] The tutorial's known-injection campaign registers `SineAmplitudeBlock(..., start_amplitude=true_amplitude)` and its assertions (ledger amplitude equals the injection, counters zero after `add`, contribution equals `true_amplitude * signal_basis`) still hold; the joint campaign registers the noise block first with `start_variance=noise_standard_deviation**2`.
- [ ] `tests/test_block_tutorial.py` passes (all five "checks passed" lines printed); `test_translation_notebook_runs_all_cells_with_domania` and `test_notebook_code_cells_execute` pass; `test_all_notebooks_contain_valid_python` passes for all four notebooks.
- [ ] `git diff --stat` for the three JSON-edited notebooks shows only the intended cells changed (a no-op round trip before editing produces an empty diff); `demo.ipynb`'s diff touches only the substituted lines.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q tests/test_block_tutorial.py tests/test_translation_example.py "tests/test_examples.py::test_notebook_code_cells_execute" "tests/test_examples.py::test_all_notebooks_contain_valid_python"` → all passed, 0 failed.

**Steps:**

- [ ] **Step 1: Confirm the round trip and prepare the editing helper**

Save this as `$TMPDIR/nbedit.py` and use it for every cell edit in this task:

```python
"""nbedit.py: replace notebook cell sources by id, preserving everything else."""

import json
import sys


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dump(nb, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def set_source(nb, cell_id, text):
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            lines = text.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            return
    raise KeyError(cell_id)


if __name__ == "__main__":  # python nbedit.py <notebook> : no-op round trip
    path = sys.argv[1]
    dump(load(path), path)
```

Run: `for nb in create_block domain_translation gb_block_eryn; do .venv/bin/python $TMPDIR/nbedit.py examples/$nb.ipynb; done; git status --short examples/`
Expected: no output from `git status` (the round trip is byte-identical). If a notebook does show as modified, stop and report: do not edit that notebook by JSON.

- [ ] **Step 2: `examples/create_block.ipynb`**

Write `$TMPDIR/edit_tutorial.py` that imports `nbedit`, loads the notebook, applies the `set_source` calls below and dumps it. Cell texts (complete sources):

`create-block-00` (markdown): keep the text, with list items 3 and 4 replaced by

```
3. Implement `initial_sample` and `sample`.
4. Start at a known injection, run a campaign, and check the posterior.
```

`create-block-03` (markdown), complete:

```
## 1. Understand the Block contract

A normal Python class qualifies as a `Block` when it has these three members; no
inheritance, decorator, or plugin registration is required.

| Member | When Wheel uses it | What it does |
| --- | --- | --- |
| `name` | Registration and ledger lookup | A unique, nonempty name within this Wheel. |
| `initial_sample(conditional_residual, covariance, *, rng)` | Once, at `wheel.add` | Draw or construct the block's starting physical state on its input grid. |
| `sample(conditional_residual, covariance, current_block_state, *, rng)` | Once per block per cycle | Advance the model against the current inputs and return its complete physical state. |

`wheel.add(block)` calls `initial_sample` with the observations minus the starts
of the blocks registered before it, the covariance when the campaign has one,
and the campaign RNG, then validates and adopts the returned state exactly like
a `sample` return. A block that starts at a known point takes that point in its
constructor and renders it; a block that starts from its prior draws from `rng`.

For block *i*, the conditional residual is

```text
observations − sum of every OTHER block's accepted TDI signal
```

Its own signal is still present. Evaluate this residual against your candidate
signal. Wheel performs subtraction between blocks; your block never subtracts
another block or adds its own old contribution back.

```text
wheel.add(block)
       ↓
residual of earlier starts + covariance + rng → initial_sample
       ↓
complete BlockState → validate → ledger + residual + covariance
       ↓
conditional residual + covariance + current result + rng → sample   (each cycle)
       ↓
complete BlockState → validate → ledger + residual + covariance
```

Wheel passes independent physical snapshots and commits its RNG only with an
accepted result. A failed call, `initial_sample` included, leaves the published
physical state and Wheel RNG unchanged, but does not undo changes inside the
block. Model code must manage retries or restore its own `state`. Earlier
successful blocks remain committed. Orbit resources are shared and must be
treated as immutable.
```

`create-block-07` (markdown): item 2 becomes "2. `_basis` validates the input conventions; `_render` builds a complete result; `initial_sample` renders the configured `start_amplitude`, or a prior draw when none is configured." and the last paragraph becomes "A helper beginning with `_` is ordinary implementation code; Wheel calls only `initial_sample` and `sample`. This entire class can be copied into your own module."

`create-block-08` (code, tagged `definitions`): the existing class with three changes. After `steps_per_call: int = 8` add `start_amplitude: float | None = None`; at the end of `__post_init__` add

```python
        if self.start_amplitude is not None and not np.isfinite(self.start_amplitude):
            raise ValueError("start_amplitude must be finite")
```

and between `_render` and `sample` add

```python
    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        amplitude = self.start_amplitude
        if amplitude is None:
            amplitude = rng.normal(self.prior_mean, self.prior_std)
        return self._render(conditional_residual, amplitude)
```

`create-block-11` (code), complete:

```python
signal_block = SineAmplitudeBlock(
    name="signal", frequency_hz=frequency_hz, start_amplitude=true_amplitude
)
assert isinstance(signal_block, Block)

# check_block executes the model, initial_sample included. Give it a separate
# instance so its internal counters do not advance the block used in the main
# campaign.
probe = deepcopy(signal_block)
check_block(
    block_under_test=probe,
    observed_data=observed,
    initial_covariance=covariance,
    num_cycles=3,
    random_seed=5,
)
assert probe.state.num_proposals == 3 * probe.steps_per_call
assert signal_block.state.num_proposals == 0
start = signal_block.initial_sample(observed, covariance, rng=np.random.default_rng(0))
assert start.parameters == {"amplitude": true_amplitude}
print("Start parameters:", start.parameters)
print("Initial sampler state:", signal_block.state)
```

`known-injection-start` (markdown), complete:

```
## 6. Register the block and let it render its start

`wheel.add(block)` calls `initial_sample` once and adopts what it returns. Our
block was configured with `start_amplitude=true_amplitude`, so its start is the
injected amplitude and its rendered signal, one consistent physical state;
proposal counters stay on the block. Model authors must check physical
consistency and prior support; Wheel can only check the exchange contract.

`initial_sample` receives the residual on the domain and WDM grid requested at
registration, or the observation grid if no domain is requested. Here both are
time-domain. Wheel validates, copies, and converts the returned state to its
observation grid before adopting it. A covariance in the start must be on the
same grid as the signal and follows the usual ownership rules, including for a
noise-only start.

The observations still contain the injection plus noise. The start sets the
ledger and residual; the declared prior stays the same and amplitude continues
to be sampled. This initializes one block; a full campaign checkpoint would
also need the other blocks' states and the Wheel RNG state.

Leave `start_amplitude` unset and the same block draws its start from the prior
with the campaign RNG, so a seeded campaign reproduces its start as well as its
chain. Keep the prior in the sampling target regardless of the start.
```

`known-injection-checks` (code): line 3 becomes `wheel.add(signal_block, data_domain="time")`; everything else unchanged (the assertions on `true_amplitude`, the counters, the contribution, the pristine observations and the residual all still hold).

`create-block-13` (code): replace `assert initial_at_injection.parameters == {"amplitude": true_amplitude}` with `assert signal_block.start_amplitude == true_amplitude`.

`create-block-18` (code): both constructions gain `start_amplitude=true_amplitude`, and the loop body becomes `campaign.add(block, data_domain="time")`.

`create-block-19` (markdown): the sentence beginning "The initial signal uses the requested time-domain grid" becomes "`initial_sample` receives the requested time-domain grid too, even though the observations are stored in frequency coordinates."

`create-block-23` (code, tagged `definitions`): after `prior_scale: float = 0.25` add `start_variance: float | None = None`; at the end of `__post_init__` add

```python
        if self.start_variance is not None and not (
            np.isfinite(self.start_variance) and self.start_variance > 0
        ):
            raise ValueError("start_variance must be finite and positive")
```

and between `_result` and `sample` add

```python
    def initial_sample(self, conditional_residual, covariance, *, rng):
        variance = self.start_variance
        if variance is None:
            variance = self.prior_scale / rng.gamma(self.prior_shape)
        return self._result(conditional_residual, variance)
```

`create-block-24` (code), complete:

```python
joint_campaign = Wheel(observed, random_seed=9)
# The noise block goes first: its start publishes the covariance the signal
# sampler needs, and each block starts from the residual of the earlier starts.
joint_campaign.add(
    WhiteNoiseVarianceBlock(start_variance=noise_standard_deviation**2),
    data_domain="time",
)
joint_campaign.add(
    SineAmplitudeBlock(
        name="signal", frequency_hz=frequency_hz, start_amplitude=true_amplitude
    ),
    data_domain="time",
)
joint_campaign.run(50)
noise_result = joint_campaign.ledger["noise"]
assert noise_result.template is None
assert noise_result.covariance is not None
np.testing.assert_array_equal(joint_campaign.contribution("noise")["A"], 0.0)
np.testing.assert_allclose(
    joint_campaign.covariance.A,
    noise_result.parameters["variance"],
)
print("Last noise variance draw:", noise_result.parameters["variance"])
print("Noise-only block checks passed")
```

`create-block-25` (markdown): the bullet beginning "Implement `sample`; supply a complete, consistent `initial_block_state`" becomes

```
- Implement `initial_sample` and `sample`. Take a chosen start (an injection, a
  checkpoint) as block configuration and render it in `initial_sample`;
  otherwise draw the start from the prior with the `rng` Wheel passes, and
  initialize the model's own `State` either way.
```

- [ ] **Step 3: `examples/domain_translation.ipynb`**

`translation-16` (markdown): the last paragraph becomes "Each block's `initial_sample` publishes a zero template on the grid Wheel hands it, so the start is on the selected grid by construction; registration calls it once. Use a separate block instance for each independent campaign, because internal state belongs to it."

`translation-17` (code): add to `DomainProbe`, above `sample`:

```python
    def initial_sample(self, conditional_residual, covariance, *, rng) -> BlockState:
        assert conditional_residual.data_domain == self.expected_domain
        # This campaign carries no covariance.
        assert covariance is None
        return conditional_residual.zero_block_state()
```

`translation-18` (code): the loop becomes

```python
for reference_data in (observed, spectrum, wdm_finer_frequency):
    domain = reference_data.data_domain
    block = DomainProbe(f"{domain}_block", domain)
    blocks[block.name] = block
    options = {"num_time_divisions": 8} if domain == "wdm" else {}
    wheel.add(block, data_domain=domain, **options)
```

(the rest of the cell is unchanged; `num_sample_calls == 2` still holds because `initial_sample` does not touch the counter).

- [ ] **Step 4: `examples/gb_block_eryn.ipynb`**

`0cf42b71` (code): last line becomes `from gb_model import FixedLISANoise, GBBlock, inject_gb, waveform`.

`ebeb17db` (markdown): the first paragraph's first two sentences become "`GBBlock.initial_sample` draws each walker from the declared uniform-box prior with the campaign RNG; `Wheel.add` calls it once and adopts the first walker's rendered signal as the start. Every `sample` call receives `conditional_residual`, `covariance` and `current_block_state`."

`7261596e` (code): replace the lines from `# Distinct child streams make initialization` through `)  # enchilada delivers the GB input grid` with

```python
wheel = Wheel(observed, initial_covariance=covariance, random_seed=0)  # enchilada
wheel.add(block, data_domain="frequency")  # enchilada draws the walkers on the GB input grid
```

- [ ] **Step 5: `examples/demo.ipynb` by text substitution**

`demo.ipynb` does not round-trip through `json`; edit the raw file with a script that asserts each old fragment occurs exactly once:

```python
from pathlib import Path

path = Path("examples/demo.ipynb")
text = path.read_text(encoding="utf-8")
substitutions = [
    (
        '"from enchilada import BlockState, L1Data, Wheel\\n",',
        '"from enchilada import L1Data, Wheel\\n",',
    ),
    (
        '"and a ledger of physical `BlockState` snapshots. Supply `initial_block_state`\\n",\n'
        '    "when registering each block; registration does not call the sampler.\\n",',
        '"and a ledger of physical `BlockState` snapshots. Registration calls each block\'s\\n",\n'
        '    "`initial_sample` once and adopts the start it returns.\\n",',
    ),
    ('"wheel.add(ucb, initial_block_state=BlockState())\\n",', '"wheel.add(ucb)\\n",'),
    ('"wheel.add(mbhb, initial_block_state=BlockState())\\n",', '"wheel.add(mbhb)\\n",'),
    (
        "Its caller-side helpers draw genuine priors with a separate initialization RNG, and it reads",
        "Its blocks draw genuine priors in `initial_sample`, it reads",
    ),
    (
        "provide `name` and implement `sample(conditional_residual, covariance, current_block_state, *, rng)` "
        "(`src/enchilada/block.py` documents the contract). Prepare a complete `initial_result`, then run "
        "`enchilada.testing.check_block(block_under_test=your_block, observed_data=toy_observed, "
        "initial_block_state=initial_result, num_cycles=2, initial_covariance=None, random_seed=0)` "
        "before joining a shared campaign. Use a separate block instance for this check: it runs the model "
        "and advances its internal state. This validates the initial physical state and two sampling cycles; "
        "the optional initial covariance and random seed configure the test campaign.",
        "provide `name` and implement `initial_sample(conditional_residual, covariance, *, rng)` and "
        "`sample(conditional_residual, covariance, current_block_state, *, rng)` "
        "(`src/enchilada/block.py` documents the contract). Run "
        "`enchilada.testing.check_block(block_under_test=your_block, observed_data=toy_observed, "
        "num_cycles=2, initial_covariance=None, random_seed=0)` before joining a shared campaign. "
        "Use a separate block instance for this check: it runs the model, `initial_sample` included, "
        "and advances its internal state. This validates the start and two sampling cycles; "
        "the optional initial covariance and random seed configure the test campaign.",
    ),
]
for old, new in substitutions:
    assert text.count(old) == 1, old[:60]
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
```

If an assertion fails, open the raw file at the corresponding line (`grep -n` for a distinctive phrase) and copy the exact text, including JSON escapes, into `old`.

- [ ] **Step 6: Gates and commit**

Run: `grep -c "initial_block_state\|draw_gb_prior\|initial_at_injection\|initial_noise_result" examples/*.ipynb` → four zeros.
Run: `git diff --stat examples/*.ipynb` → four notebooks, modest line counts (tens, not hundreds, for the three JSON-edited ones).
Run the **Verify** command (the tutorial notebook runs 1200 cycles; allow a minute).

```bash
git add examples/create_block.ipynb examples/domain_translation.ipynb examples/gb_block_eryn.ipynb examples/demo.ipynb
git commit -m "Notebooks: blocks render or draw their own starts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: README, CHANGELOG, wording sweep, full gates

**Goal:** The README and CHANGELOG describe the new contract, no stale wording about supplied starts remains anywhere outside the historical 0.3.0 changelog section and `docs/superpowers`, and the complete suite with all gates is green on the branch.

**Files:**
- Modify: `README.md:92-104` (quick-start snippet), `:224-255` (registration paragraph), `:358,362` (API table), `:385-420` (`check_block` section), `:452-490` (domain selection), `:564-568` (validation bullet), `:742` (limitations)
- Modify: `CHANGELOG.md` (`## [0.4.0] — Unreleased`: Added and Changed)
- Modify: whatever the sweep in Step 3 finds

**Acceptance Criteria:**
- [ ] `grep -n "initial_block_state\|initial_result\|caller-side\|draw_gb_prior\|draw_sine_prior\|draw_white_noise_prior" README.md` prints nothing; the API table rows read `add(block_to_register, *, data_domain=None, num_frequency_divisions=None, num_time_divisions=None)` and `check_block(block_under_test, observed_data, num_cycles=2, *, initial_covariance=None, random_seed=0)`.
- [ ] README states that the Block protocol requires `name`, `initial_sample` and `sample`, that registration calls `initial_sample` with the residual of earlier starts on the block's grid, the covariance and the campaign RNG, that a chosen start is block configuration, and that a block on another grid than the covariance is refused at `add` or `run`.
- [ ] CHANGELOG 0.4.0 has an Added bullet for `Block.initial_sample` and a **Breaking** bullet for the removal of `initial_block_state` from `Wheel.add` and `check_block`, and its covariance bullet says "refused at `add` or `run`" and "a start or returned covariance".
- [ ] `grep -rn "initial_block_state\|draw_gb_prior\|draw_sine_prior\|draw_white_noise_prior\|initial_result\|does not call the sampler\|never calls model initialization" src tests examples benchmarks README.md` prints nothing, and in `CHANGELOG.md` the only hits are in the `## [0.3.0]` section or below.
- [ ] Full suite green with coverage at or above the gate; ruff check, ruff format --check and mypy clean; `tests/test_packaging_docs.py` (README link check) passes.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q --cov=enchilada` → all passed, 0 failed (4 skips acceptable), coverage ≥ 95 %; `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy` → clean.

**Steps:**

- [ ] **Step 1: README**

Quick-start snippet (lines 95-103) becomes

```python
wheel = Wheel(observed_data=observed)
wheel.add(block_to_register=ucb)
wheel.add(block_to_register=mbhb)
wheel.run(num_cycles=3)
```

and, if that snippet's import line lists `BlockState` without another use in the snippet, drop it from the import.

Replace lines 224-255 (from "Every `Wheel.add` requires a complete `initial_block_state`" through "Noise-only initial states are supported too.") with:

````markdown
Every block draws or constructs its own start. `Wheel.add` calls the block's
`initial_sample` once with the observations minus the adopted starts of the
blocks registered before it, on the block's grid, together with the current
covariance (`None` until a block supplies one) and the campaign RNG, then
validates and adopts the returned state exactly like a `sample` return. For a
known injection, make the start block configuration:

```python
signal_block = SineAmplitudeBlock(
    name="signal", frequency_hz=0.004, start_amplitude=injected_amplitude
)
wheel.add(block_to_register=signal_block, data_domain="time")
```

A block whose start is not configured draws it from its prior with the `rng`
it is handed, so a seeded campaign reproduces its starts as well as its
chains; registration consumes the campaign RNG only when `initial_sample`
draws from it. The Block protocol requires `name`, `initial_sample` and
`sample`. The [Block authoring tutorial](examples/create_block.ipynb) shows a
complete example. Return signal arrays on the block's requested domain and
WDM grid, or the observation grid if no domain is selected; Wheel converts
them to the observation grid for storage. A covariance in the start follows
the normal ownership rules and must be on the block's own grid, like its
template; Wheel holds it on that grid (see
[Time, frequency, and WDM translation](#time-frequency-and-wdm-translation)).
Noise-only starts are supported too, and blocks start in registration order:
register a noise block before the blocks whose starts need its covariance.
````

API table rows 358 and 362:

```markdown
| Register a block | `add(block_to_register, *, data_domain=None, num_frequency_divisions=None, num_time_divisions=None)` |
| Check a block | `check_block(block_under_test, observed_data, num_cycles=2, *, initial_covariance=None, random_seed=0)` |
```

`check_block` section (lines 391-412) becomes:

````markdown
```python
from enchilada.testing import check_block

check_block(
    block_under_test=MyBlock(name="ucb"),
    observed_data=toy_observed,
    num_cycles=2,
    initial_covariance=None,
    random_seed=0,
)
```

This registers the block, which calls its `initial_sample` on the grid of
`observed_data`, then runs the requested number of sampling cycles, advancing
the supplied block's own state. Construct `MyBlock` with whatever start it
should render; `check_block` does not select a separate block domain. The
optional initial covariance and random seed configure the test campaign, and
the seed also drives `initial_sample`; the defaults are shown above. Each
model must independently validate its prior, likelihood, and sampler; see the
conjugate example in `examples/toy_fit.py`.
````

Domain selection (lines 452-466):

````markdown
```python
wheel.add(frequency_block, data_domain="frequency")
wheel.add(wdm_block, data_domain="wdm", num_frequency_divisions=8)
```

Each block's `initial_sample` receives the residual on its selected domain and
WDM grid and returns its start there. Wheel converts
````

(continue with the existing "the conditional residual and previous signal into the selected representation, ..." sentence). Lines 481-484 become "a block registered on another grid is refused at `add` or `run` with `ValueError` naming both grids, and a start or returned covariance on another grid than its block's is refused at `add` or `run`."

Validation bullet (lines 564-566): "`Wheel` checks `name`, `initial_sample` and `sample` before registration, then validates every start or returned signal's channels, lengths, real/complex kind, and finiteness." Limitation at line 742: "a block on another grid is refused at `add` or `run`."

- [ ] **Step 2: CHANGELOG**

Under `## [0.4.0] — Unreleased` → `### Added`, append:

```markdown
- `Block.initial_sample(conditional_residual, covariance, *, rng)`: every
  block draws or constructs its own start. `Wheel.add` calls it once with the
  observations minus the starts of the blocks registered so far, on the
  block's grid, the current covariance and the campaign RNG, and adopts the
  result through the same validation as a `sample` return; a block without it
  is refused with `TypeError`. `EchoBlock` publishes an empty start.
```

Under `### Changed`, after the existing **Breaking** bullets, append:

```markdown
- **Breaking:** `Wheel.add(initial_block_state=...)` and
  `check_block(initial_block_state=...)` are gone; a chosen start (an
  injection, a checkpoint) is block configuration rendered by
  `initial_sample`, and the example prior helpers (`draw_sine_prior`,
  `draw_white_noise_prior`, `draw_gb_prior`) became the blocks'
  `initial_sample`. Registration now consumes campaign RNG draws when the
  start draws from `rng`, and a block registered on another grid than the
  current covariance is refused at `add`, before `initial_sample` is called.
```

In the existing covariance bullet (lines 60-63) change "refused at `run` with `ValueError`" to "refused at `add` or `run` with `ValueError`" and "a seed or returned covariance" to "a start or returned covariance". Leave the `## [0.3.0]` section as history.

- [ ] **Step 3: Wording sweep**

Run:

```bash
grep -rn "initial_block_state\|draw_gb_prior\|draw_sine_prior\|draw_white_noise_prior\|initial_result\|does not call the sampler\|never calls model initialization\|caller-side\|seed covariance\|noise-only seed" src tests examples benchmarks README.md CHANGELOG.md
```

Expected: hits only inside `CHANGELOG.md` at or below `## [0.3.0]`. Fix every other hit in place (rename `seed` → `start` in test names and docstrings where it means the registration state; leave `random_seed` alone).

- [ ] **Step 4: Full gates**

Run the **Verify** commands. Expected: `571 passed, 4 skipped` or close to it (the new test file adds cases and a few tests were merged; any count is fine as long as 0 failed), coverage line `TOTAL ... 9x%` at or above 95, and three clean tool runs. If coverage dropped below the gate, the uncovered lines are in `wheel.py`'s new `add`; add a test to `tests/test_initial_sample.py` for the specific branch rather than lowering the gate.

- [ ] **Step 5: Commit**

```bash
git add README.md CHANGELOG.md
# plus any file the sweep changed, by name
git commit -m "Docs: registration calls initial_sample; initial_block_state is gone

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## After the last task (controller)

1. Final whole-branch review (the subagent-driven-development skill's final reviewer): `git diff main...block-initial-sample --stat` should touch exactly the files in the task map plus the spec and this plan.
2. Full gates one more time on the final commit: `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q --cov=enchilada`, `.venv/bin/python -m ruff check src tests examples benchmarks`, `.venv/bin/python -m ruff format --check src tests examples benchmarks`, `.venv/bin/python -m mypy`.
3. Push and open the PR, each with the sandbox disabled for that one command (the keychain is unreachable inside it):
   - `git push -u nsgs block-initial-sample`
   - `gh pr create --repo nsgs-analysis-center/enchilada --base main --head block-initial-sample --title "Blocks draw their own start: Block.initial_sample" --body-file $TMPDIR/pr-body.md`, where the body summarises the contract change, the reversal of 5b6ff3b, the breaking removal, the migration rule for block authors (configure the start, render it in `initial_sample`, draw from `rng` otherwise), the test plan (gates above), and ends with the line `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
4. Known CI noise, not to be fixed here: the `Lint` workflow's `check-docstring-first` failure predates this branch; macOS lanes may be cancelled for lack of an arm64 runner.
