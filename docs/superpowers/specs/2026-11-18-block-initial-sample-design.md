# Block-Owned Initial Sample — Design

**Date:** 2026-11-18
**Status:** approved in conversation; implementation plan follows
**Builds on:** `main` at 1e2872b (after PRs #6, #7 and #8 merged). This work lands on
branch `block-initial-sample`, cut from `main`.

## 1. Goal

A block draws its own starting physical state. The `Block` protocol gains a required
method `initial_sample(conditional_residual, covariance, *, rng) -> BlockState`, which
`Wheel.add` calls once at registration and adopts exactly as it adopts a `sample`
return. The `initial_block_state` argument disappears from `Wheel.add` and from
`check_block`. Starting a block at a chosen point (an injection, a checkpointed
estimate) becomes block configuration: the block is constructed with its start and
`initial_sample` renders it.

This deliberately reverses the decision recorded in commit 5b6ff3b and in the current
docstrings ("Registration requires and adopts an explicit initial_block_state. It never
calls model initialization or consumes random draws."). Reviewers should not restore
that rule.

## 2. Decisions already made

| Decision | Choice |
|---|---|
| Signature | `initial_sample(self, conditional_residual: L1Data, covariance: Covariance \| None, *, rng: np.random.Generator) -> BlockState`: `sample` minus `current_block_state` (user, 2026-11-18) |
| Override | none. `Wheel.add` always calls `initial_sample`; `initial_block_state` is removed from `Wheel.add` and `check_block`, and a chosen start is block configuration (user, 2026-11-18) |
| RNG | the campaign RNG, through the existing candidate/commit path: `add` advances a copy and commits it with the state, so one `random_seed` reproduces every start and the chain. Registration now consumes draws; the separate "initialization RNG" in the examples goes away (default, stated here) |
| Inputs at `add` | the current working residual (observed minus the adopted starts of every block registered so far) translated to the block's grid, and the current covariance, which must be on the block's grid. A covariance on another grid is refused at `add` with the message `run` uses today (default: fail one call earlier) |
| Adoption | the existing `_adopt` with `source_method_name="initial_sample"`: grid and finiteness checks, covariance ownership, `NoiseOverwrittenWarning`, atomic commit, and translation back to the observation grid through `_canonical_state`, all unchanged |

## 3. The protocol: `src/enchilada/block.py`

```python
@runtime_checkable
class Block(Protocol):
    name: str  # as today, behind the TYPE_CHECKING property trick

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw or construct this block's starting physical state on the input grid."""
        ...

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Advance the supplied state and return the complete new block state."""
        ...
```

Docstring changes. The sentence pair "`Wheel.add` requires an explicit
`initial_block_state`; it calls no initialization hook. `sample` advances the latest
accepted `current_block_state` and returns a complete new state." becomes:
"`Wheel.add` calls `initial_sample` once with the block's starting inputs and adopts
the returned state; `sample` then advances the latest accepted `current_block_state`
and returns a complete new state. Both receive the conditional residual on the
block's registered grid, the covariance when the campaign has one, and the campaign
RNG." The paragraph "Each call receives independent snapshots ..." applies to both
methods. A block that starts at a known point takes that point in its constructor and
returns it from `initial_sample`; a block that starts from its prior draws from `rng`.
Because the protocol is `runtime_checkable`, `isinstance(x, Block)` is now False for
an object without `initial_sample`.

## 4. Registration: `src/enchilada/wheel.py`

```python
def add(
    self,
    block_to_register: Block,
    *,
    data_domain: str | None = None,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
) -> None:
```

Steps, in order; the first five are today's checks with one added:

1. Resolve the block grid from `data_domain` and the division counts (unchanged,
   including the "WDM division counts require data_domain='wdm'" error).
2. Name must be a non-empty string; not already registered (unchanged).
3. `sample` must be callable (unchanged message).
4. `initial_sample` must be callable, else
   `TypeError(f"block {name!r} does not implement initial_sample; a Block needs name, initial_sample(conditional_residual, covariance, *, rng) and sample(conditional_residual, covariance, current_block_state, *, rng).")`.
5. Build the inputs with the same helper `run` uses, generalised to a grid:
   `_block_inputs` takes the block grid and an optional `exclude_block_name`; with no
   name it starts from the working residual (a new block has no signal yet). It raises
   today's "covariance does not translate between grids" `ValueError` when the
   covariance is on another grid, translates the residual when the block grid differs
   from the observation grid, and deep-copies the covariance.
6. `candidate_rng = deepcopy(self._rng)`;
   `new_block_state = block.initial_sample(conditional_residual=..., covariance=..., rng=candidate_rng)`.
7. When the block grid differs from the observation grid,
   `new_block_state = self._canonical_state(new_block_state, conditional_residual)`,
   exactly as `run` treats a `sample` return (a non-`BlockState` passes through so
   that `_adopt` raises the type error).
8. `self._adopt(name, new_block_state, "initial_sample", candidate_rng, block_to_register=block, block_grid=block_grid)`.

Consequences, which the docstrings state:

- An exception from `initial_sample`, a non-`BlockState` return (`TypeError`
  `"{name}.initial_sample must return a BlockState, got X"`), an off-grid or
  non-finite template, or a covariance on another grid leaves the ledger, the block
  list, the residual and the Wheel RNG unchanged.
- Blocks start in registration order: block *i*'s `initial_sample` sees the
  observations minus the adopted starts of blocks *0..i-1*. A noise block that other
  blocks need for their starts is registered first; a block registered before any
  covariance exists receives `covariance=None`.
- A start that publishes a covariance makes the block the covariance owner at
  registration, with the usual takeover warning.
- The class docstring sentences "Registration requires and adopts an explicit
  initial_block_state. It never calls model initialization or consumes random draws."
  become "Registration calls the block's `initial_sample` with the residual of the
  blocks registered so far on the block's grid, the covariance when the campaign has
  one, and the campaign RNG, then adopts the returned state like a `sample` return."
  The `add` docstring is rewritten around the steps above. The sentence "a block on
  another grid is refused at run with ValueError naming both grids" becomes "at add
  or run".

The explicit `check_on_grid` call that `add` makes today on the seed's template before
translating it is dropped; the `run` path's treatment is the precedent, and
`transform` plus `_adopt` produce the errors.

## 5. `src/enchilada/testing.py`

- `check_block(block_under_test, observed_data, num_cycles=2, *, initial_covariance=None, random_seed=0)`.
  Its docstring says registration calls `initial_sample` on the scratch Wheel and that
  the block's start is checked by the same rules as its samples.
- `EchoBlock.initial_sample` returns `BlockState()` without printing. Its docstring
  says "Register with `wheel.add(block)`; `initial_sample` publishes an empty state and
  every `sample` call does the same".

## 6. Examples and benchmark

| File | Change |
|---|---|
| `examples/demo.py`, `examples/demo.ipynb` | `wheel.add(ucb)`, `wheel.add(mbhb)`; the `BlockState` import goes if unused; prose about the empty initial state says the EchoBlocks publish empty starts themselves. `demo.ipynb` is edited by text substitution (it does not round-trip through the JSON tooling) |
| `examples/domain_translation.py`, `.ipynb` | `DomainEcho.initial_sample` runs `_check_input` and returns `BlockState()`; registrations drop the keyword |
| `examples/toy_fit.py` | `SineBlock.initial_sample` draws `amplitude ~ N(0, prior_std)` from `rng` and renders it; `WhiteNoiseBlock.initial_sample` draws `variance = scale / rng.gamma(shape)` and renders it; `draw_sine_prior` and `draw_white_noise_prior` are deleted; `run_toy_fit` seeds one Wheel and drops the spawned initialization stream and its comment. The chain's numbers change because starts now consume campaign draws; the convergence test's tolerances are unchanged and must still pass |
| `examples/gb_model.py`, `examples/gb_block_eryn.ipynb` | `GBBlock.initial_sample(conditional_residual, covariance, *, rng)` replaces `draw_gb_prior` with the same body (reset `block.state` to fresh walkers and an Eryn `RandomState` seeded from `rng`, return the rendered first walker); the class docstring's registration recipe becomes `wheel.add(block, data_domain="frequency")`, and the `sample` error message "Initialize GBBlock.state with draw_gb_prior or a checkpoint" names `initial_sample` instead; the notebook's import, initialization-RNG cell and prose go |
| `examples/mojito_wdm_wheel.py` | `wheel.add(EchoBlock(name="echo"), ...)` without the keyword; unused import removed |
| `benchmarks/benchmark_wheel.py` | `FixedSignal.initial_sample` returns the fixed state it was built with (constructed from the residual's grid or stored at construction) |
| `examples/create_block.ipynb` | `SineAmplitudeBlock` gains a field `start_amplitude: float \| None = None` (validated finite when given) and `initial_sample`: it renders `start_amplitude` when set, otherwise draws `rng.normal(prior_mean, prior_std)` and renders that. Section 6 constructs `signal_block = SineAmplitudeBlock(name="signal", frequency_hz=..., start_amplitude=true_amplitude)` and calls `wheel.add(signal_block, data_domain="time")`; its assertions (ledger amplitude equals the injection, counters still zero, contribution equals the injected signal, pristine data untouched) stay. The `check_block` call drops `initial_block_state`. The tutorial's `WhiteNoiseVarianceBlock` gains a field `start_variance: float \| None = None` and `initial_sample`: it publishes `_result(conditional_residual, start_variance)` when set, otherwise `prior_scale / rng.gamma(prior_shape)`; the joint campaign constructs it with the known variance and registers it before the signal block. Prose in cells 00, 03, 07, 12 ("known-injection-start"), 19, 22 and 25 describes the new flow; "Wheel does not call an initializer or draw a prior" and "registration consumes no Wheel RNG draws" are gone |

## 7. Tests

### 7.1 `tests/test_initial_block_state.py` → `tests/test_initial_sample.py`

The file pins the old decision by name, so it is renamed and its tests inverted rather
than patched. Docstring: "Wheel.add draws each block's start from its initial_sample
and adopts it like a sample." `SeededBlock` becomes `StartingBlock` with an
`initial_sample` that returns the amplitude-2 state on the grid it received and records
the call.

| Old test | New test |
|---|---|
| `test_sample_only_block_implements_protocol` | `test_block_without_initial_sample_is_refused`: `isinstance(block, Block)` is False; `add` raises `TypeError` matching `initial_sample`; the ledger is empty and the Wheel RNG state is unchanged |
| `test_initial_result_is_required_even_if_block_can_draw_prior` | `test_registration_calls_initial_sample_once_with_the_block_inputs`: the first block receives the observations (equal values, independent array), `covariance=None` on a Wheel without one and the Wheel's covariance otherwise, and a `np.random.Generator`; after `run(1)` the call count is still 1 |
| `test_registration_does_not_access_model_initialization` | `test_initial_sample_consumes_the_campaign_rng_reproducibly`: two Wheels seeded alike produce identical starts and identical first samples; a Wheel whose block draws at start continues its chain from a different generator state than one whose block does not |
| `test_seed_skips_prior_and_round_trips_through_selected_domain` (parametrized over the five domain pairs) | `test_initial_sample_receives_inputs_on_the_selected_grid_and_round_trips`: the block asserts the residual's domain and grid, returns a template on that grid, and the ledger holds it on the observation grid |
| `test_seed_snapshot_is_independent_including_noise_and_nested_parameters` | `test_adopted_start_is_an_independent_snapshot`: mutating the returned state's arrays and nested parameters after `add` leaves the ledger unchanged |
| `test_invalid_seed_leaves_registration_and_rng_unchanged`, `test_failed_seed_translation_does_not_register_block` | `test_failed_initial_sample_leaves_registration_and_rng_unchanged`, parametrized over: raises; returns a non-`BlockState`; returns an off-grid template; returns a non-finite template; returns a covariance on another grid |
| `test_seed_without_signal_can_sample_in_selected_domain` | `test_start_without_signal_can_sample_in_selected_domain` |
| `test_seed_translation_does_not_transform_unrelated_observations` | dropped: the residual is now translated at `add` by design |
| `test_noise_only_seed_establishes_covariance_owner` | `test_noise_only_start_establishes_covariance_owner` |
| `test_seeded_noise_takeover_warning_can_fail_atomically` | `test_start_noise_takeover_warning_can_fail_atomically` |
| `test_check_block_can_exercise_an_explicit_injection` | `test_check_block_can_exercise_a_configured_start`: a block constructed with its start |
| new | `test_later_blocks_start_from_the_residual_of_earlier_starts`: block B's `initial_sample` receives observed minus A's adopted start, on B's grid |
| new | `test_covariance_on_another_grid_is_refused_at_add`: Wheel with a time covariance, block registered with `data_domain="frequency"`: `ValueError` naming both grids at `add`, nothing registered |
| new | `test_start_that_publishes_a_covariance_is_handed_to_the_next_block`: noise block first, signal block second receives that covariance |

### 7.2 Other test files

- `tests/test_testing.py`: `test_initial_block_state_is_required` and
  `test_initial_block_state_must_be_a_block_state` go; a new test checks that
  `check_block` refuses a block without `initial_sample` with the Wheel's `TypeError`;
  the EchoBlock tests register with `wheel.add(block)`.
- `tests/test_typing.py`: the mypy fixtures gain `initial_sample` on every conforming
  block, and a block lacking it must fail to type-check as a `Block` (the existing
  `sample_only` fixture flips from conforming to non-conforming).
- Every other file that calls `wheel.add(..., initial_block_state=...)` or
  `check_block(..., initial_block_state=...)` (`test_wheel.py`, `test_wheel_translation.py`,
  `test_state.py`, `test_block_state.py`, `test_block_state_contract.py`,
  `test_ledger.py`, `test_toy_translation.py`, `test_translation_review.py`,
  `test_packaging_docs.py`, `test_examples.py`, `test_gb_example_contract.py`) is
  updated mechanically: a test block whose explicit seed was a constant returns that
  constant from `initial_sample`. A `tests/conftest.py` helper that wraps a block with a
  fixed start (`FixedStart(block, state)` or a mixin) is allowed to keep that diff
  small; the plan decides its shape.
- `tests/test_examples.py`, `tests/test_toy_translation.py` and
  `tests/test_gb_example_contract.py` tests that called `draw_sine_prior`,
  `draw_white_noise_prior` or `draw_gb_prior` call `initial_sample` on a block instance
  with an explicit `np.random.default_rng(...)`, or simply register the block and read
  the ledger.

### 7.3 Gates

`ruff check`, `ruff format --check` (notebooks excluded from formatting as always),
`mypy`, and the full suite with the 95 % coverage gate; the example tests, the toy-fit
convergence test and the notebook test must pass with their current tolerances.

## 8. Documentation

- `README.md`: the registration paragraph and snippet (today's "Every `Wheel.add`
  requires a complete `initial_block_state` ..." through "Noise-only initial states are
  supported too.") is rewritten around `initial_sample`: a short block showing the
  method, what it receives, registration order, the campaign RNG, the configured-start
  pattern for an injection, and covariance ownership at registration. The API table
  rows for `add` and `check_block`, the `check_block` section, the domain-selection
  snippet ("Select a representation when registering a block") and its following
  paragraph, the covariance sentence "refused at `run`" (now "at `add` or `run`"), and
  the validation bullet "every supplied initial or returned signal" (now "every
  returned signal, at registration and after each call") are updated.
- `CHANGELOG.md`, `[0.4.0] — Unreleased`: under Added, "`Block.initial_sample`: a
  block draws or constructs its own starting state; `Wheel.add` calls it and adopts
  the result like a `sample` return." Under Changed, a second **Breaking** bullet:
  `Wheel.add` and `check_block` lose `initial_block_state`; registration consumes
  campaign RNG draws; a covariance on another grid than the block's is refused at
  `add`; `isinstance(x, Block)` requires `initial_sample`.
- Docstrings in `block.py`, `wheel.py` and `testing.py` as in sections 3 to 5.
- The four notebooks as in section 6.

## 9. Non-goals

- No campaign checkpoint or restore; a configured start initializes one block.
- No retry or fallback when `initial_sample` fails; the exception propagates.
- No change to `sample`, `BlockState`, `Ledger`, translation or covariance rules.
- No new helper on `L1Data`; blocks build their start with the existing
  `block_state` and `zero_block_state`.

## 10. Self-review

Placeholders: none. Consistency: the inputs, RNG and adoption rules in sections 2, 4 and
7 agree; the tutorial's configured start in section 6 is the pattern section 1 names.
Scope: one mechanism, one branch, one plan. Ambiguity: the only open shape is the
conftest helper in 7.2, left to the plan.
