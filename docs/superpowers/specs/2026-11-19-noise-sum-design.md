# Noise blocks sum their covariances: roles at `Wheel.add`

**Date:** 2026-11-19. **Branch:** `noise-sum`, cut from `main` at be133a3 (after PR #9).

## 1. Problem

`Wheel` keeps one covariance with one owner. A second block that publishes a
covariance replaces the first and triggers `NoiseOverwrittenWarning`; the
warning's docstring tells the author to "publish a combined covariance from a
single noise block". That rules out the campaign this package exists for, where
instrument noise, galactic confusion and other components are separate models
sampled by separate blocks.

The numerator already works the way the denominator should. `Wheel` keeps a
persistent sum of every block's template and hands block *i* the data minus the
other blocks' templates. This design gives the covariance the same treatment: a
persistent sum of every noise block's covariance, with a noise block receiving
the other noise blocks' sum and a signal block receiving the whole sum.

## 2. Decisions made with the user (2026-11-19)

| Question | Decision |
|---|---|
| What a noise block receives as `covariance` | The sum over the *other* noise blocks, None when it is alone. Its own last publication stays on `current_block_state.covariance`. Signal blocks receive the sum over all noise blocks. |
| How a role is declared | `wheel.add(block, *, role=...)` with `role` a required keyword, `"signal"` or `"noise"`; strict: a signal block may not publish a covariance, a noise block may not publish a template. A joint model is split into two blocks. |
| Which grid the sum lives on | The first publication fixes it. A later noise block on another grid is refused at `add`, before its `initial_sample` runs. |
| `Wheel(initial_covariance=...)` | Removed. Every covariance comes from a noise block's `initial_sample`; a known noise model is a `FixedNoiseBlock`. |
| Where the sum is kept | A persistent `NoiseSum` mirroring `SignalSum`: per-registration-index slots, `total` and `excluding(i)`, candidates discardable. |
| Demonstration | A new small example, `examples/noise_floor.py`; the tutorial's conjugate noise block stays single-noise and says so. |

## 3. Public API

### 3.1 `Wheel`

```python
Wheel(observed_data, random_seed=None)
```

`initial_covariance` is gone. A campaign has no covariance until a noise block
publishes one at registration.

```python
wheel.add(
    block_to_register,
    *,
    role,                      # "signal" | "noise", required
    data_domain=None,
    num_frequency_divisions=None,
    num_time_divisions=None,
)
```

`role` is validated first: anything but the two strings raises
`ValueError("role must be 'signal' or 'noise', got ...")` before the block is
inspected. The rest of `add` is unchanged from PR #9: name and duplicate checks,
`sample` and `initial_sample` callable checks, `_block_inputs`, the candidate
RNG, `initial_sample`, `_canonical_state`, `_adopt`.

`wheel.covariance` returns the sum over all noise blocks as an independent
`Covariance`, or None when no noise block has published. The ledger keeps each
noise block's own publication, as today.

`NoiseOverwrittenWarning` is removed from `enchilada.wheel` and from the
package namespace. Nothing overwrites anything.

### 3.2 What each role contributes and receives

| | Signal block | Noise block |
|---|---|---|
| Contributes | a template (absent template contributes zero, as today) | a covariance, on every `initial_sample` and `sample` call |
| May not return | a covariance | a template |
| `conditional_residual` | data minus every other signal block's template | data minus every signal block's template |
| `covariance` | sum over all noise blocks, or None | sum over the other noise blocks, or None |
| `current_block_state` (in `sample`) | its ledger entry | its ledger entry, whose `covariance` is its own last publication |

Registration order still matters the way PR #9 documented: blocks start in
registration order, so a signal block whose start needs the covariance
registers after the noise blocks, and a noise block registered second receives
the first one's covariance at its `initial_sample`.

### 3.3 `Covariance.__add__`

```python
total = instrument + confusion
```

Two covariances on the same grid with the same channel order add elementwise;
the result is a new validated `Covariance` on that grid, with the usual float64
or complex128 promotion. A different grid raises `ValueError` naming both grids
("a covariance does not add across grids"); different channel order raises
`ValueError` naming both; any other right operand returns `NotImplemented`.
This is how a noise block forms others plus own for its own likelihood.

### 3.4 `FixedNoiseBlock`

```python
from enchilada import FixedNoiseBlock
wheel.add(FixedNoiseBlock("floor", covariance), role="noise", data_domain="frequency")
```

A core block (new module `src/enchilada/fixed_noise.py`, exported from
`enchilada`). `initial_sample` and `sample` both return
`BlockState(covariance=<copy of the configured covariance>)` and ignore their
inputs and `rng`. The constructor requires a `Covariance` (else `TypeError`) and
stores an independent copy. The covariance must be on the grid the block is
registered on; `Wheel` enforces that at adoption exactly as for any noise block,
so a frequency-grid covariance means `data_domain="frequency"` at `add`. It is
the migration path for every `initial_covariance` caller.

### 3.5 `check_block`

```python
check_block(
    block_under_test,
    observed_data,
    num_cycles=2,
    *,
    role,
    noise_blocks=(),
    random_seed=0,
)
```

Registers each block in `noise_blocks` with `role="noise"`, in order, then
`block_under_test` with `role`, then runs `num_cycles`. All registrations are on
the observation grid, as today. A signal block that needs a covariance gets it
through a `FixedNoiseBlock` in `noise_blocks`; a noise block under test receives
the `noise_blocks` sum as its `covariance`.

## 4. Inside `Wheel`

### 4.1 State

| Attribute | Meaning |
|---|---|
| `_roles: dict[str, str]` | block name to `"signal"` or `"noise"`, set at `add`, never changed |
| `_noise_sum: NoiseSum` | persistent sum of noise blocks' covariances by registration index |
| `_covariance_grid: Grid | None` | fixed by the first noise publication; None before |

`_covariance` and `_noise_owner_block_name` are removed. One registration index
serves both trees: a signal block's slot in `_noise_sum` and a noise block's
slot in `_signal_sum` simply stay empty (a noise block's template is refused
before adoption, so its signal leaf is always None).

### 4.2 `NoiseSum`

`src/enchilada/_signal_sum.py` becomes `src/enchilada/_sum_tree.py`: the node
functions take a `combine` callable, and two thin frozen dataclasses wrap them.
`SignalSum` keeps its name, API and the dict-of-arrays combine. `NoiseSum`
holds `Covariance | None` per slot and combines with `Covariance.__add__`;
`total` and `excluding(index)` return `Covariance | None`. Both trees are
persistent: replacing a leaf copies only its path, so a candidate built for a
failed adoption is discarded without touching accepted state.

### 4.3 Inputs by role (`_block_inputs`)

```python
def _block_inputs(self, block_name, block_grid, *, registered) -> tuple[L1Data, Covariance | None]
```

- Residual: unchanged. `residual(exclude_block_name=...)` when registered,
  `working_residual` otherwise; translated to `block_grid`.
- Grid check: when `_covariance_grid` is not None and differs from
  `block_grid`, refuse with `ValueError`. Signal role keeps today's text
  ("... a covariance does not translate between grids ..."); noise role says
  `"block '<name>' is registered on grid X but the campaign covariance is on grid Y; a covariance does not add across grids"`.
  This runs before `initial_sample` at `add` and before `sample` at `run`.
- Covariance: `_noise_sum.total` for a signal block; for a noise block
  `_noise_sum.excluding(index)` when registered, `total` when not yet
  registered (its slot does not exist). The result is handed over as
  `Covariance.copy(...)`, or None when the sum is empty.

### 4.4 Adoption by role (`_adopt`)

After the existing `BlockState` type check, deep copy, template grid and
finiteness checks:

| Role | State content | Outcome |
|---|---|---|
| signal | `covariance is not None` | `ValueError("<name>.<method> returned a covariance but '<name>' is registered as a signal block")` |
| noise | `template is not None` | `ValueError("<name>.<method> returned a template but '<name>' is registered as a noise block")` |
| noise | `covariance is None` | `ValueError("<name>.<method> must return a covariance because '<name>' is registered as a noise block")` |
| noise | covariance present | revalidated through `Covariance.copy`; channel order must match the data (existing message); grid must equal `block_grid` (existing message); then `candidate_noise_sum = self._noise_sum.with_covariance(index, covariance)` and `candidate_covariance_grid = block_grid` |

The candidate ledger, signal sum, residual, indices, domains, roles, noise sum,
covariance grid and RNG are all computed before the single commit block, so
any failure leaves everything unchanged, including the noise sum. The
owner bookkeeping and the overwrite warning are gone.

Positive definiteness: `Covariance` validates every publication positive
definite at every point; a sum of such matrices is positive definite, and
`__add__` builds its result through the validating constructor anyway, so a
bad sum cannot be handed to a block silently.

### 4.5 Unchanged

Signals stored on the observation grid; `_canonical_state`; the candidate RNG
and its commit; `on_cycle_complete`; the ledger; `residual(...)`;
`working_residual`; translation of residuals and templates; everything PR #9
settled about `initial_sample`.

## 5. Examples, notebooks, benchmark

| File | Change |
|---|---|
| `examples/toy_fit.py` | `role="signal"` for the sines, `role="noise"` for `WhiteNoiseBlock`; module docstring sentence on roles |
| `examples/gb_model.py` | `GBBlock.initial_sample` docstring precondition: "register a noise block (for a known PSD, `FixedNoiseBlock`) on the frequency grid first" |
| `examples/gb_block_eryn.ipynb` | `Wheel(observed, random_seed=0)`; `wheel.add(FixedNoiseBlock("noise", covariance), role="noise", data_domain="frequency")` before `wheel.add(block, role="signal", data_domain="frequency")`; prose |
| `examples/domain_translation.py`, `.ipynb` | `role="signal"` on the probes; prose where registration is described |
| `examples/demo.py`, `demo.ipynb` | `role="signal"`; `check_block(..., role="signal")` in prose |
| `examples/mojito_wdm_wheel.py` | `role="signal"` on its `EchoBlock` registration |
| `benchmarks/benchmark_wheel.py` | `role="signal"` |
| `examples/create_block.ipynb` | `role=` on every `wheel.add`; known-injection campaign registers `FixedNoiseBlock("noise", covariance)` instead of `Wheel(initial_covariance=...)`; `check_block(..., role="signal", noise_blocks=(FixedNoiseBlock(...),))` where it needs a covariance; `WhiteNoiseVarianceBlock.initial_sample`/`sample` refuse a non-None `covariance` with `ValueError("WhiteNoiseVarianceBlock models the whole noise; register it as the only noise block")`; contract table, flow diagram, section 6 and checklist prose updated for roles and the sum; one sentence pointing to `examples/noise_floor.py` for several noise blocks |
| `examples/noise_floor.py` (new) | One channel of white noise on a known floor. `FixedNoiseBlock("floor", floor_covariance)` plus `ExcessNoiseBlock("excess")`, a noise block holding a variance `v` with a log-normal-walk Metropolis step: it receives the floor as `covariance`, forms `floor + own` with `Covariance.__add__`, evaluates the Gaussian likelihood of the residual under each total, accepts or rejects with `rng`, and publishes `Covariance.like(residual, {ch: full(N, v)})`. `initial_sample` draws `v` from its log-uniform prior. `run_example()` returns the posterior mean of `v` after burn-in; `main()` prints it against the truth |
| `tests/test_noise_floor_example.py` (new) | skip-safe like the other example tests: the posterior mean of `v` is within tolerance of the truth, the floor block's ledger covariance never changes, `wheel.covariance` equals floor plus the excess block's last publication |

## 6. Tests

New or rewritten; everything else in the suite gains `role=` and loses
`initial_covariance` mechanically.

| Behaviour | Test |
|---|---|
| tree: total of zero, one, three slots; `excluding` each slot; a discarded candidate leaves the accepted tree unchanged; slot replacement | `tests/test_sum_tree.py` (new; parametrized over both trees) |
| `Covariance.__add__`: elementwise sum, dtype promotion, cross terms, grid mismatch, channel-order mismatch, non-Covariance operand | `tests/test_covariance.py::TestAdd` |
| `add` refuses a bad role before touching the block | `tests/test_wheel.py::TestRoles::test_role_is_validated_first` |
| a signal block's covariance is refused; a noise block's template is refused; a noise block's missing covariance is refused; each at `initial_sample` and at `sample`; each leaves ledger, residual, noise sum and RNG unchanged | `tests/test_wheel.py::TestRoles` (parametrized over method) |
| two noise blocks: each receives the other's covariance; a signal block receives the sum; `wheel.covariance` equals the sum; three noise blocks: each receives the sum of the other two | `tests/test_wheel.py::TestNoiseSum` |
| the lone noise block receives None; a noise block registered second receives the first one's covariance at `initial_sample` | `tests/test_initial_sample.py` (rewrites of the owner tests) |
| the first publication fixes the grid; a noise block on another grid is refused at `add` before `initial_sample`; a signal block on another grid is refused at `add` and at `run` | `tests/test_wheel_translation.py` (rewrites of the "covariance on another grid" tests, plus a WDM noise block pair) |
| `FixedNoiseBlock`: type check, independent copy, both methods publish it, works under `check_block` as the block under test and in `noise_blocks` | `tests/test_testing.py::TestFixedNoiseBlock`, `tests/test_fixed_noise.py` |
| `check_block(role=..., noise_blocks=...)`: order of registration, signal block gets the sum, noise block under test gets the others | `tests/test_testing.py::TestCheckBlock` |
| typing: `Role` literal accepted by mypy; `NoiseOverwrittenWarning` import fails | `tests/test_typing.py` |
| `enchilada.NoiseOverwrittenWarning` and `Wheel(initial_covariance=...)` are gone | `tests/test_wheel.py::TestRemovedApi` |

Gates: full suite with coverage at or above 95 %, `ruff check`,
`ruff format --check`, `mypy`, README doc tests.

## 7. Documentation

- README: API rows for `Wheel`, `add`, `check_block`, `FixedNoiseBlock`,
  `Covariance.__add__`; the covariance section explains the sum, the two input
  rules and the grid rule; the noise-ownership paragraphs and the Known
  limitations entry "One noise model at a time" go; a short "Several noise
  blocks" paragraph points at `examples/noise_floor.py`.
- CHANGELOG 0.4.0. Breaking: `role` required at `Wheel.add`;
  `Wheel(initial_covariance=...)` removed; `NoiseOverwrittenWarning` removed;
  a signal block may not publish a covariance and a noise block must publish
  one on every call and may not publish a template. Added: `FixedNoiseBlock`,
  `Covariance.__add__`, the noise sum and the per-role inputs,
  `check_block(role=..., noise_blocks=...)`, `examples/noise_floor.py`.
- Docstrings: `Wheel` class and `add`, `Block` protocol, `BlockState`
  (`covariance` "replacing the current noise model; components are not
  summed" becomes "this block's contribution to the noise sum"), `check_block`.

## 8. Non-goals

- Translating a covariance between grids. The sum lives on one grid.
- Partial noise models that cover some channels or some points only. Every
  publication is a complete positive-definite covariance; a component that is
  zero somewhere is not representable, by `Covariance`'s own rules.
- A joint signal-and-noise block. Split it.
- Changing a block's role after registration, or reading roles back through
  a public accessor. The ledger entry shows what a block published.
- Subtracting covariances.
