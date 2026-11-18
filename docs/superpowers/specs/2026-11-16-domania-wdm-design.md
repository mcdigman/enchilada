# domania as the WDM Backend, Folded Layout — Design

**Date:** 2026-11-16
**Status:** approved in conversation; implementation plan follows
**Builds on:** the Mojito loader on branch `mojito-loader` (PR #7), itself stacked on
`covariance-simplification` (PR #6). This work lands on branch `domania-wdm`, stacked on
`mojito-loader`.

## 1. Goal

Let a campaign load Mojito L1 data, decimate it, translate it to the WDM domain and hand
that WDM observation to a `Wheel`, with the translation done by domania 3.0 from PyPI.
Along the way enchilada's WDM domain adopts domania's folded `(Nf, Nt)` coefficient
layout, and the private `wdm` package stops being a dependency of anything.

## 2. Decisions already made

Recorded so the plan and the implementers do not re-ask them.

| Decision | Choice |
|---|---|
| WDM layout | folded `(Nf, Nt)`, domania's layout: every cell carries data; `active_mask` and every inactive-cell rule are deleted (user, 2026-11-16) |
| Record fitting | the caller sizes the window so the decimated record has `Nf * Nt` samples; the arithmetic lives in the example and the README, not in `transform` or the loader (user, 2026-11-16) |
| Backend source | domania 3.0.0 from PyPI; the private `wdm` package is dropped entirely (user, 2026-11-16) |
| Dependency form | domania is a **core** dependency, not an extra; with it NumPy ≥ 2.1 and SciPy ≥ 1.14.1 become core requirements (user, 2026-11-16) |
| Placement of the WDM step | after `load_mojito_l1` and `decimate`, as `transform(data, "wdm", ...)`; the loader stays time-domain (follows the 2026-11-14 decision "window + separate decimate") |
| Example | `examples/mojito_wdm_wheel.py`, the Wheel side of the Mojito story; `load_mojito.py` stays the data side |

This spec supersedes three records of the covariance-simplification spec
(`2026-11-11-covariance-simplification-design.md`): the decision-table row "domainia
dependency: none in this release; the core stays NumPy-only", the row "the padded
`(Nf + 1, Nt)` layout and the active-pixel rule", and the non-goal "moving the data
transforms (`_transforms.py`, the `wdm` backend) to domainia". domania's own
folded-WDM spec (section 6) left the layout choice to enchilada and recommended the fold.

## 3. domania 3.0, as verified

Facts read from PyPI and from `~/Documents/lisa/domainia` at ef7b691 (package `domania`
3.0.0), and measured on this laptop with domania installed in enchilada's `.venv`.

- Distribution `domania` 3.0.0 on PyPI (wheel and sdist uploaded 2026-11-16),
  `requires-python >= 3.12`, dependencies `numpy >= 2.1, < 3` and
  `scipy >= 1.14.1, < 2`, pure Python, typed (`Typing :: Typed`).
- `domania.transform(x, source, target, *, N, dt, source_wdm=None, target_wdm=None,
  window=MEYER)`. `source` and `target` are `"time"`, `"frequency"` or `"wdm"`, the same
  names enchilada uses. `x` may carry any leading axes, so enchilada's channel stack goes
  through in one call. `N` is the record's sample count. A `"wdm"` side needs its
  `(Nf, Nt)`; the two WDM grids may differ, and WDM → WDM then regrids through the rfft.
  When source equals target, grid included, the result is a validated copy.
- Layouts on the trailing axes: time is real `(N,)`; frequency is complex `(N // 2 + 1,)`
  with `dt * rfft(x)` normalisation, exactly enchilada's convention; WDM is real `(Nf, Nt)`
  folded. Slot `(i, j)` with `i >= 1` is band `i` at time `j`; row 0 holds band 0 (DC) at
  time `j` in its even slots and band `Nf` (Nyquist) at time `j - 1` in its odd ones.
  `domania.wilson.grid.cells(Nf, Nt)` returns each slot's band and time.
- Results are float64 (complex128 on frequency); a frequency result has exactly zero
  imaginary part at DC and even-`N` Nyquist. A frequency *input* may carry such a residue
  up to 1e-10 of its slice's peak; more is refused. enchilada's frequency data are exactly
  canonical (`check_rfft_endpoints`), so this never fires.
- Errors: `TypeError` for a wrong kind of argument, `ValueError` for a wrong value (unknown
  domain, missing or misplaced WDM grid, odd or non-positive `Nf`/`Nt`, `Nf * Nt != N`,
  non-finite input, complex input on time or WDM). enchilada validates grids and
  finiteness first, so these surface only on a programming error.
- Normalisation: atoms of unit norm, `sum(w**2) == dt * sum(x**2)`. Measured: white noise
  of variance σ² per sample gives `E[w**2] = σ² / fs` at every folded cell (mean ratio
  0.999 over 4000 draws on an 8 × 64 grid; DC slots 1.005, Nyquist slots 0.999). This is
  the constant enchilada's `white_covariance` already uses, now with no inactive cells.
- Cost: one 25,245,696-sample channel (a two-year Mojito record at 0.4 Hz on an
  Nf = 512 grid) transforms time → WDM in 2.9 s and back in 2.6 s, round-trip error
  6.3e-15. Memory is about four copies of the channel stack.
- `domania.wilson.forward_time(x, Nf, Nt, dt)` is the per-channel kernel the tests compare
  against.

## 4. The folded layout in enchilada

### 4.1 `enchilada/domains.py`

- `WDMGrid` keeps its two fields and their positive-even validation.
  `array_shape` becomes `(num_frequency_divisions, num_time_divisions)`.
  `num_time_samples` is unchanged. `active_mask` is deleted. The class docstring states
  the slot rule from section 3 and that every cell carries data.
- `Grid.channel_shape` returns `array_shape` for WDM. `Grid.active_mask` is deleted.
- `resolve_wdm_grid` is unchanged.
- `check_wdm_coefficients(channel_data, wdm_grid, context_label)` keeps three checks per
  channel: numpy array, shape equal to `array_shape`, real floating dtype. The
  inactive-zero check and its message go. The docstring says the inverse reads every cell.

### 4.2 `enchilada/data.py`

- `L1Data._resolve_and_check_num_time_samples` derives
  `WDMGrid(num_frequency_divisions=first.shape[0], num_time_divisions=first.shape[1])`.
- The `channel_data` field docstring describes WDM arrays as real coefficients shaped
  `(num_frequency_divisions, num_time_divisions)` in domania's folded layout, with the
  slot rule, and records the blind spot: both axes are even, so a transposed coefficient
  array is accepted as a grid with swapped divisions; a template checked against a known
  grid (`check_on_grid`) still fails on shape. The `wdm_grid` field docstring loses
  "inactive".

### 4.3 `enchilada/block_state.py`

`check_on_grid` is unchanged in logic; it calls the slimmer `check_wdm_coefficients`.

### 4.4 `enchilada/covariance.py`

`Covariance._validated_blocks` drops `active = grid.active_mask`, the
`blocks[:, :, ~active]` zero check and its message. The positivity check runs over every
point ("every channel's variance must be positive at every point"), and the batched
Cholesky runs over all points ("covariance must be positive definite at every point").
The class docstring loses "zero blocks at inactive WDM pixels".

### 4.5 `tests/conftest.py`

`white_covariance` fills every cell: `np.full(grid.channel_shape, scale * levels[ch])`
for each channel, with `scale = 1 / fs` on a WDM grid as before. Its docstring loses the
inactive sentence.

## 5. The adapter: `enchilada/_transforms.py`

- `import domania` at module level; domania is a core dependency. `_load_wdm`, the
  Python 3.13 gate and the `forward_many` / `inverse_many` probe are deleted.
- A module-level seam

  ```python
  def _wdm_transform(stacked, source_domain, target_domain, *, num_time_samples,
                     sample_interval_s, source_wdm_grid, target_wdm_grid):
      """domania.transform on the channel stack. Tests patch this to prove the
      grid checks run before any WDM transform."""
  ```

  maps each `WDMGrid` (or `None`) to `(Nf, Nt)` and calls `domania.transform(stacked,
  source_domain, target_domain, N=num_time_samples, dt=sample_interval_s,
  source_wdm=..., target_wdm=...)`.
- `transform_channels` keeps its order: validate the two domain names, require finite
  input, return copies when source and target agree grid included, stack the channels.
  Then, if `"wdm"` is the source or the target: require the corresponding `WDMGrid`
  (the existing `ValueError`s "WDM source requires wdm_grid" / "WDM target requires
  wdm_grid"), and call `_wdm_transform`. Otherwise the NumPy time ↔ frequency path is
  unchanged, including the complex64 widening. The output finiteness check and the
  per-channel copies stay.
- The docstring says WDM conversions go through domania and that the time ↔ frequency
  path is NumPy only.

## 6. Translation and Wheel

No API change. `transform(value, "wdm", num_frequency_divisions=..., num_time_divisions=...,
reference_data=...)`, `L1Data.to_time` / `to_frequency`, and
`Wheel.add(..., data_domain="wdm", num_frequency_divisions=...)` keep their signatures and
semantics; WDM → WDM regrids pass both grids to domania as they passed both to the old
backend. Docstring edits only:

- `translation.py`: "`num_frequency_divisions`: WDM frequency divisions (Nf, the row
  count)"; the sentences about the separately installed local backend and Python ≥ 3.13
  go; "the WDM backend's normalized-edge convention" becomes "domania's folded layout".
- `wheel.py` and `data.py`: no `Nf + 1` wording remains (grep for `+ 1` near WDM text).

## 7. Packaging

`pyproject.toml`:

```toml
dependencies = [
    # domania (the WDM transform) needs numpy>=2.1 and scipy>=1.14.1; scipy is also
    # imported directly by resample.py and orbits.py.
    "numpy>=2.1",
    "scipy>=1.14.1",
    "domania>=3.0,<4",
]

[project.optional-dependencies]
numeric-orbits = [
    "h5py>=3.12.1",       # first h5py with NumPy 2 wheels; lisaorbits' own floor
    "lisaorbits>=3.0.3",  # sets the >=3.12 python floor with domania
]
mojito = [
    "h5py>=3.12.1",       # load_mojito_l1 reads the L1 file
]
```

- The `requires-python` comment names domania and lisaorbits as the ≥ 3.12 floors.
- The `[[tool.mypy.overrides]]` comment stops saying scipy is lazily imported behind the
  extras; the override itself stays (scipy and h5py ship no type information).
- `resample.py` and `orbits.py` import scipy at module level; the three lazy imports and
  their "optional dependency" comments go. `mojito.py` keeps its lazy `h5py` import.
- `uv.lock` is regenerated with uv 0.12.1 (sandbox disabled; `uv lock --check` passes).
- Trial resolution with `--resolution lowest-direct` and both extras gives domania 3.0.0,
  scipy 1.14.1, h5py 3.12.1, lisaorbits 3.0.3 and numpy 2.1.x (verified 2026-11-16).

## 8. CI and release

`.github/workflows/ci.yml` and `.github/workflows/release.yml`, edited the same way:

- The `test` lanes keep `uv sync --locked --extra numeric-orbits --extra mojito`; domania
  arrives as core, and the WDM tests run on both operating systems and both Python
  versions.
- `core-only`: install line unchanged; the comment becomes "The core is numpy, scipy and
  domania. Prove it imports and tests on every supported interpreter; orbit and Mojito
  tests skip without h5py and lisaorbits."
- `core-minimum`: install `uv pip install --only-binary numpy --only-binary scipy
  --group dev -e . "numpy==2.1.0" "scipy==1.14.1" "domania==3.0.0"`; the verification asserts
  `numpy.__version__ == "2.1.0"`, imports domania and prints its version, and asserts
  `importlib.util.find_spec(name) is None` for `("h5py", "lisaorbits")` only.
- `floors`: unchanged (`--resolution lowest-direct --extra numeric-orbits --extra mojito`).
- The `wdm:` jobs, their `secrets:` blocks and `wdm` in `needs: [...]` are deleted.
- `.github/workflows/wdm.yml` is deleted. The README's link to it goes too
  (`tests/test_packaging_docs.py` checks README links resolve).
- The sdist lanes are unchanged.

## 9. Example: `examples/mojito_wdm_wheel.py`

The Wheel side of the Mojito story; `load_mojito.py` remains the data side. Needs the
`mojito` extra and a Mojito L1 file by `argv[1]` or `ENCHILADA_MOJITO_L1`, with the same
usage exit as `load_mojito.py`.

```python
SECONDS_PER_DAY = 86400.0

def fit_window(duration_s, *, dt, factor, num_frequency_divisions):
    """Longest window of at most duration_s whose record, decimated by factor, has
    num_frequency_divisions * num_time_divisions samples with num_time_divisions even.
    Returns (native_duration_s, num_time_divisions)."""
    samples_per_column = factor * num_frequency_divisions
    num_time_divisions = int(duration_s / (samples_per_column * dt))
    num_time_divisions -= num_time_divisions % 2
    if num_time_divisions < 2:
        raise ValueError(f"duration_s={duration_s} holds fewer than two WDM time "
                         f"columns of {samples_per_column * dt} s")
    return num_time_divisions * samples_per_column * dt, num_time_divisions
```

`main(path, *, num_frequency_divisions=512, factor=10, duration_s=7 * SECONDS_PER_DAY)`:

1. Probe the file's sampling with a cheap load, `load_mojito_l1(path, channels=("A",),
   duration_s=60.0, orbit=False)`, and read `dt` from it.
2. `native_duration_s, num_time_divisions = fit_window(duration_s, dt=dt, factor=factor,
   num_frequency_divisions=num_frequency_divisions)`.
3. `native = load_mojito_l1(path, duration_s=native_duration_s)`;
   `data = decimate(native, factor)`, which has exactly `Nf * Nt` samples.
4. `observed = transform(data, "wdm", num_frequency_divisions=num_frequency_divisions)`.
5. Print, one line each: the window (`native_duration_s`, as `factor × Nf × Nt` native
   samples at the file's `dt`); the grid (`Nf`, `Nt`, `dF = 1 / (2 Nf dt)` in Hz,
   `dT = Nf dt` in s, `observed.wdm_grid.array_shape`); per channel the RMS coefficient
   `"  {name}: rms {value:.3e} per WDM coefficient"`.
6. `wheel = Wheel(observed, random_seed=0)`; `wheel.add(EchoBlock(name="echo"),
   initial_block_state=BlockState())`; `wheel.run(num_cycles=1)`; print the residual's
   `data_domain` and `wdm_grid` to show the block was served on the WDM grid. Return the
   Wheel.

Defaults: Nf = 512 at 0.4 Hz gives dF = 3.906e-4 Hz and dT = 1280 s; a week gives
Nt = 472 and 2,416,640 native samples (604,160 s). Nf is a science choice the user may
change; the example takes it as a keyword.

## 10. Tests

### 10.1 Example tests, in `tests/test_examples.py`

- `test_mojito_wdm_wheel_example_builds_a_wdm_wheel`: `write_mojito_file(path, n=4096)`
  (1024 s at 4 Hz); `main(path, num_frequency_divisions=8, factor=4, duration_s=1000.0)`.
  `fit_window` gives Nt = 124 (int(1000 / 8) = 125, rounded down to even) and a native
  window of 992 s = 3968 samples; decimation by 4 yields 992 samples; the observation is
  `data_domain == "wdm"` with `wdm_grid == WDMGrid(num_frequency_divisions=8,
  num_time_divisions=124)` and coefficient shape `(8, 124)`; `wheel.residual()` is on
  that grid. The printed output carries "WDM grid", "per WDM coefficient" for each channel
  and the residual line. Running the script without a path exits non-zero with "usage".
  Skips without h5py.
- `test_mojito_wdm_wheel_example_on_the_real_file`: opt-in via `ENCHILADA_MOJITO_L1`,
  runs the defaults (a week, Nf = 512) and checks
  `wdm_grid == WDMGrid(num_frequency_divisions=512, num_time_divisions=472)` and that the
  observation's `start_time_gps` equals the file's `tdis/sampling` `t0`.
- `fit_window` unit cases: the 1000 s / dt 0.25 / factor 4 / Nf 8 case above; a
  duration that holds exactly two columns; a duration below two columns raises
  `ValueError`.

### 10.2 Layout change

- `tests/test_domains.py`: `array_shape == (4, 8)`; the active-mask tests and the
  "inactive" cases of `check_wdm_coefficients` go; against the `(4, 8)` grid the refused
  shapes are `(5, 8)`, `(4, 7)` and `(2, 16)`: the last is a well-formed folded shape
  that is not the grid's, so the shape check, not the parity check, must catch it.
- `tests/test_grid.py`: `channel_shape == (8, 8)`; the `active_mask` assertions go; the
  constructor-error cases are unchanged.
- `tests/test_data.py::TestWDMData`: valid arrays are `np.zeros(grid.array_shape)`;
  refused shapes become `(5, 8)`, `(4, 7)`, `(2, 16)` against a 64-sample record and the
  "inactive" invalid case goes; a new test pins the blind spot: a 64-sample `L1Data`
  built from `np.zeros((16, 4))` is accepted with
  `wdm_grid == WDMGrid(num_frequency_divisions=16, num_time_divisions=4)`, while a block
  state whose template is `(16, 4)` translated with `reference_data` on the `(4, 16)`
  grid is refused with "WDM shape".
- `tests/test_block_state.py`: the "inactive" case goes; the accepted signal is
  `np.ones(grid.array_shape)`.
- `tests/test_covariance.py`: `wdm_data` is built on `(4, 4)`;
  `test_time_and_wdm_blocks_must_be_real` fills the diagonal with `np.ones`;
  `test_inactive_wdm_pixels_must_be_entirely_zero` goes; the match string
  "positive definite at every active point" (line 193) becomes "positive definite at
  every point"; `test_wdm_structural_zeros`
  becomes a test that a WDM covariance holds its variance at every cell;
  `test_wdm_derivation_keeps_structural_zeros` becomes the AET derivation on folded
  cells, checked against the time-domain derivation value by value.
- `tests/test_translation_review.py:23` and `tests/test_wheel_translation.py:139`: the
  variance arrays are `np.ones(grid.array_shape)`.

### 10.3 Backend change

- `tests/test_translation.py`: `requires_wdm` and `import wdm` go; the round-trip test
  compares each channel to `domania.wilson.forward_time(x, 8, 8, dt)` with
  `assert_array_equal`, and the whole stack to `domania.transform(stack, "time", "wdm",
  N=64, dt=dt, target_wdm=(8, 8))`; regrid shapes become `(16, 4)`; the three tests on
  the old backend's import errors (`local install hint`, `dependency failure`,
  `unrelated package / Python`) are deleted; the white-noise variance test asserts
  σ² / fs at every cell with no mask.
- `tests/test_wheel_translation.py:223-239` and `:309-320`, and
  `tests/test_initial_block_state.py:236-250`: the sentinel is
  `monkeypatch.setattr("enchilada._transforms._wdm_transform", unexpected)`; the
  "backend unavailable" test in `test_initial_block_state.py` becomes "an error raised
  by the WDM transform propagates", raising from the same seam.
- `tests/test_toy_translation.py`, `tests/test_gb_example_contract.py`,
  `tests/test_translation_example.py`: `importorskip("wdm")` goes; the shapes report in
  `test_translation_example.py` is `[8, 8]` for both selectors.
- Every `pytest.importorskip("scipy", ...)` in `tests/` goes (scipy is core); the h5py
  and lisaorbits skips stay.

### 10.4 Gates

Full suite with coverage ≥ 95 % (`.venv/bin/python -m pytest -q --cov=enchilada`), ruff
check and format on `src tests examples benchmarks`, mypy clean, `uv lock --check` with
uv 0.12.1, and the two opt-in real-file tests passing locally with the measured
transform time reported in the PR.

## 11. Documentation

README:

- The install paragraph (line 44 area): the core depends on NumPy ≥ 2.1, SciPy ≥ 1.14.1
  and domania ≥ 3.0; `numeric-orbits` adds h5py and lisaorbits; `mojito` adds h5py;
  `decimate` needs no extra. The install block's comments follow.
- "Time, frequency, and WDM translation": rewritten for domania and the folded layout:
  coefficient arrays are `(num_frequency_divisions, num_time_divisions)`; the slot rule
  from section 3; a pointer to `domania.wilson.grid.cells` for each slot's band and
  time; the sentences about the local `wdm` checkout, Python ≥ 3.13 and the integration
  workflow go, and so does the link to `wdm.yml`.
- Covariance text: "squared WDM coefficients in WDM" stays; "(zero at the structurally
  inactive edge pixels)", "zero blocks at inactive WDM pixels" and
  "`(num_frequency_divisions + 1, num_time_divisions)`" go; "at every active point"
  becomes "at every point".
- The Mojito section gains a paragraph, "Fitting a window to a WDM grid": the record
  must have `Nf * Nt` samples after decimation, both even; choose Nf, take the largest
  even Nt with `factor * Nf * Nt * dt <= duration`, and pass `factor * Nf * Nt * dt` as
  `duration_s`; the full Mojito Light record (252,465,200 samples) does not factor onto
  any power-of-two Nf, so a campaign drops a few hundred to a few thousand samples
  (Nf = 512 at 0.4 Hz: 824 samples, 34 minutes); points at the example.
- The development block syncs the two extras; the CI paragraph describes the
  core-minimum lane at numpy 2.1.0, scipy 1.14.1 and domania 3.0.0.
- `examples/domain_translation.py` docstring and the notebook's setup and prose cells:
  no sibling `wdm` checkout; `uv sync` suffices; "inactive WDM edge" prose goes.

CHANGELOG 0.4.0:

- Changed — Breaking: core dependencies are `numpy>=2.1`, `scipy>=1.14.1` and
  `domania>=3.0,<4`; the WDM layout is domania's folded `(Nf, Nt)` and
  `WDMGrid.active_mask`, `Grid.active_mask` and the inactive-zero rules on coefficients
  and covariances are removed; the private `wdm` backend, its Python ≥ 3.13 requirement
  and the `wdm.yml` workflow are gone; the `mojito` extra is h5py only and `decimate`
  needs no extra.
- Added: `examples/mojito_wdm_wheel.py`.
- The unreleased `Grid` entry stops listing `active_mask`.

## 12. Non-goals

- Delivering a covariance to a block on another grid, noise components and
  `domania.wdm_covariance` / `frequency_covariance`: the components spec that follows.
- A window parameter on `WDMGrid` or `transform` (domania's default Meyer window only).
- Trimming or padding a record inside `transform` or the loader (decision 2).
- Guidance on choosing Nf beyond the example's default.
- Any change to the loader's clock, channels or gap handling.

## 13. Self-review

- Placeholders: none. Every file edit names its target and its new content.
- Consistency: the folded shape `(Nf, Nt)` is used in sections 3, 4, 6, 9, 10 and 11;
  the white-noise constant σ² / fs in sections 3, 4.5 and 10.3; the example's test
  numbers (Nt = 124, 992 samples, `(8, 124)`) in sections 9 and 10.1 follow from the
  `fit_window` code in section 9.
- Scope: one plan. The layout change, the adapter, packaging, CI, the example and the
  documentation are coupled by the shape change and land together.
- Ambiguity: `_wdm_transform` is the one seam tests patch; `fit_window` is the one place
  the window arithmetic lives; the probe load reads 60 s of one channel without the
  orbit.
