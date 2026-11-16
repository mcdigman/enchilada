# domania as the WDM Backend, Folded Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a campaign load Mojito L1 data, decimate it, translate it to the WDM domain with domania 3.0 and hand that WDM observation to a `Wheel`; along the way enchilada's WDM domain adopts domania's folded `(Nf, Nt)` layout and the private `wdm` package stops being a dependency of anything.

**Architecture:** Five coupled edits that land on one branch. (1) `WDMGrid.array_shape` becomes `(Nf, Nt)` and every "inactive cell" rule disappears from `domains.py`, `data.py`, `covariance.py` and the test fixtures. (2) `_transforms.py` imports domania at module level and routes every conversion with a WDM side through one seam, `_wdm_transform`, which tests patch; the NumPy time ↔ frequency path is untouched. (3) `pyproject.toml` makes domania, NumPy ≥ 2.1 and SciPy ≥ 1.14.1 core; the extras shrink to h5py (+ lisaorbits); the CI `wdm` jobs and workflow go. (4) `examples/mojito_wdm_wheel.py` fits a window to a WDM grid, loads, decimates, translates and runs a Wheel. (5) README and CHANGELOG describe the new layout and dependencies.

**Tech Stack:** Python 3.12+, NumPy ≥ 2.1, SciPy ≥ 1.14.1, domania 3.0.0 (`domania.transform`, `domania.wilson.forward_time`, `domania.wilson.grid.cells`), h5py (extra), pytest, ruff, mypy, uv 0.12.1. Tooling in this repo: `.venv/bin/python -m pytest|ruff|mypy`, never `uv run`; `uv lock` only with the sandbox disabled (controller step).

**User decisions (already made):**
- WDM layout is domania's folded `(Nf, Nt)`: every cell carries data; `WDMGrid.active_mask`, `Grid.active_mask` and every inactive-cell rule are deleted (user, 2026-11-16).
- The caller sizes the window so the decimated record has `Nf * Nt` samples; the arithmetic lives in the example and the README, not in `transform` or the loader (user, 2026-11-16).
- Backend is domania 3.0.0 from PyPI; the private `wdm` package is dropped entirely (user, 2026-11-16).
- domania is a **core** dependency, not an extra; with it NumPy ≥ 2.1 and SciPy ≥ 1.14.1 become core requirements (user, 2026-11-16).
- The WDM step comes after `load_mojito_l1` and `decimate`, as `transform(data, "wdm", ...)`; the loader stays time-domain.
- The example is `examples/mojito_wdm_wheel.py` (the Wheel side of the Mojito story); `load_mojito.py` stays the data side.

**Spec:** `docs/superpowers/specs/2026-11-16-domania-wdm-design.md`. **Branch:** `domania-wdm` (cut from `mojito-loader`, which is PR #7 on top of `covariance-simplification`, PR #6). Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Stage named paths only, never `git add -A`/`.`/`-u`.

**Corrections to the spec, found while planning (the plan is authoritative where they differ):**
- Spec §10.3 says the shapes report in `tests/test_translation_example.py` becomes `[8, 8]` for both selectors. The example uses 128 samples, so `num_frequency_divisions=8` gives `(8, 16)` and `num_time_divisions=8` gives `(16, 8)`. The notebook's cell `translation-06` asserts change the same way.
- Spec §10.2 lists `(2, 16)` among the shapes `tests/test_data.py` refuses with "positive even integer". Against a 64-sample `replace(observed, ...)` that shape derives a valid `WDMGrid(2, 16)` and fails the *product* check ("num_time_samples must equal the WDM grid division product"); it gets its own match string.
- Spec §10.2 omits `tests/test_wheel_translation.py:459` (a second `active_mask` site) and the row slice at `:140`, `variances[1:-1] *= np.arange(1, Nf)[:, None]`, which breaks on a folded `(8, 8)` array (6 rows, 7 multipliers); it becomes `variances[1:] *= ...`.
- Spec §11 puts the notebook edits under documentation. Cells `translation-06` and `translation-08` are *code* (`(9, 16)`/`(17, 8)` asserts, `~grid.active_mask`) and the notebook test stops skipping the moment `importorskip("wdm")` goes, so the whole notebook and the example docstring move into Task 2.
- domania refuses `source_wdm` on a non-WDM source and `target_wdm` on a non-WDM target (`ValueError: source_wdm is only for a 'wdm' source`). `_wdm_transform` passes a grid only for the WDM side(s).
- Module docstrings at `resample.py:10`, `orbits.py:13-14` and `mojito.py:18` still say the core is NumPy-only / scipy is lazy; Task 3 fixes them with the imports.

**Repo facts an implementer needs:**
- `WDMGrid(*, num_frequency_divisions, num_time_divisions)` is frozen and keyword-only; `Grid(data_domain, num_time_samples, sample_rate_hz, wdm_grid=None)` is frozen and positional. `L1Data` derives `wdm_grid` from the first channel's array shape; `replace(data, ...)` passes every field, including `num_time_samples`.
- `check_wdm_coefficients(channel_data, wdm_grid, context_label)` in `domains.py` is called by `L1Data._validate_channel_data_lengths` and by `block_state.check_on_grid`; its "must have WDM shape" message is what `transform(state, "wdm", reference_data=...)` raises for a template on the wrong grid.
- `transform_channels` in `_transforms.py` is the only caller of the backend; `translation.transform` passes `source_wdm_grid=source.wdm_grid` (None for non-WDM data) and `target_wdm_grid` from `_target_grid` (None for non-WDM targets).
- domania 3.0.0 is installed in `.venv` (numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2 also present). Measured: `domania.transform(stack, "time", "wdm", N=64, dt=dt, target_wdm=(8, 8))[i]` equals `domania.wilson.forward_time(stack[i], 8, 8, dt)` bit for bit; the frequency route `(dt * rfft) / dt` is exact for power-of-two `dt` only. `domania.transform` accepts float32 time and complex64 frequency input and returns float64 / complex128; WDM → frequency output has exactly zero imaginary part at DC and Nyquist. Same-grid WDM → WDM returns a copy; a different `target_wdm` regrids.
- `domania.wilson.grid.cells(4, 4)` returns bands `[[0,4,0,4],[1,1,1,1],[2,2,2,2],[3,3,3,3]]` and times `[[0,0,2,2],[0,1,2,3],...]`: row 0 even slots are DC at time `j`, odd slots Nyquist at time `j - 1`.
- White noise of variance σ² per sample gives `E[w**2] = σ² / fs` at every folded cell (measured ratio 0.999 overall, 1.005 on the DC slots, 0.999 on the Nyquist slots).
- `tests/conftest.py` is imported as `from conftest import ...`; it must stay importable without h5py (import h5py inside functions). `write_mojito_file(path, n=4096)` writes a Mojito-layout file of `n` samples at `MOJITO_DT = 0.25` starting at `MOJITO_T0`.
- `tests/test_docs.py` compiles every ```` ```python ```` block in README, checks keyword names of calls into `_CALLABLES`, and requires every `enchilada.X` in README prose to be an attribute of the package. `tests/test_packaging_docs.py` requires every local README link target to exist.
- `examples/domain_translation.ipynb` round-trips through `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"` byte for byte; edit cells by `id` with a script, never by hand. Never run `ruff format` on a notebook (it is checked, not rewritten).
- Running notebooks or anything that imports matplotlib needs `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl` (the default cache dir is not writable in the sandbox).
- The two opt-in real-file tests read `ENCHILADA_MOJITO_L1`; without it they skip.

**Grep inventory, every hit assigned to one task** (run `grep -rn "active_mask\|requires_wdm\|importorskip(\"wdm\|importorskip(\"scipy\|>=3\.13\|≥3\.13\|--python 3\.13\|NumPy-only\|requires only NumPy\|depends only on NumPy\|scipy lazily\|from scipy.*lazy" src tests examples benchmarks README.md` to re-check at the end):

| Site | Task |
|---|---|
| `src/enchilada/domains.py` (`active_mask` ×3, inactive check) | 1 |
| `src/enchilada/data.py:29-30, 207-213` | 1 |
| `src/enchilada/covariance.py:106-109, 165-177` | 1 |
| `src/enchilada/_transforms.py:105` (`active_mask` zeroing) | 1 (line deleted); whole module rewritten in 2 |
| `tests/conftest.py:69` | 1 |
| `tests/test_domains.py`, `tests/test_grid.py:22,30,36-37`, `tests/test_data.py:84-166`, `tests/test_block_state.py:12-52`, `tests/test_covariance.py:33-42,126-131,172-187,193,288-293,434-445`, `tests/test_translation_review.py:23`, `tests/test_wheel_translation.py:139-140,459` | 1 |
| `src/enchilada/_transforms.py` (`_load_wdm`, Python 3.13 gate), `src/enchilada/translation.py:74,83-87` | 2 |
| `tests/test_translation.py` (`requires_wdm`, `import wdm`, three backend-error tests, `active_mask` at 186 and 313), `tests/test_wheel_translation.py:7,17-19,90,93,126,223-239,250-262,309-320`, `tests/test_initial_block_state.py:3,24-26,121-122,237-251`, `tests/test_translation_review.py:3,91-103`, `tests/test_toy_translation.py:25-26`, `tests/test_gb_example_contract.py:133-134`, `tests/test_translation_example.py:11,26-29` | 2 |
| `examples/domain_translation.py:1-10`, `examples/domain_translation.ipynb` cells `translation-01,-05,-06,-07,-08` | 2 |
| `pyproject.toml`, `src/enchilada/resample.py:10,37,87`, `src/enchilada/orbits.py:13-14,118`, `src/enchilada/mojito.py:18`, `tests/test_orbits.py:6`, `tests/test_resample.py:8-11`, `tests/test_mojito.py:9`, `tests/test_examples.py:56,346`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `.github/workflows/wdm.yml` (delete), `README.md:436-449`, `uv.lock` | 3 |
| `examples/mojito_wdm_wheel.py` (new), `tests/test_examples.py` (new tests) | 4 |
| `README.md:44-58,292-293,335-339,428-434,579-580,624-625, new paragraph after 661, 692-700`, `CHANGELOG.md` | 5 |

---

### Task 1: Folded `(Nf, Nt)` WDM layout

**Goal:** `WDMGrid.array_shape` is `(num_frequency_divisions, num_time_divisions)`, no `active_mask` exists anywhere, `check_wdm_coefficients` and `Covariance` validate every cell alike, and every layout test and fixture is on the folded shape.

**Files:**
- Modify: `src/enchilada/domains.py` (module docstring, `WDMGrid` docstring and `array_shape`, delete `WDMGrid.active_mask` and `Grid.active_mask`, `check_wdm_coefficients`)
- Modify: `src/enchilada/data.py:22-30` (field docstring), `:207-213` (grid derivation)
- Modify: `src/enchilada/covariance.py:106-109` (class docstring), `:165-177` (`_validated_blocks` tail)
- Modify: `src/enchilada/_transforms.py:105` (delete the `active_mask` zeroing line; Task 2 rewrites the module)
- Modify: `tests/conftest.py:45-72` (`white_covariance`)
- Rewrite: `tests/test_domains.py`
- Modify: `tests/test_grid.py:19-37`, `tests/test_data.py:124-166`, `tests/test_block_state.py:12-52`, `tests/test_covariance.py:33-42,126-131,172-187,193,288-293,434-445`, `tests/test_translation_review.py:23`, `tests/test_wheel_translation.py:139-140,459`

**Acceptance Criteria:**
- [ ] `WDMGrid(num_frequency_divisions=4, num_time_divisions=8).array_shape == (4, 8)`; neither `WDMGrid` nor `Grid` has an `active_mask` attribute; `Grid("wdm", 64, 1.0, WDMGrid(8, 8)).channel_shape == (8, 8)`.
- [ ] `check_wdm_coefficients` accepts any real floating `(4, 8)` array on that grid, including nonzero odd slots of row 0, and refuses `(5, 8)`, `(4, 7)`, `(2, 16)` and `(32,)` with `ValueError` matching "shape", complex with `TypeError` "real", integer with `TypeError` "floating", a nested list with `TypeError` "numpy array".
- [ ] A 64-sample `L1Data` built from `(4, 16)` arrays derives `WDMGrid(4, 16)`; from `(16, 4)` arrays it derives `WDMGrid(16, 4)`; a `(16, 4)` template translated with `reference_data` on the `(4, 16)` grid is refused with "WDM shape".
- [ ] `Covariance` on a WDM grid requires a positive variance at every cell (message "every channel's variance must be positive at every point") and positive definiteness at every cell ("covariance must be positive definite at every point"); the AET derivation of constant XYZ levels on a WDM grid equals the time-domain derivation at every cell.
- [ ] `white_covariance` on WDM data fills every cell with `sigma**2 / fs`.
- [ ] `grep -rn active_mask src tests` prints nothing.

**Verify:** `.venv/bin/python -m pytest -q` → all pass (WDM-backend tests still skip: `wdm` is not installed); `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy` → clean.

**Steps:**

- [ ] **Step 1: Rewrite `tests/test_domains.py` (the failing tests first)**

Replace the whole file with:

```python
"""WDM grids describe domania's folded coefficient layout without a transform."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_wdm_grid_is_keyword_only_and_immutable():
    from enchilada.domains import WDMGrid

    with pytest.raises(TypeError):
        WDMGrid(4, 8)
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    with pytest.raises(FrozenInstanceError):
        grid.num_frequency_divisions = 8
    assert grid.array_shape == (4, 8)
    assert grid.num_time_samples == 32


@pytest.mark.parametrize("name", ["num_frequency_divisions", "num_time_divisions"])
@pytest.mark.parametrize("value", [0, -2, 3, True, False, 4.0, np.float64(4.0)])
def test_wdm_grid_rejects_invalid_divisions(name, value):
    from enchilada.domains import WDMGrid

    arguments = {"num_frequency_divisions": 4, "num_time_divisions": 8, name: value}
    with pytest.raises(ValueError, match=name):
        WDMGrid(**arguments)


def test_numpy_integer_grid_counts_are_normalised_and_every_cell_is_data():
    from enchilada.domains import Grid, WDMGrid

    grid = WDMGrid(num_frequency_divisions=np.int64(4), num_time_divisions=np.int32(8))
    assert type(grid.num_frequency_divisions) is int
    assert type(grid.num_time_divisions) is int
    assert grid.array_shape == (4, 8)
    assert int(np.prod(grid.array_shape)) == grid.num_time_samples
    assert not hasattr(grid, "active_mask")
    assert not hasattr(Grid("wdm", 32, 1.0, grid), "active_mask")


@pytest.mark.parametrize(
    "arguments",
    [
        {"num_frequency_divisions": 4},
        {"num_time_divisions": 8},
        {"num_frequency_divisions": 4, "num_time_divisions": 8},
    ],
)
def test_resolve_wdm_grid_derives_only_the_missing_division(arguments):
    from enchilada.domains import WDMGrid, resolve_wdm_grid

    assert resolve_wdm_grid(32, **arguments) == WDMGrid(
        num_frequency_divisions=4, num_time_divisions=8
    )


@pytest.mark.parametrize(
    "num_time_samples,arguments",
    [
        (32, {}),
        (32, {"num_frequency_divisions": 4, "num_time_divisions": 4}),
        (32, {"num_frequency_divisions": 6}),
        (12, {"num_frequency_divisions": 4}),
        (32, {"num_frequency_divisions": True}),
        (32, {"num_time_divisions": 4.0}),
        (0, {"num_frequency_divisions": 4}),
        (-32, {"num_frequency_divisions": 4}),
        (True, {"num_frequency_divisions": 4}),
        (32.0, {"num_frequency_divisions": 4}),
    ],
)
def test_resolve_wdm_grid_rejects_ambiguous_or_incompatible_counts(
    num_time_samples, arguments
):
    from enchilada.domains import resolve_wdm_grid

    with pytest.raises(ValueError):
        resolve_wdm_grid(num_time_samples, **arguments)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wdm_coefficients_keep_values_and_input_buffers(dtype):
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    coefficients = np.arange(grid.num_time_samples, dtype=dtype).reshape(
        grid.array_shape
    )
    original = coefficients.copy()
    check_wdm_coefficients({"A": coefficients}, grid, "data")
    np.testing.assert_array_equal(coefficients, original)


def test_row_zero_odd_slots_carry_nyquist_data_and_are_accepted():
    """In the folded layout row 0 interleaves DC (even slots) with Nyquist (odd
    slots); there is no structural zero to enforce."""
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    coefficients = np.zeros(grid.array_shape)
    coefficients[0, 1::2] = np.nextafter(0.0, 1.0)
    check_wdm_coefficients({"A": coefficients}, grid, "data")


@pytest.mark.parametrize(
    "coefficients,error,match",
    [
        (np.zeros((5, 8)), ValueError, "shape"),
        (np.zeros((4, 7)), ValueError, "shape"),
        (np.zeros((2, 16)), ValueError, "shape"),  # well formed, but not this grid
        (np.zeros(32), ValueError, "shape"),
        (np.zeros((4, 8), complex), TypeError, "real"),
        (np.zeros((4, 8), int), TypeError, "floating"),
        ([[0.0] * 8] * 4, TypeError, "numpy array"),
    ],
)
def test_wdm_coefficients_reject_data_off_the_grid(coefficients, error, match):
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    with pytest.raises(error, match=match):
        check_wdm_coefficients({"A": coefficients}, grid, "data")
```

- [ ] **Step 2: Run the domains tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_domains.py -q`
Expected: failures in `test_wdm_grid_is_keyword_only_and_immutable` (`(5, 8) != (4, 8)`), `..._every_cell_is_data` (`hasattr` true), `..._keep_values_and_input_buffers` (shape), `test_row_zero_odd_slots...` ("inactive"), and the `(5, 8)` / complex / int cases of `..._reject_data_off_the_grid`.

- [ ] **Step 3: Fold the layout in `src/enchilada/domains.py`**

Line 1, module docstring:

```python
"""Representation metadata: domains, the WDM grid and the checks on its arrays."""
```

Replace the `WDMGrid` class docstring (lines 26-32) and `array_shape` (lines 43-46), and delete the `active_mask` property (lines 52-57), so the class reads:

```python
@dataclass(frozen=True, kw_only=True)
class WDMGrid:
    """The WDM frequency and time divisions of an unchanged observation.

    Both division counts are positive even integers. A coefficient array has
    shape ``(num_frequency_divisions, num_time_divisions)`` in domania's folded
    layout, and every cell carries data: slot ``(i, j)`` with ``i >= 1`` is
    band ``i`` at time ``j``; row 0 holds band 0 (DC) at time ``j`` in its even
    slots and band ``num_frequency_divisions`` (Nyquist) at time ``j - 1`` in
    its odd slots (``domania.wilson.grid.cells`` lists each slot's band and
    time). Sampling interval and epoch belong to ``L1Data``.
    """

    num_frequency_divisions: int
    num_time_divisions: int

    def __post_init__(self) -> None:
        for name in ("num_frequency_divisions", "num_time_divisions"):
            object.__setattr__(
                self, name, _positive_even_integer(getattr(self, name), name)
            )

    @property
    def array_shape(self) -> tuple[int, int]:
        """Frequency rows followed by time columns: ``(Nf, Nt)``."""
        return self.num_frequency_divisions, self.num_time_divisions

    @property
    def num_time_samples(self) -> int:
        """Number of samples in the underlying real time series."""
        return self.num_frequency_divisions * self.num_time_divisions
```

Delete `Grid.active_mask` (lines 121-126, the property between `channel_shape` and `observation_duration_s`).

Replace `check_wdm_coefficients` (lines 192-217) with:

```python
def check_wdm_coefficients(
    channel_data: dict[str, np.ndarray],
    wdm_grid: WDMGrid,
    context_label: str,
) -> None:
    """Require real floating coefficient arrays of the grid's folded shape.

    The inverse transform reads every cell, so there is no structural zero to
    enforce. Finiteness remains the responsibility of the Wheel and transform
    boundaries.
    """
    for channel_name, coefficients in channel_data.items():
        label = f"{context_label}[{channel_name!r}]"
        if not isinstance(coefficients, np.ndarray):
            raise TypeError(f"{label} must be a numpy array")
        if coefficients.shape != wdm_grid.array_shape:
            raise ValueError(
                f"{label} must have WDM shape {wdm_grid.array_shape}, "
                f"got {coefficients.shape}"
            )
        if not np.issubdtype(coefficients.dtype, np.floating):
            raise TypeError(f"{label} WDM coefficients must be real floating values")
```

- [ ] **Step 4: Derive the grid from the folded shape in `src/enchilada/data.py`**

Lines 22-30 of the `channel_data` field docstring become:

```
        channel_data: Channel name -> real or complex floating-point array, with
            keys matching `channel_names`. Time arrays contain `num_time_samples`
            real values.
            Frequency arrays contain `num_time_samples // 2 + 1` complex values
            with normalization ``dt * rfft(x)``; DC and even-length Nyquist are
            real. WDM arrays contain real coefficients shaped
            ``(num_frequency_divisions, num_time_divisions)`` in domania's
            folded layout: slot ``(i, j)`` with ``i >= 1`` is band ``i`` at time
            ``j``; row 0 holds band 0 (DC) at time ``j`` in its even slots and
            the Nyquist band at time ``j - 1`` in its odd slots. Both axes are
            even, so a transposed coefficient array is accepted as a grid with
            swapped divisions; a template checked against a known grid
            (`check_on_grid`) still fails on shape.
```

In `_resolve_and_check_num_time_samples` (about line 210) change the derivation to:

```python
        if self.data_domain == "wdm":
            grid = WDMGrid(
                num_frequency_divisions=first.shape[0],
                num_time_divisions=first.shape[1],
            )
```

- [ ] **Step 5: Validate every cell in `src/enchilada/covariance.py`**

Class docstring lines 106-109 become:

```
    Construction validates channel names, the grid, dtype, shape, finiteness,
    exact Hermitian symmetry, realness where the domain requires it, positive
    variances at every point and positive definiteness at every point. The
    array is stored as a fresh read-only copy. Equality is identity; compare
    arrays explicitly.
```

In `_validated_blocks`, replace everything from `variances = np.real(...)` to `return blocks` (lines 165-178) with:

```python
        variances = np.real(np.diagonal(blocks, axis1=0, axis2=1))  # (*S, K)
        if np.any(variances <= 0):
            raise ValueError(
                "every channel's variance must be positive at every point"
            )
        matrices = np.moveaxis(blocks, (0, 1), (-2, -1))  # (*S, K, K)
        try:
            np.linalg.cholesky(matrices)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "covariance must be positive definite at every point"
            ) from exc
        blocks.setflags(write=False)
        return blocks
```

- [ ] **Step 6: Drop the zeroing line from `src/enchilada/_transforms.py`**

Delete line 105, `        converted[:, ~target_wdm_grid.active_mask] = 0.0` (the only reference to the deleted attribute; Task 2 rewrites this module). Leave everything else.

- [ ] **Step 7: Fill every cell in `tests/conftest.py::white_covariance`**

Replace lines 45-72 with:

```python
def white_covariance(data, variance):
    """White noise of `variance` per time sample, on `data`'s grid.

    `variance` is one number for every channel or a channel -> number mapping.
    With X = dt * rfft(x) a stationary one-sided PSD S gives bin variance
    C = (Tobs / 2) S at every rfft bin; white noise has S = 2 sigma^2 / fs, so
    C = N sigma^2 / fs^2. On a WDM grid every cell has variance sigma^2 / fs.
    """
    grid = data.grid
    if grid.data_domain == "time":
        scale = 1.0
    elif grid.data_domain == "frequency":
        scale = grid.num_time_samples / grid.sample_rate_hz**2
    else:
        scale = 1.0 / grid.sample_rate_hz
    levels = (
        variance
        if isinstance(variance, Mapping)
        else {ch: variance for ch in data.channel_names}
    )
    return Covariance.like(
        data,
        {
            ch: np.full(grid.channel_shape, scale * levels[ch])
            for ch in data.channel_names
        },
    )
```

- [ ] **Step 8: Update `tests/test_grid.py`**

Delete line 22 (`assert grid.active_mask.shape == (64,) and grid.active_mask.all()`) and line 30 (`assert grid.active_mask.shape == (bins,) and grid.active_mask.all()`). Replace `test_wdm_grid` (lines 32-37) with:

```python
    def test_wdm_grid(self):
        wdm = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
        grid = Grid("wdm", 64, 1.0, wdm)
        assert grid.channel_shape == (8, 8)
        assert grid.wdm_grid is wdm
```

- [ ] **Step 9: Update `tests/test_data.py::TestWDMData`**

Replace `test_inferred_grid_requires_positive_even_divisions` (lines 124-128) with:

```python
    @pytest.mark.parametrize(
        ("shape", "match"),
        [
            ((5, 8), "positive even integer"),
            ((4, 7), "positive even integer"),
            ((2, 16), "WDM grid division product"),  # well formed, 32 != 64 samples
        ],
    )
    def test_inferred_grid_must_be_even_and_match_the_record(
        self, observed, shape, match
    ):
        channels = {name: np.zeros(shape) for name in observed.channel_names}
        with pytest.raises(ValueError, match=match):
            replace(observed, data_domain="wdm", channel_data=channels)
```

In `test_transform_rejects_buffers_that_no_longer_match_the_grid` (lines 130-146) change `np.zeros((5, 16))` to `np.zeros((4, 16))` and `np.zeros((9, 8))` to `np.zeros((8, 8))`.

Replace `test_wdm_validates_each_channel` (lines 148-166) with:

```python
    @pytest.mark.parametrize("invalid", ["shape", "complex", "integer"])
    def test_wdm_validates_each_channel(self, observed, invalid):
        from enchilada.domains import WDMGrid

        grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
        channels = {ch: np.zeros(grid.array_shape) for ch in observed.channel_names}
        error = ValueError
        if invalid == "shape":
            channels["E"] = np.zeros((8, 8))
        elif invalid == "complex":
            channels["E"] = channels["E"].astype(complex)
            error = TypeError
        else:
            channels["E"] = channels["E"].astype(int)
            error = TypeError
        with pytest.raises(error, match="E"):
            replace(observed, data_domain="wdm", channel_data=channels)

    def test_transposed_coefficients_pass_as_a_swapped_grid_until_checked_on_one(
        self, observed
    ):
        """Both axes are even, so L1Data cannot tell (16, 4) from (4, 16); a
        template checked against a known grid still fails on shape."""
        from enchilada.domains import WDMGrid

        swapped = replace(
            observed,
            data_domain="wdm",
            channel_data={name: np.zeros((16, 4)) for name in observed.channel_names},
        )
        assert swapped.wdm_grid == WDMGrid(
            num_frequency_divisions=16, num_time_divisions=4
        )
        reference = replace(
            observed,
            data_domain="wdm",
            channel_data={name: np.zeros((4, 16)) for name in observed.channel_names},
        )
        state = BlockState(template=dict(swapped.channel_data))
        with pytest.raises(ValueError, match="WDM shape"):
            transform(state, "wdm", reference_data=reference)
```

- [ ] **Step 10: Update `tests/test_block_state.py` (lines 12-52)**

```python
def test_wdm_result_factory_accepts_a_full_grid_and_zero_factory_matches_grid(
    observed,
):
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    observed = replace(
        observed,
        data_domain="wdm",
        channel_data={ch: np.zeros(grid.array_shape) for ch in observed.channel_names},
    )
    signal = {ch: np.ones(grid.array_shape) for ch in observed.channel_names}
    result = observed.block_state(signal, parameters={"amplitude": 1})
    assert result.parameters == {"amplitude": 1}
    np.testing.assert_array_equal(result.template["A"], signal["A"])
    zero = observed.zero_block_state()
    assert zero.template["A"].shape == grid.array_shape
    np.testing.assert_array_equal(zero.template["A"], 0.0)
    assert observed.block_state().template is None


@pytest.mark.parametrize("invalid", ["shape", "complex"])
def test_wdm_result_factory_rejects_incompatible_signal(observed, invalid):
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    observed = replace(
        observed,
        data_domain="wdm",
        channel_data={ch: np.zeros(grid.array_shape) for ch in observed.channel_names},
    )
    signal = {ch: np.zeros(grid.array_shape) for ch in observed.channel_names}
    error = ValueError
    if invalid == "shape":
        signal["A"] = np.zeros((8, 8))
    else:
        signal["A"] = signal["A"].astype(complex)
        error = TypeError
    with pytest.raises(error, match="A"):
        observed.block_state(signal)
```

- [ ] **Step 11: Update `tests/test_covariance.py`**

Line 34 comment: `    # Built by hand on the folded (Nf, Nt) layout; no transform needed.`

Line 129 (`test_time_and_wdm_blocks_must_be_real`): `        blocks[0, 0] = blocks[1, 1] = 1.0`

Replace `test_inactive_wdm_pixels_must_be_entirely_zero` (lines 172-187) with:

```python
    def test_wdm_variances_are_required_at_every_cell(self):
        data = wdm_data()
        blocks = stacked(data.grid.channel_shape, (1.0, 2.0))
        cov = Covariance(blocks, channel_names=data.channel_names, grid=data.grid)
        np.testing.assert_array_equal(cov.A, 1.0)
        blocks[0, 0, 0, 1] = 0.0  # row 0, odd slot: a Nyquist cell, not padding
        with pytest.raises(ValueError, match="variance must be positive at every point"):
            Covariance(blocks, channel_names=data.channel_names, grid=data.grid)
```

Line 193: `match="positive definite at every point"`.

Replace `test_wdm_structural_zeros` (lines 288-293) with:

```python
    def test_wdm_variance_at_every_cell(self):
        data = wdm_data()
        variances = np.full(data.grid.channel_shape, 3.0)
        cov = Covariance.like(data, {"A": variances, "E": variances})
        np.testing.assert_array_equal(cov.A, 3.0)
        assert cov.grid.data_domain == "wdm"
```

Replace `test_wdm_derivation_keeps_structural_zeros` (lines 434-445) with:

```python
    def test_wdm_derivation_matches_the_time_domain_derivation_cell_by_cell(self):
        levels = dict(zip(XYZ, (1.0, 2.0, 3.0), strict=True))
        wdm = wdm_data(channel_names=XYZ)
        aet_wdm = Covariance.like(
            wdm,
            {
                name: np.full(wdm.grid.channel_shape, level)
                for name, level in levels.items()
            },
        ).in_basis("AET")
        time = time_data(16, channel_names=XYZ)
        aet_time = Covariance.like(
            time, {name: np.full(16, level) for name, level in levels.items()}
        ).in_basis("AET")
        assert aet_wdm.grid == wdm.grid
        expected = aet_time.blocks[:, :, 0][:, :, None, None]
        np.testing.assert_allclose(
            aet_wdm.blocks, np.broadcast_to(expected, aet_wdm.blocks.shape)
        )
```

- [ ] **Step 12: Update the two remaining `active_mask` sites**

`tests/test_translation_review.py:23`: `    variances = np.ones(grid.array_shape)`

`tests/test_wheel_translation.py:139-140`:

```python
    variances = np.ones(grid.array_shape)
    variances[1:] *= np.arange(1, grid.num_frequency_divisions)[:, None]
```

`tests/test_wheel_translation.py:459`: `    variances = np.ones(reference.grid.channel_shape)`

- [ ] **Step 13: Run the gates**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass; the `requires_wdm` tests and `tests/test_translation_example.py` still skip (`wdm` is not installed).

Run: `grep -rn active_mask src tests examples benchmarks | grep -v ipynb`
Expected: no output (the notebook's `active_mask` is Task 2's).

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy`
Expected: clean.

- [ ] **Step 14: Commit**

```bash
git add src/enchilada/domains.py src/enchilada/data.py src/enchilada/covariance.py src/enchilada/_transforms.py tests/conftest.py tests/test_domains.py tests/test_grid.py tests/test_data.py tests/test_block_state.py tests/test_covariance.py tests/test_translation_review.py tests/test_wheel_translation.py
git commit -m "WDM arrays adopt domania's folded (Nf, Nt) layout; active_mask and the inactive-cell rules go

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: WDM conversions through domania

**Goal:** `_transforms.py` imports domania at module level and routes every conversion with a WDM side through the seam `_wdm_transform`; the translation tests, the translation walkthrough example and its notebook run against domania with no `wdm` skip anywhere.

**Files:**
- Rewrite: `src/enchilada/_transforms.py`
- Modify: `src/enchilada/translation.py:74,83-87` (docstring)
- Modify: `tests/test_translation.py` (header; round-trip, regrid and white-noise tests; delete the three backend-error tests; add the seam test)
- Modify: `tests/test_wheel_translation.py:7,17-19,90,93,126,223-239,250-262,309-320`
- Modify: `tests/test_initial_block_state.py:3,24-26,121-122,237-251`
- Modify: `tests/test_translation_review.py:3,91-103`
- Modify: `tests/test_toy_translation.py:25-26`, `tests/test_gb_example_contract.py:133-134`, `tests/test_translation_example.py:11,26-29,40`
- Modify: `examples/domain_translation.py:1-10` (docstring)
- Modify: `examples/domain_translation.ipynb` cells `translation-01`, `-05`, `-06`, `-07`, `-08` (by id, with the script below)

**Acceptance Criteria:**
- [ ] `transform(data, "wdm", num_frequency_divisions=8)` on a 64-sample two-channel record returns `(8, 8)` float64 arrays equal, bit for bit, to `domania.wilson.forward_time(x, 8, 8, dt)` per channel and to `domania.transform(stack, "time", "wdm", N=64, dt=dt, target_wdm=(8, 8))` for the stack (time source); the frequency source agrees to `atol=1e-13`; every route back to time or frequency agrees with the direct transform to `1e-12`.
- [ ] `transform(first, "wdm", num_time_divisions=4)` regrids `(8, 8)` coefficients to `(16, 4)` and equals the direct `(16, 4)` transform.
- [ ] `transform_channels` raises "WDM source requires wdm_grid" / "WDM target requires wdm_grid" before `_wdm_transform` is called; a WDM conversion calls `_wdm_transform` exactly once with the `(channels, N)` stack, both domain names, `num_time_samples`, `sample_interval_s` and both grids.
- [ ] White noise of variance σ² per sample has mean squared coefficient `σ² / fs` at every folded cell (ratio within 0.05 overall, within 0.1 on row 0).
- [ ] `grep -rn "requires_wdm\|importorskip(\"wdm\|import wdm\|_load_wdm\|>=3\.13\|≥3\.13\|--python 3\.13" src tests examples` prints nothing; `python examples/domain_translation.py` reports `wdm_shapes == {"frequency_selector": [8, 16], "time_selector": [16, 8]}`; the notebook's code cells run under `MPLBACKEND=Agg`.
- [ ] `.venv/bin/python -m mypy` is clean with `import domania` (domania ships `py.typed`).

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q` → all pass with no WDM skips (only Eryn/GB and the opt-in real-file tests skip); `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy` → clean.

**Steps:**

- [ ] **Step 1: Rewrite the backend tests in `tests/test_translation.py`**

Header (lines 1-15) becomes:

```python
"""Object conversion preserves physical grids, numerical values and owned state."""

from dataclasses import replace

import domania
import numpy as np
import pytest

from conftest import white_covariance
from enchilada import BlockState, L1Data, WDMGrid
from enchilada.translation import transform
```

Replace `test_actual_wdm_backend_and_all_round_trip_routes` (lines 171-204) with:

```python
@pytest.mark.parametrize("source_domain", ["time", "frequency"])
@pytest.mark.parametrize("sample_interval_s", [0.125, 10.0])
def test_domania_wdm_coefficients_and_all_round_trip_routes(
    source_domain, sample_interval_s
):
    data = observations(sample_interval_s=sample_interval_s)
    source = transform(data, source_domain)
    converted = transform(source, "wdm", num_frequency_divisions=8)
    assert converted.wdm_grid == WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    stack = np.stack([data.channel_data[name] for name in data.channel_names])
    expected_stack = domania.transform(
        stack, "time", "wdm", N=64, dt=sample_interval_s, target_wdm=(8, 8)
    )
    for index, name in enumerate(data.channel_names):
        expected = domania.wilson.forward_time(
            data.channel_data[name], 8, 8, sample_interval_s
        )
        np.testing.assert_array_equal(expected_stack[index], expected)
        assert converted.channel_data[name].shape == (8, 8)
        assert converted.channel_data[name].dtype == np.float64
        if source_domain == "time":
            np.testing.assert_array_equal(converted.channel_data[name], expected)
        else:
            # (dt * rfft) / dt is exact only for power-of-two dt.
            np.testing.assert_allclose(
                converted.channel_data[name], expected, rtol=0, atol=1e-13
            )
    for target in ("time", "frequency"):
        restored = transform(converted, target)
        expected = transform(data, target)
        assert restored.wdm_grid is None
        for name in data.channel_names:
            np.testing.assert_allclose(
                restored.channel_data[name], expected.channel_data[name], atol=1e-12
            )
    np.testing.assert_allclose(
        converted.to_time().channel_data["A"], data.channel_data["A"], atol=1e-14
    )
    np.testing.assert_allclose(
        converted.to_frequency().channel_data["A"],
        transform(data, "frequency").channel_data["A"],
        atol=1e-12,
    )
```

In `test_wdm_division_selectors_regrid_without_resampling` (lines 207-240) delete the `@requires_wdm` line and change `(17, 4)` to `(16, 4)`.

Delete `test_missing_wdm_backend_has_a_local_install_hint` (line 244), `test_backend_dependency_failure_is_not_reported_as_missing_wdm` (line 268) and `test_unrelated_wdm_package_and_unsupported_python_have_clear_errors` (line 280). Keep `test_first_wdm_conversion_requires_a_valid_division_count` and `test_joint_wdm_result_converts_signal_with_a_wdm_reference`.

Add, after `test_joint_wdm_result_converts_signal_with_a_wdm_reference`:

```python
def test_wdm_conversions_reach_domania_only_after_the_grid_checks(monkeypatch):
    import enchilada._transforms as transforms

    calls = []

    def recording(stacked, source_domain, target_domain, **options):
        calls.append((stacked.shape, source_domain, target_domain, options))
        return np.zeros((stacked.shape[0], 8, 8))

    monkeypatch.setattr(transforms, "_wdm_transform", recording)
    data = observations()
    with pytest.raises(ValueError, match="WDM target requires wdm_grid"):
        transforms.transform_channels(
            data.channel_data,
            source_domain="time",
            target_domain="wdm",
            num_time_samples=64,
            sample_rate_hz=4.0,
        )
    with pytest.raises(ValueError, match="WDM source requires wdm_grid"):
        transforms.transform_channels(
            {"A": np.zeros((8, 8))},
            source_domain="wdm",
            target_domain="time",
            num_time_samples=64,
            sample_rate_hz=4.0,
        )
    assert calls == []
    converted = transform(data, "wdm", num_frequency_divisions=8)
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    assert converted.wdm_grid == grid
    assert calls == [
        (
            (2, 64),
            "time",
            "wdm",
            {
                "num_time_samples": 64,
                "sample_interval_s": 0.25,
                "source_wdm_grid": None,
                "target_wdm_grid": grid,
            },
        )
    ]
```

Replace the white-noise test (from `@requires_wdm` at line 292 to the end of the file) with:

```python
def test_white_noise_wdm_coefficient_variance_matches_white_covariance():
    """E[w^2] == sigma^2 / fs at every folded cell for white noise of variance
    sigma^2 per sample: the WDM constant used by white_covariance."""
    fs, n, sigma = 0.2, 1 << 12, 0.7
    rng = np.random.default_rng(3)
    acc = None
    trials = 40
    for _ in range(trials):
        t = L1Data(
            channel_data={"A": rng.normal(0.0, sigma, n)},
            sample_rate_hz=fs,
            channel_names=("A",),
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
        )
        wdm = transform(t, "wdm", num_frequency_divisions=64)
        power = wdm.channel_data["A"] ** 2
        acc = power if acc is None else acc + power
    measured = acc / trials
    predicted = white_covariance(wdm, sigma**2).A
    np.testing.assert_array_equal(predicted, sigma**2 / fs)
    ratio = float(np.mean(measured / predicted))
    assert ratio == pytest.approx(1.0, abs=0.05)
    row_zero = float(np.mean(measured[0] / predicted[0]))  # DC and Nyquist slots
    assert row_zero == pytest.approx(1.0, abs=0.1)
```

- [ ] **Step 2: Run the translation tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_translation.py -q`
Expected: `ImportError: The local wdm backend requires Python >=3.13` or `WDM conversion requires the local wdm package` from the round-trip, regrid, seam and white-noise tests (the old `_load_wdm` is still in place).

- [ ] **Step 3: Rewrite `src/enchilada/_transforms.py`**

```python
"""Array transforms used by object translation.

Time <-> frequency conversions are NumPy ``rfft``/``irfft`` under the
``dt * rfft(x)`` convention. Every conversion with a WDM side goes through
domania, whose folded ``(Nf, Nt)`` coefficient layout `WDMGrid` describes.
"""

import domania
import numpy as np

from enchilada.domains import DOMAINS, WDMGrid


def _divisions(grid: WDMGrid) -> tuple[int, int]:
    return grid.num_frequency_divisions, grid.num_time_divisions


def _wdm_transform(
    stacked: np.ndarray,
    source_domain: str,
    target_domain: str,
    *,
    num_time_samples: int,
    sample_interval_s: float,
    source_wdm_grid: WDMGrid | None,
    target_wdm_grid: WDMGrid | None,
) -> np.ndarray:
    """domania.transform on the channel stack. Tests patch this to prove the
    grid checks run before any WDM transform."""
    # domania refuses a grid on a side that is not WDM, so pass one only there.
    source_wdm = (
        _divisions(source_wdm_grid)
        if source_domain == "wdm" and source_wdm_grid is not None
        else None
    )
    target_wdm = (
        _divisions(target_wdm_grid)
        if target_domain == "wdm" and target_wdm_grid is not None
        else None
    )
    return domania.transform(
        stacked,
        source_domain,
        target_domain,
        N=num_time_samples,
        dt=sample_interval_s,
        source_wdm=source_wdm,
        target_wdm=target_wdm,
    )


def transform_channels(
    channel_data: dict[str, np.ndarray],
    *,
    source_domain: str,
    target_domain: str,
    num_time_samples: int,
    sample_rate_hz: float,
    source_wdm_grid: WDMGrid | None = None,
    target_wdm_grid: WDMGrid | None = None,
) -> dict[str, np.ndarray]:
    """Transform validated channel grids, returning independently owned arrays.

    Time <-> frequency uses NumPy only. Any conversion with a WDM side calls
    domania through `_wdm_transform` after the grid checks; the coefficients
    come back in domania's folded layout, so every WDM cell carries data.
    """
    if source_domain not in DOMAINS or target_domain not in DOMAINS:
        raise ValueError("source_domain and target_domain must name supported domains")
    for name, values in channel_data.items():
        if not np.isfinite(values).all():
            raise ValueError(f"channel_data[{name!r}] must contain finite values")
    if source_domain == target_domain and source_wdm_grid == target_wdm_grid:
        return {name: values.copy() for name, values in channel_data.items()}
    sample_interval_s = 1.0 / sample_rate_hz
    channel_names = tuple(channel_data)
    stacked = np.stack([channel_data[name] for name in channel_names])
    if "wdm" in (source_domain, target_domain):
        if source_domain == "wdm" and source_wdm_grid is None:
            raise ValueError("WDM source requires wdm_grid")
        if target_domain == "wdm" and target_wdm_grid is None:
            raise ValueError("WDM target requires wdm_grid")
        if source_domain == "frequency":
            stacked = stacked.astype(
                np.result_type(stacked.dtype, np.complex128), copy=False
            )
        else:
            stacked = stacked.astype(np.float64, copy=False)
        converted = _wdm_transform(
            stacked,
            source_domain,
            target_domain,
            num_time_samples=num_time_samples,
            sample_interval_s=sample_interval_s,
            source_wdm_grid=source_wdm_grid,
            target_wdm_grid=target_wdm_grid,
        )
    elif source_domain == "time":
        converted = sample_interval_s * np.fft.rfft(stacked, axis=-1)
    else:
        # Normalize in at least double precision so complex64 coefficients do
        # not underflow or overflow before the inverse transform sees them.
        stacked = stacked.astype(
            np.result_type(stacked.dtype, np.complex128), copy=False
        )
        converted = np.fft.irfft(
            stacked / sample_interval_s, n=num_time_samples, axis=-1
        )
    if not np.isfinite(converted).all():
        raise ValueError("domain transformation produced non-finite coefficients")
    return {
        name: values.copy()
        for name, values in zip(channel_names, converted, strict=True)
    }
```

- [ ] **Step 4: Fix the `transform` docstring in `src/enchilada/translation.py`**

Line 74: `        num_frequency_divisions: WDM frequency divisions (Nf, the row count).`

Lines 82-87 become (the paragraph after `reference_data:`):

```
    Preserves observation metadata, orbit, published physical parameters and a
    block state's covariance. Signal arrays follow dt*rfft and domania's folded
    WDM layout. A Covariance does not translate between domains: passing one
    raises TypeError; ``Covariance.copy`` makes an independent copy.
```

- [ ] **Step 5: Run the translation tests again**

Run: `.venv/bin/python -m pytest tests/test_translation.py -q`
Expected: all pass.

- [ ] **Step 6: Replace the `_load_wdm` sentinels and the `wdm` skips in the other tests**

`tests/test_wheel_translation.py`: delete line 7 (`import importlib.util`) and lines 17-19 (`requires_wdm = ...`). Lines 90 and 93: `pytest.param("wdm", marks=requires_wdm)` becomes `"wdm"`. Delete the `@requires_wdm` decorator at line 126. Replace lines 223-239 with:

```python
def test_initial_wdm_signal_must_match_selected_grid_before_the_transform(monkeypatch):
    def unexpected_transform(*args, **kwargs):
        raise AssertionError("an invalid initial WDM grid reached domania")

    monkeypatch.setattr("enchilada._transforms._wdm_transform", unexpected_transform)
    data = observations()
    wheel = Wheel(data)
    # Both grids represent 64 samples, but selected (Nf, Nt) = (4, 16)
    # requires shape (4, 16), not the (8, 8) coefficients of an (8, 8) grid.
    initial = BlockState(
        template={name: np.zeros((8, 8)) for name in data.channel_names}
    )
    with pytest.raises(ValueError, match="shape"):
        wheel.add(
            ConstantSignal("wrong_grid", 1.0, "wdm"),
            initial_block_state=initial,
            data_domain="wdm",
            num_frequency_divisions=4,
            num_time_divisions=16,
        )
```

(the trailing assertions of that test, `assert len(wheel.ledger) == 0` and the residual loop, stay). In `test_wheel_rejects_mutated_wdm_buffers_with_same_time_sample_count` (lines 250-262) change `np.zeros((5, 16))` to `np.zeros((4, 16))` and `np.zeros((9, 8))` to `np.zeros((8, 8))`. Replace lines 309-314 with:

```python
def test_wdm_block_on_a_time_covariance_is_refused_before_the_transform(monkeypatch):
    def unexpected_transform(*args, **kwargs):
        raise AssertionError("the grid check must come before any WDM transform")

    monkeypatch.setattr("enchilada._transforms._wdm_transform", unexpected_transform)
```

`tests/test_initial_block_state.py`: delete line 3 (`import importlib.util`) and lines 24-26 (`requires_wdm = ...`). Lines 121-122 become `        ("time", "wdm"),` and `        ("wdm", "wdm"),`. Replace lines 237-251 (through the `with pytest.raises(...)` block) with:

```python
def test_failed_seed_translation_does_not_register_block(observed, monkeypatch):
    """An error raised by the WDM transform propagates and leaves the campaign
    untouched."""

    def failing_transform(*args, **kwargs):
        raise RuntimeError("test WDM transform failure")

    monkeypatch.setattr("enchilada._transforms._wdm_transform", failing_transform)
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    initial = BlockState(template={"A": np.zeros(grid.array_shape)})
    wheel = Wheel(observed, random_seed=7)
    rng_before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="test WDM transform failure"):
        wheel.add(
            SeededBlock(),
            initial_block_state=initial,
            data_domain="wdm",
            num_frequency_divisions=8,
        )
```

(the assertions after the `with` block stay as they are).

`tests/test_translation_review.py`: delete line 3 (`import importlib.util`); replace the parametrize at lines 91-103 with:

```python
@pytest.mark.parametrize("canonical_domain", ["time", "wdm"])
```

`tests/test_toy_translation.py`: delete lines 25-26 (`if canonical_domain == "wdm":` and the `importorskip`).

`tests/test_gb_example_contract.py`: delete lines 133-134 (`if storage_domain == "wdm":` and the `importorskip`).

`tests/test_translation_example.py`: delete line 11 (`pytest.importorskip("wdm", ...)`) and the blank line after it; lines 26-29 become

```python
    assert report["wdm_shapes"] == {
        "frequency_selector": [8, 16],
        "time_selector": [16, 8],
    }
```

and rename `test_translation_notebook_runs_all_cells_with_real_backend` to `test_translation_notebook_runs_all_cells_with_domania`.

- [ ] **Step 7: Update the walkthrough example docstring, `examples/domain_translation.py:1-10`**

```python
"""Translate data and block states, then run a mixed-domain Wheel.

From the enchilada checkout:
    uv sync
    uv run --no-sync python examples/domain_translation.py

This example needs only enchilada; its WDM transforms come from domania, a
core dependency. The zero-signal blocks exercise representation handoffs
without fitting a model.
"""
```

- [ ] **Step 8: Edit the notebook cells by id**

Save this script as `$TMPDIR/edit_translation_nb.py` and run it from the repo root with `.venv/bin/python "$TMPDIR/edit_translation_nb.py"`:

````python
"""Point examples/domain_translation.ipynb at domania's folded layout."""

import json
from pathlib import Path

PATH = Path("examples/domain_translation.ipynb")

SETUP = """## Setup

Run these commands in a terminal **from the enchilada repository root**:

```sh
uv sync --extra examples
uv pip install matplotlib
uv run --no-sync jupyter lab examples/domain_translation.ipynb
```

Select the kernel from that environment and run the cells from top to bottom.
The WDM transforms come from domania, a core dependency, so the project sync
is all they need; Matplotlib is used only for this notebook's plots. Install it
after the sync and launch with `--no-sync` so the environment retains it."""

LAYOUT = """## 2. Choose either WDM division count

The counts are linked by

```text
num_time_samples = num_frequency_divisions * num_time_divisions
```

Both counts must be positive even integers. Supply either one and the other is
derived. For our 128 samples, eight frequency divisions produce 16 time
divisions; eight time divisions produce 16 frequency divisions.

Each channel's WDM array has shape `(num_frequency_divisions,
num_time_divisions)`: domania's folded layout, in which every cell carries
data. Row `i >= 1` holds band `i` at each time index; row 0 holds the DC band
at even time indices and the Nyquist band, one step behind, at odd ones
(`domania.wilson.grid.cells` lists each slot's band and time).
`L1Data.wdm_grid` is read-only metadata derived automatically from this layout,
and is `None` for time or frequency data. Configure divisions through `transform`
or `Wheel.add(data_domain="wdm", ...)`; do not pass a grid to `L1Data`.
No padding or trimming is performed."""

PLOTS = """The two grids trade time resolution for frequency resolution. The WDM plots use
coefficient indices, and all representations show channel A. Row 0 interleaves
the DC band (even columns) with the Nyquist band (odd columns)."""

MARKDOWN = {"translation-01": SETUP, "translation-05": LAYOUT, "translation-07": PLOTS}

CODE_REPLACEMENTS = {
    "translation-06": [
        (
            'assert wdm_data.channel_data["A"].shape == (9, 16)',
            'assert wdm_data.channel_data["A"].shape == (8, 16)',
        ),
        (
            'assert wdm_finer_frequency.channel_data["A"].shape == (17, 8)',
            'assert wdm_finer_frequency.channel_data["A"].shape == (16, 8)',
        ),
    ],
    "translation-08": [
        (
            "    grid = representation.wdm_grid\n"
            "    visible_coefficients = np.ma.array(\n"
            '        representation.channel_data["A"], mask=~grid.active_mask\n'
            "    )\n"
            "    mesh = axis.imshow(\n"
            "        visible_coefficients,\n",
            "    grid = representation.wdm_grid\n"
            "    mesh = axis.imshow(\n"
            '        representation.channel_data["A"],\n',
        ),
    ],
}

raw = PATH.read_text()
notebook = json.loads(raw)
assert json.dumps(notebook, indent=1, ensure_ascii=False) + "\n" == raw, "format drift"
cells = {cell["id"]: cell for cell in notebook["cells"]}
for cell_id, text in MARKDOWN.items():
    cells[cell_id]["source"] = text.splitlines(keepends=True)
for cell_id, replacements in CODE_REPLACEMENTS.items():
    source = "".join(cells[cell_id]["source"])
    for old, new in replacements:
        assert source.count(old) == 1, (cell_id, old)
        source = source.replace(old, new)
    cells[cell_id]["source"] = source.splitlines(keepends=True)
PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print("edited", list(MARKDOWN) + list(CODE_REPLACEMENTS))
````

Then confirm nothing else in the notebook mentions the old layout:

Run: `grep -n "active_mask\|wdm checkout\|../wdm\|3.13\|+ 1,\|(9, 16)\|(17, 8)\|Inactive" examples/domain_translation.ipynb`
Expected: no output.

- [ ] **Step 9: Run the gates**

Run: `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q`
Expected: all pass; `-rs` shows only Eryn/GB skips and the opt-in real-file test, no "WDM"/"wdm" skip reasons.

Run: `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q tests/test_translation_example.py -rs`
Expected: 2 passed.

Run: `grep -rn "requires_wdm\|importorskip(\"wdm\|import wdm\|_load_wdm\|>=3\.13\|≥3\.13\|--python 3\.13\|wdm checkout\|../wdm" src tests examples`
Expected: no output.

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/enchilada/_transforms.py src/enchilada/translation.py tests/test_translation.py tests/test_wheel_translation.py tests/test_initial_block_state.py tests/test_translation_review.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_translation_example.py examples/domain_translation.py examples/domain_translation.ipynb
git commit -m "WDM conversions go through domania; the private wdm backend and its skips are gone

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Packaging and CI: domania, NumPy ≥ 2.1 and SciPy as core

**Goal:** `pyproject.toml` declares domania, NumPy ≥ 2.1 and SciPy ≥ 1.14.1 as core and shrinks the extras to h5py (+ lisaorbits); scipy is imported at module level; every scipy skip in the tests goes; the CI and release workflows lose the `wdm` jobs and test the new core floors; `wdm.yml` and the README's link to it are gone; `uv.lock` is regenerated.

**Files:**
- Modify: `pyproject.toml:11-12,32-36,45-54,99-103`
- Modify: `src/enchilada/resample.py:10,13-18,37,87`, `src/enchilada/orbits.py:13-14,21-22,118`, `src/enchilada/mojito.py:18`
- Modify: `tests/test_orbits.py:6`, `tests/test_resample.py:8-11`, `tests/test_mojito.py:9`, `tests/test_examples.py:56,346`
- Modify: `.github/workflows/ci.yml:59-61,78,81-82,93-107,128-131`, `.github/workflows/release.yml:75-90,109-115`
- Delete: `.github/workflows/wdm.yml`
- Modify: `README.md:436-449`
- Regenerate: `uv.lock` (controller, sandbox disabled)

**Acceptance Criteria:**
- [ ] `pyproject.toml` has `dependencies = ["numpy>=2.1", "scipy>=1.14.1", "domania>=3.0,<4"]`, `numeric-orbits = ["h5py>=3.12.1", "lisaorbits>=3.0.3"]`, `mojito = ["h5py>=3.12.1"]`, a `requires-python` comment naming domania and lisaorbits, and a mypy-override comment that no longer calls scipy lazy.
- [ ] `grep -n "scipy" src/enchilada/*.py` shows only module-level imports in `resample.py` and `orbits.py` plus the docstrings; `grep -rn 'importorskip("scipy' tests` prints nothing; h5py and lisaorbits skips stay.
- [ ] `.github/workflows/wdm.yml` does not exist; neither workflow mentions `wdm`; `core-minimum` installs `"numpy==2.1.0" "scipy==1.14.1" "domania==3.0.0"` with `--only-binary numpy --only-binary scipy` and asserts those versions plus the absence of h5py and lisaorbits only.
- [ ] README has no link to `wdm.yml` and no mention of a local `wdm` checkout or Python ≥ 3.13 (`tests/test_packaging_docs.py` and `tests/test_docs.py` pass).
- [ ] `~/.local/bin/uv lock --check` passes with the regenerated `uv.lock` (controller step).

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q` → all pass; `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy` → clean; `grep -rn "wdm" .github; grep -n "wdm\.yml\|wdm checkout\|\.\./wdm\|unpublished local" README.md` → no output; controller: `~/.local/bin/uv lock --check` → "Resolution is up-to-date" (or no output, exit 0).

**Steps:**

- [ ] **Step 1: Remove the scipy skips from the tests (they now fail nothing, and pin that scipy is core)**

`tests/test_orbits.py`: delete line 6 (`pytest.importorskip("scipy", ...)`) and the blank line after it.

`tests/test_resample.py` lines 1-11 become:

```python
"""decimate: anti-aliased rate reduction of time-domain L1Data."""

import math

import numpy as np
import pytest

from enchilada import L1Data, decimate
from enchilada.resample import anti_alias_taps, edge_samples
```

`tests/test_mojito.py`: delete line 9 (`pytest.importorskip("scipy", reason="the attached NumericOrbit needs scipy")`); keep line 8 (h5py) and the `# noqa: E402` comments on the imports that follow it.

`tests/test_examples.py`: delete line 56 (`pytest.importorskip("scipy", reason="notebook orbit cells need scipy")`) and line 346 (`pytest.importorskip("scipy", reason="decimate needs scipy")`).

Run: `.venv/bin/python -m pytest -q tests/test_orbits.py tests/test_resample.py tests/test_mojito.py tests/test_examples.py`
Expected: pass (scipy and h5py are installed locally).

- [ ] **Step 2: Declare the core in `pyproject.toml`**

Lines 11-12:

```toml
# Floor set by domania>=3.0 and lisaorbits>=3.0.3; both need >=3.12.
requires-python = ">=3.12"
```

Lines 32-36:

```toml
dependencies = [
    # domania (the WDM transform) needs numpy>=2.1 and scipy>=1.14.1; scipy is
    # also imported directly by resample.py and orbits.py. The core-minimum CI
    # lane pins these three floors.
    "numpy>=2.1",
    "scipy>=1.14.1",
    "domania>=3.0,<4",
]
```

Lines 45-54:

```toml
[project.optional-dependencies]
numeric-orbits = [
    "h5py>=3.12.1",       # lisaorbits' own floor; h5py wheels for NumPy 2 begin at 3.11
    "lisaorbits>=3.0.3",  # sets the >=3.12 python floor with domania
]
mojito = [
    "h5py>=3.12.1",       # load_mojito_l1 reads the L1 file
]
```

Lines 99-103:

```toml
[[tool.mypy.overrides]]
# scipy and h5py ship no type information; their use here is narrow.
module = ["scipy.*", "h5py.*"]
ignore_missing_imports = true
```

- [ ] **Step 3: Import scipy at module level**

`src/enchilada/resample.py`: line 10 becomes `SciPy is a core dependency, so ``decimate`` needs no extra.`; the import block (lines 13-18) becomes

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin, kaiserord, resample_poly

from enchilada.data import L1Data
```

and the two lazy imports go: delete line 37 (`    from scipy.signal import firwin, kaiserord  # lazy: optional dependency`) and line 87 (`    from scipy.signal import resample_poly  # lazy: optional dependency`).

`src/enchilada/orbits.py`: lines 13-14 of the module docstring become

```
NumericOrbit interpolates with SciPy, a core dependency; its HDF5 loader
imports h5py on demand (the ``numeric-orbits`` or ``mojito`` extra).
```

the import block (lines 21-22) becomes

```python
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
```

and line 118 (`        from scipy.interpolate import CubicSpline  # optional orbit dependency`) is deleted.

`src/enchilada/mojito.py:18`: `Imports h5py lazily; the ``mojito`` extra provides it.`

Run: `.venv/bin/python -m ruff check src && .venv/bin/python -m ruff format --check src && .venv/bin/python -m mypy`
Expected: clean (ruff orders the scipy imports; if `ruff check` reports I001, run `.venv/bin/python -m ruff check --fix src` and re-check).

- [ ] **Step 4: Edit `.github/workflows/ci.yml`**

Lines 59-61 (the comment above `core-only`):

```yaml
  # The core is numpy, scipy and domania. Prove it imports and tests on every
  # supported interpreter, including 3.14 (which the LISA dev stack runs); the
  # orbit and Mojito tests skip without h5py and lisaorbits.
```

Line 78: `      - name: Test (no coverage gate -- orbit and Mojito tests skip without h5py)`

Lines 81-82 (the comment above `core-minimum`):

```yaml
  # Install the core at its declared floors without the extras: uv sync would
  # resolve them too and let lisaorbits raise the floors.
```

Lines 93-107 (the two steps) become:

```yaml
      - name: Install the core at its dependency floors
        run: |
          uv venv --python 3.12
          uv pip install --only-binary numpy --only-binary scipy --group dev -e . "numpy==2.1.0" "scipy==1.14.1" "domania==3.0.0"
      - name: Verify the tested dependency boundary
        run: |
          uv run --no-sync python - <<'PY'
          import importlib.util

          import domania
          import numpy
          import scipy

          assert numpy.__version__ == "2.1.0", numpy.__version__
          assert scipy.__version__ == "1.14.1", scipy.__version__
          assert domania.__version__ == "3.0.0", domania.__version__
          for name in ("h5py", "lisaorbits"):
              assert importlib.util.find_spec(name) is None, name
          print("core minimum: NumPy", numpy.__version__, "SciPy", scipy.__version__,
                "domania", domania.__version__)
          PY
```

Delete the `wdm:` job (lines 128-131, from `  wdm:` through `      WDM_DEPLOY_KEY: ${{ secrets.WDM_DEPLOY_KEY }}` and the blank line after it).

- [ ] **Step 5: Edit `.github/workflows/release.yml` the same way**

Lines 76-90 (the two `core-minimum` steps) become the same two steps as in Step 4, under the existing comment line 75 replaced by `      # Install the core at its declared floors without the extras.`. Delete the `wdm:` job (lines 109-112) and change line 115 to `    needs: [gate, core-only, core-minimum, floors]`.

- [ ] **Step 6: Delete the WDM workflow and the README paragraph that links it**

```bash
git rm .github/workflows/wdm.yml
```

In `README.md` replace lines 436-449 (from `Time/Fourier transforms use only NumPy. WDM uses the unpublished local `wdm`` through `backend version and pinned revision.`) with:

````markdown
Time/Fourier transforms are plain NumPy FFTs. WDM conversions go through
[domania](https://pypi.org/project/domania/), a core dependency, so a plain
project sync is enough to run them:

```sh
uv sync
uv run --no-sync python examples/domain_translation.py
```
````

Run: `grep -rn "wdm" .github; grep -n "wdm\.yml\|wdm checkout\|\.\./wdm\|unpublished local" README.md`
Expected: no output.

- [ ] **Step 7: Run the gates**

Run: `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q`
Expected: all pass.

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy`
Expected: clean.

- [ ] **Step 8: Commit the code (implementer)**

```bash
git add pyproject.toml src/enchilada/resample.py src/enchilada/orbits.py src/enchilada/mojito.py tests/test_orbits.py tests/test_resample.py tests/test_mojito.py tests/test_examples.py .github/workflows/ci.yml .github/workflows/release.yml README.md
git commit -m "domania, NumPy 2.1 and SciPy are core; extras shrink to h5py; the wdm CI jobs and workflow go

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

`uv.lock` is left to the controller: `uv lock` cannot run inside the sandbox.

- [ ] **Step 9: Regenerate the lock (controller, sandbox disabled)**

Run with the sandbox disabled: `~/.local/bin/uv lock` then `~/.local/bin/uv lock --check`.
Expected: the lock lists `domania` 3.0.0 as a core dependency, `scipy` under core, `h5py` and `lisaorbits` only under the extras; `--check` exits 0. (`revision = 3` is what uv 0.12.1 writes; a newer uv elsewhere may have written 5. Harmless.)

Then:

```bash
git add uv.lock
git commit -m "Lock: domania 3.0 and scipy as core, h5py 3.12.1 floors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Example `examples/mojito_wdm_wheel.py`

**Goal:** A runnable example that fits a window to a WDM grid, loads it from a Mojito L1 file, decimates, translates to WDM and serves a diagnostic block on that grid, with a synthetic-file test, `fit_window` unit tests and an opt-in real-file test.

**Files:**
- Create: `examples/mojito_wdm_wheel.py`
- Modify: `tests/test_examples.py` (append four tests after `test_load_mojito_example_reads_a_synthetic_file`)

**Acceptance Criteria:**
- [ ] `fit_window(1000.0, dt=0.25, factor=4, num_frequency_divisions=8) == (992.0, 124)`; `fit_window(16.0, ...) == (16.0, 2)`; `fit_window(15.9, ...)` raises `ValueError` matching "fewer than two WDM time columns".
- [ ] On `write_mojito_file(path, n=4096)`, `main(path, num_frequency_divisions=8, factor=4, duration_s=1000.0)` returns a `Wheel` whose `residual()` is `data_domain == "wdm"` on `WDMGrid(8, 124)` with `(8, 124)` arrays, 992 samples at 1.0 Hz, `start_time_gps == MOJITO_T0`; stdout carries `window: 992.0 s`, `WDM grid: Nf = 8, Nt = 124`, `  {name}: rms ... per WDM coefficient` for A, E and T, and `residual: domain wdm`.
- [ ] Running the script without a path and without `ENCHILADA_MOJITO_L1` exits non-zero with "usage".
- [ ] With `ENCHILADA_MOJITO_L1` set, `main(path)` (defaults: a week, Nf = 512, factor 10) gives `WDMGrid(512, 472)` and `start_time_gps` equal to the file's `tdis/sampling` `t0`.
- [ ] The example's `dF`/`dT` lines use the decimated `data.dt` (defaults: `dF = 3.906e-04 Hz`, `dT = 1280.0 s`).

**Verify:** `.venv/bin/python -m pytest -q tests/test_examples.py -k "mojito_wdm_wheel or fit_window"` → 3 passed (synthetic, two `fit_window` cases) + 1 passed (refusal), real-file test skipped; `.venv/bin/python -m ruff check examples tests && .venv/bin/python -m ruff format --check examples tests` → clean.

**Steps:**

- [ ] **Step 1: Write the tests (append to `tests/test_examples.py`)**

```python
def test_mojito_wdm_wheel_example_builds_a_wdm_wheel(tmp_path, capsys):
    pytest.importorskip("h5py", reason="the Mojito loader needs h5py")
    from conftest import MOJITO_T0, write_mojito_file
    from enchilada import WDMGrid

    path = tmp_path / "mojito.h5"
    write_mojito_file(path, n=4096)  # 1024 s at 4 Hz
    example = load_example("mojito_wdm_wheel")
    wheel = example.main(path, num_frequency_divisions=8, factor=4, duration_s=1000.0)
    residual = wheel.residual()
    assert residual.data_domain == "wdm"
    assert residual.wdm_grid == WDMGrid(
        num_frequency_divisions=8, num_time_divisions=124
    )
    assert residual.num_time_samples == 992  # 3968 native samples / 4
    assert residual.sample_rate_hz == 1.0
    assert residual.start_time_gps == MOJITO_T0
    for name in residual.channel_names:
        assert residual.channel_data[name].shape == (8, 124)
    out = capsys.readouterr().out
    assert "window: 992.0 s" in out, out
    assert "WDM grid: Nf = 8, Nt = 124" in out, out
    for name in residual.channel_names:
        assert f"  {name}: rms" in out, out
    assert out.count("per WDM coefficient") == len(residual.channel_names), out
    assert "residual: domain wdm" in out, out
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "mojito_wdm_wheel.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in os.environ.items() if k != "ENCHILADA_MOJITO_L1"},
    )
    assert proc.returncode != 0
    assert "usage" in (proc.stderr + proc.stdout).lower()


@pytest.mark.parametrize(
    ("duration_s", "expected"),
    [
        (1000.0, (992.0, 124)),  # int(1000 / 8) = 125 columns, rounded down to even
        (16.0, (16.0, 2)),  # exactly two 8 s columns
    ],
)
def test_fit_window_rounds_down_to_an_even_number_of_columns(duration_s, expected):
    example = load_example("mojito_wdm_wheel")
    assert (
        example.fit_window(duration_s, dt=0.25, factor=4, num_frequency_divisions=8)
        == expected
    )


def test_fit_window_refuses_a_window_shorter_than_two_columns():
    example = load_example("mojito_wdm_wheel")
    with pytest.raises(ValueError, match="fewer than two WDM time columns"):
        example.fit_window(15.9, dt=0.25, factor=4, num_frequency_divisions=8)


REAL_MOJITO_FILE = os.environ.get("ENCHILADA_MOJITO_L1")


@pytest.mark.skipif(
    not REAL_MOJITO_FILE, reason="set ENCHILADA_MOJITO_L1 to a Mojito L1 file"
)
def test_mojito_wdm_wheel_example_on_the_real_file(capsys):
    """A week at Nf = 512: 2,416,640 native samples per channel, decimated by 10
    to 512 x 472 WDM coefficients. Takes tens of seconds."""
    h5py = pytest.importorskip("h5py")
    from enchilada import WDMGrid

    with h5py.File(REAL_MOJITO_FILE, "r") as f:
        t0 = float(f["tdis/sampling"].attrs["t0"])
    example = load_example("mojito_wdm_wheel")
    wheel = example.main(REAL_MOJITO_FILE)
    residual = wheel.residual()
    assert residual.wdm_grid == WDMGrid(
        num_frequency_divisions=512, num_time_divisions=472
    )
    assert residual.start_time_gps == t0
    assert "WDM grid: Nf = 512, Nt = 472" in capsys.readouterr().out
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -q tests/test_examples.py -k "mojito_wdm_wheel or fit_window"`
Expected: `FileNotFoundError` / `AttributeError` from `load_example("mojito_wdm_wheel")` (no such file).

- [ ] **Step 3: Write `examples/mojito_wdm_wheel.py`**

```python
"""Load a Mojito L1 window, decimate it, store it as WDM and run a Wheel on it.

Needs the `mojito` extra (h5py) and a Mojito L1 file:

    uv run --no-sync python examples/mojito_wdm_wheel.py /path/to/mojito_light_..._L1.h5

or set ENCHILADA_MOJITO_L1 to the file path. `load_mojito.py` is the data side
of the Mojito story; this is the Wheel side. A WDM observation needs exactly
`num_frequency_divisions * num_time_divisions` samples, both counts even, and
neither `transform` nor the loader pads or trims, so `fit_window` sizes the
window first: the longest one of at most `duration_s` whose decimated record
fits the grid.
"""

import os
import sys

import numpy as np

from enchilada import BlockState, Wheel, decimate, load_mojito_l1, transform
from enchilada.testing import EchoBlock

SECONDS_PER_DAY = 86400.0


def fit_window(duration_s, *, dt, factor, num_frequency_divisions):
    """Longest window of at most `duration_s` whose record, decimated by `factor`,
    has `num_frequency_divisions * num_time_divisions` samples with
    `num_time_divisions` even. Returns `(native_duration_s, num_time_divisions)`.

    `dt` is the file's native sampling interval; one WDM time column spans
    `factor * num_frequency_divisions` native samples.
    """
    samples_per_column = factor * num_frequency_divisions
    num_time_divisions = int(duration_s / (samples_per_column * dt))
    num_time_divisions -= num_time_divisions % 2
    if num_time_divisions < 2:
        raise ValueError(
            f"duration_s={duration_s} holds fewer than two WDM time columns of "
            f"{samples_per_column * dt} s"
        )
    return num_time_divisions * samples_per_column * dt, num_time_divisions


def main(
    path, *, num_frequency_divisions=512, factor=10, duration_s=7 * SECONDS_PER_DAY
):
    """Fit a window, load and decimate it, translate to WDM, run one Wheel cycle."""
    # A cheap probe for the file's sampling interval: one minute of one channel.
    probe = load_mojito_l1(path, channels=("A",), duration_s=60.0, orbit=False)
    native_duration_s, num_time_divisions = fit_window(
        duration_s,
        dt=probe.dt,
        factor=factor,
        num_frequency_divisions=num_frequency_divisions,
    )
    native = load_mojito_l1(path, duration_s=native_duration_s)
    data = decimate(native, factor)  # exactly Nf * Nt samples
    observed = transform(data, "wdm", num_frequency_divisions=num_frequency_divisions)
    grid = observed.wdm_grid
    print(
        f"window: {native_duration_s} s = {factor} x {num_frequency_divisions} x "
        f"{num_time_divisions} native samples at dt = {probe.dt} s"
    )
    frequency_resolution_hz = 1.0 / (2.0 * grid.num_frequency_divisions * data.dt)
    time_resolution_s = grid.num_frequency_divisions * data.dt
    print(
        f"WDM grid: Nf = {grid.num_frequency_divisions}, "
        f"Nt = {grid.num_time_divisions}, dF = {frequency_resolution_hz:.3e} Hz, "
        f"dT = {time_resolution_s} s, shape {grid.array_shape}"
    )
    for name in observed.channel_names:
        rms = float(np.sqrt(np.mean(observed.channel_data[name] ** 2)))
        print(f"  {name}: rms {rms:.3e} per WDM coefficient")
    wheel = Wheel(observed, random_seed=0)
    wheel.add(EchoBlock(name="echo"), initial_block_state=BlockState())
    wheel.run(num_cycles=1)  # the block is served on the WDM grid
    residual = wheel.residual()
    print(f"residual: domain {residual.data_domain} on {residual.wdm_grid}")
    return wheel


if __name__ == "__main__":
    file_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ENCHILADA_MOJITO_L1")
    )
    if not file_path:
        sys.exit("usage: mojito_wdm_wheel.py PATH  (or set ENCHILADA_MOJITO_L1)")
    main(file_path)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_examples.py -k "mojito_wdm_wheel or fit_window" -rs`
Expected: 4 passed, 1 skipped (`set ENCHILADA_MOJITO_L1 to a Mojito L1 file`).

Run: `.venv/bin/python -m ruff check examples tests && .venv/bin/python -m ruff format --check examples tests`
Expected: clean (if `ruff format --check` lists the new file, run `.venv/bin/python -m ruff format examples/mojito_wdm_wheel.py tests/test_examples.py` and re-check).

- [ ] **Step 5: Commit**

```bash
git add examples/mojito_wdm_wheel.py tests/test_examples.py
git commit -m "Example: fit a Mojito window to a WDM grid and run a Wheel on it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: README and CHANGELOG

**Goal:** The README describes the core dependencies, the folded WDM layout and how to fit a Mojito window to a WDM grid, and the CHANGELOG records the breaking changes and the new example.

**Files:**
- Modify: `README.md:44-58,292-293,335-339,428-434,579-580,624-625,692-700`, plus a new subsection after line 661 (the end of the "Conventions the loader fixes" list)
- Modify: `CHANGELOG.md:11-14,23-24,26-33,65-66` (0.4.0 section)

Line numbers are from the tree before Task 3: Task 3's README edit (lines 436-449 become 8 lines) shifts every later README line by −6, and this task's own CHANGELOG insertion shifts the lines below it by +3. Every step quotes its anchor text; find by text, not by number.

**Acceptance Criteria:**
- [ ] `grep -n "active_mask\|inactive\|+ 1, num_time\|NumPy-only\|only on NumPy\|1.26.4\|wdm.yml\|every active point" README.md CHANGELOG.md` prints nothing except the historical 0.3.0 CHANGELOG entries (lines below `## [0.3.0]`).
- [ ] README states the core depends on NumPy ≥ 2.1, SciPy ≥ 1.14.1 and domania ≥ 3.0; `numeric-orbits` adds h5py and lisaorbits; `mojito` adds h5py; `decimate` needs no extra.
- [ ] README's translation section gives the folded shape `(num_frequency_divisions, num_time_divisions)`, the row-0 slot rule and the pointer to `domania.wilson.grid.cells`.
- [ ] README's Mojito section has a "Fitting a window to a WDM grid" subsection with the `factor * Nf * Nt * dt` rule, the 824-sample / 34-minute remainder for Nf = 512 at 0.4 Hz on the full Mojito Light record, and a link to `examples/mojito_wdm_wheel.py`.
- [ ] CHANGELOG 0.4.0 has the Breaking entry for the core dependencies and the folded layout, an Added entry for `examples/mojito_wdm_wheel.py`, the `Grid` entry without `active_mask`, and the `mojito` extra described as h5py only.
- [ ] `tests/test_docs.py`, `tests/test_packaging_docs.py` pass.

**Verify:** `.venv/bin/python -m pytest -q tests/test_docs.py tests/test_packaging_docs.py` → pass; `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q --cov=enchilada` → all pass, coverage ≥ 95 %; `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy` → clean.

**Steps:**

- [ ] **Step 1: README install paragraph and block (lines 44-58)**

Replace from `The core package depends only on NumPy` through the closing fence of the install block with:

````markdown
The core depends on NumPy (`>=2.1`), SciPy (`>=1.14.1`) and
[domania](https://pypi.org/project/domania/) (`>=3.0,<4`), which performs the
WDM transforms. GBGPU, Eryn, and LISA Analysis Tools
are confined to the optional galactic-binary example. Installing enchilada or
any of its extras does not install them; running that example requires the
separate `examples/requirements-gb.txt` installation described below.

Loading tabulated spacecraft ephemerides (`enchilada.orbits.NumericOrbit`)
needs the `numeric-orbits` extra; reading Mojito L1 files (`enchilada.mojito`)
needs the `mojito` extra. Decimating (`enchilada.resample`) needs no extra:

```sh
uv sync --extra numeric-orbits   # adds h5py, lisaorbits
uv sync --extra mojito           # or this: adds h5py only (no lisaorbits)

# For the notebooks, install the orbit and example extras together:
uv sync --extra numeric-orbits --extra examples
uv run --no-sync jupyter lab examples/demo.ipynb
```
````

- [ ] **Step 2: README covariance text**

Lines 292-293: replace `` `C_k = N * dt**2 * sigma**2` at every bin and `dt * sigma**2` at every active `` / `` WDM pixel (zero at the structurally inactive edge pixels). A block that `` with

```
`C_k = N * dt**2 * sigma**2` at every bin and `dt * sigma**2` at every WDM
cell. A block that
```

Lines 335-339: replace `` `(num_frequency_divisions + 1, num_time_divisions)` WDM). Construction checks `` through `read-only.` with

```
`(num_frequency_divisions, num_time_divisions)` WDM). Construction checks
finite entries, exact Hermitian symmetry, real values where the domain requires
them, positive variances at every point and positive definiteness at every
point; the array is stored read-only.
```

- [ ] **Step 3: README translation layout paragraph (lines 428-434)**

Replace from `Each WDM channel contains a real `float64` array of shape` through `passing `wdm_grid` to `L1Data`.` with:

```
Each WDM channel is a real `float64` array of shape
`(num_frequency_divisions, num_time_divisions)` in domania's folded layout, in
which every cell carries data: row `i >= 1` holds band `i` at each time index,
and row 0 holds the DC band at even time indices and the Nyquist band, one
step behind, at odd ones (`domania.wilson.grid.cells(Nf, Nt)` lists each
slot's band and time). `L1Data` derives its read-only `wdm_grid` metadata from
this coefficient layout. It is a frozen `WDMGrid` for WDM data and `None` for
time or frequency data; copies preserve the grid. Choose division counts
through `transform` or `Wheel.add(data_domain="wdm", ...)`, rather than
passing `wdm_grid` to `L1Data`.
```

- [ ] **Step 4: README orbits and Mojito dependency sentences**

Lines 579-580: replace `HDF5 files (`from_hdf5`, which needs h5py and scipy from either the` / `` `numeric-orbits` or the `mojito` extra) and lisaorbits objects `` with

```
HDF5 files (`from_hdf5`, which needs h5py from either the `numeric-orbits` or
the `mojito` extra) and lisaorbits objects
```

Lines 624-625: replace `` `enchilada.resample.decimate` reduces the 4 Hz record to a rate a campaign can `` / `` run at. Both need the `mojito` extra. `` with

```
`enchilada.resample.decimate` reduces the 4 Hz record to a rate a campaign can
run at. The loader needs the `mojito` extra; `decimate` needs no extra.
```

- [ ] **Step 5: README "Fitting a window to a WDM grid"**

Insert after line 661 (`  file's `orbits` group; its BCRS positions are rotated to the ecliptic frame.`, the last bullet of the conventions list) and its following blank line, before the paragraph that starts `` `decimate(data, factor)` applies one fixed anti-alias policy ``:

```
### Fitting a window to a WDM grid

A WDM observation needs `num_frequency_divisions * num_time_divisions` samples
after decimation, both counts even, and neither `transform` nor the loader pads
or trims. Choose `Nf`, take the largest even `Nt` with
`factor * Nf * Nt * dt <= duration`, and pass `factor * Nf * Nt * dt` as
`duration_s` to `load_mojito_l1`. The full Mojito Light record (252,465,200
samples) does not factor onto any power-of-two `Nf`, so a campaign drops a few
hundred to a few thousand samples (`Nf = 512` at 0.4 Hz: 824 samples, 34
minutes). [`examples/mojito_wdm_wheel.py`](examples/mojito_wdm_wheel.py) does
this arithmetic in `fit_window`, loads a week, decimates by 10, translates the
record to WDM (`Nf = 512`, `Nt = 472`, `dF = 3.9e-4 Hz`, `dT = 1280 s`) and
serves a diagnostic block on that grid.

```

- [ ] **Step 6: README CI paragraph (lines 692-700)**

Replace from `CI runs lint, formatting, mypy,` through `[CHANGELOG.md](...) for release notes.` with:

```
CI runs lint, formatting, mypy, the suite behind a 95% coverage gate, artifact
builds, and an installed-wheel smoke test across Python 3.12/3.13 on Linux and
macOS, with core-only checks through Python 3.14. A separate Python 3.12 job tests
the core at its dependency floors (NumPy 2.1.0, SciPy 1.14.1, domania 3.0.0)
without h5py or lisaorbits; another job resolves the numeric-orbit and mojito
dependency floors. Tag pushes matching `v*` run these gates before PyPI Trusted
Publishing. Manual release dispatches build and validate artifacts without
publishing, including on tags. See
[CHANGELOG.md](https://github.com/AaronDJohnson/enchilada/blob/main/CHANGELOG.md) for release notes.
```

Run: `grep -n "active_mask\|inactive\|+ 1, num_time\|NumPy-only\|only on NumPy\|1.26.4\|wdm.yml\|every active point\|wdm checkout\|../wdm" README.md`
Expected: no output.

- [ ] **Step 7: CHANGELOG 0.4.0**

Lines 11-14 (the `Grid` entry): remove `` `active_mask`, `` so it reads `` with `channel_shape`, `observation_duration_s` and `same_record` ``.

After the `decimate` bullet (it ends at line 23) add:

```
- `examples/mojito_wdm_wheel.py`: fits a Mojito window to a WDM grid
  (`fit_window`), loads and decimates it, translates it to WDM and serves a
  diagnostic block on that grid; the Wheel side of the Mojito story.
```

Line 24: `- Optional extra `mojito` (h5py) for loading data without lisaorbits.`

Lines 32-33 (the end of the `Covariance` bullet): `Hermitian symmetry, realness per domain, positive variances and positive` / `definiteness with one batched Cholesky.`

At the top of `### Changed` (before the `Covariance` bullet) add:

```
- **Breaking:** the core depends on `numpy>=2.1`, `scipy>=1.14.1` and
  `domania>=3.0,<4`; every WDM conversion goes through domania. The WDM
  coefficient layout is domania's folded `(num_frequency_divisions,
  num_time_divisions)`, in which every cell carries data (row 0 interleaves the
  DC and Nyquist bands); `WDMGrid.active_mask`, `Grid.active_mask` and the
  inactive-zero rules on coefficients and covariances are removed. The private
  `wdm` backend, its Python >=3.13 requirement and the `wdm.yml` workflow are
  gone. The `numeric-orbits` and `mojito` extras no longer carry scipy, so
  `decimate` and `NumericOrbit` need no extra; both extras require
  `h5py>=3.12.1`.
```

Lines 65-66 (the migration note): `` `N * dt**2 * sigma**2` per frequency bin or `dt * sigma**2` per WDM cell. ``

- [ ] **Step 8: Run the gates**

Run: `.venv/bin/python -m pytest -q tests/test_docs.py tests/test_packaging_docs.py`
Expected: pass.

Run: `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q --cov=enchilada`
Expected: all pass; `TOTAL` coverage ≥ 95 %.

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy`
Expected: clean.

Run: `grep -rn "active_mask\|requires_wdm\|importorskip(\"wdm\|importorskip(\"scipy\|>=3\.13\|≥3\.13\|--python 3\.13\|NumPy-only\|requires only NumPy\|depends only on NumPy\|scipy lazily\|from scipy.*lazy" src tests examples benchmarks README.md`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "Docs: core dependencies, folded WDM layout, fitting a Mojito window to a WDM grid

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## After the last task (controller)

1. Real-file checks, with the Mojito Light file: `ENCHILADA_MOJITO_L1=<file> MPLBACKEND=Agg .venv/bin/python -m pytest -q tests/test_mojito.py tests/test_examples.py -k real_file` → 2 passed; note the wall time of the example test for the PR body.
2. `~/.local/bin/uv lock --check` (sandbox disabled) → up to date.
3. Push `domania-wdm` and open the PR against `mojito-loader` with `gh pr create --repo nsgs-analysis-center/enchilada` (sandbox disabled), body ending `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## Self-review

- **Spec coverage.** §4.1-4.5 → Task 1; §5 and §6 → Task 2; §7 and §8 → Task 3; §9 and §10.1 → Task 4; §10.2 → Task 1; §10.3 → Task 2 (plus the scipy skips in Task 3); §10.4 → each task's Verify and the controller checklist; §11 → Task 5 (README, CHANGELOG) and Task 2 (example docstring, notebook); §12 non-goals untouched.
- **Placeholders.** Every code step carries the code; every verify names the command and the expected result.
- **Type consistency.** `_wdm_transform(stacked, source_domain, target_domain, *, num_time_samples, sample_interval_s, source_wdm_grid, target_wdm_grid)` is the signature in Task 2's module, its seam test (`recording(stacked, source_domain, target_domain, **options)`) and the three monkeypatch sentinels (`*args, **kwargs`). `fit_window(duration_s, *, dt, factor, num_frequency_divisions) -> (native_duration_s, num_time_divisions)` and `main(path, *, num_frequency_divisions=512, factor=10, duration_s=7 * SECONDS_PER_DAY)` match between Task 4's example, its tests and Task 5's README text. `WDMGrid` is always constructed with keywords.
