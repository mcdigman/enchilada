# Mojito L1 Loader and Decimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read a DDPC Mojito L1 HDF5 file into a Wheel-ready `L1Data` and reduce its 4 Hz rate with one fixed anti-alias policy, both tested without the 33 GB file.

**Architecture:** Two small package modules behind a lazy-import extra. `enchilada/mojito.py` holds `load_mojito_l1` (window, channel map, Hz to fractional frequency, flag refusal, orbit attach); `enchilada/resample.py` holds `decimate` plus the two helpers that define its filter. Tests build a tiny synthetic file with the real layout; one opt-in test reads the real file when `ENCHILADA_MOJITO_L1` is set.

**Tech Stack:** Python 3.12, NumPy, h5py (lazy), scipy.signal (`kaiserord`, `firwin`, `resample_poly`; lazy), pytest, ruff, mypy. Tooling in this repo: `.venv/bin/python -m pytest|ruff|mypy`, never `uv run`.

**User decisions (already made):**
- Target is the DDPC Mojito Light L1 file layout (`tdis/<ch>2` in Hz, `tdis/sampling` attrs, `tdis/tdi_flags`, `orbits/` group); Mojito Heavy shares it.
- Loaded data is fractional frequency: stored Hz divided by the root attribute `laser_frequency`, `physical_observable="fractional_frequency"`.
- The loader reads a window and channel subset at the native rate; decimation is a separate function.
- Package modules `src/enchilada/mojito.py` and `src/enchilada/resample.py`, both exported from `enchilada`, behind a new extra `mojito = ["h5py>=3.0", "scipy>=1.8"]`.
- Anti-alias policy is fixed (Kaiser FIR, 120 dB, passband to 0.8 of the new Nyquist, zero-phase), no knobs.
- Deferred: `noise_estimates` reader, eta channels, brick stacking, gap handling.

**Spec:** `docs/superpowers/specs/2026-11-14-mojito-loader-design.md`. **Branch:** `mojito-loader` (cut from `covariance-simplification`).

**Repo facts an implementer needs:**
- `L1Data(channel_data, *, sample_rate_hz, tdi_generation, physical_observable, channel_names=None, num_time_samples=0, start_time_gps=0.0, data_domain="time", orbit_ephemeris=None)` validates arrays, dtype and the orbit span (`orbit_ephemeris.time_range_gps` must cover `[start_time_gps, start_time_gps + (N-1)*dt]`). `data.grid`, `data.N`, `data.t0`, `data.dt`, `data.fs` are properties.
- `NumericOrbit.from_hdf5(file_path, *, group_path="orbits", position_dataset_names=("sc_position_1", "sc_position_2", "sc_position_3"), coordinate_frame="equatorial", nominal_arm_length_m=None)` reads `sampling` attrs `t0`, `dt`, `size` and `(n, 3)` position datasets; `"equatorial"` rotates to ecliptic with the J2000 obliquity `0.40909280422232897` rad using the matrix `[[1,0,0],[0,c,s],[0,-s,c]]` applied as `positions @ R.T`.
- `tests/test_typing.py::test_public_surface_is_pinned` pins `enchilada.__all__` literally; every new export must be added there too.
- `tests/test_docs.py` compiles every ```` ```python ```` block in README, checks keyword names for callables listed in `_CALLABLES`, runs only blocks tagged `<!-- runnable -->`, and requires every `enchilada.X` in README prose to be an attribute of the package.
- `tests/conftest.py` is imported by tests as `from conftest import ...`; it must stay importable without h5py or scipy (import them inside functions).
- Prototype numbers for the filter (scipy 1.18): factor 8 gives 627 taps, interior alignment error 3.7e-7, suppression 1.8e-7, 40 edge samples; a one-input-sample misalignment would show as 7.9e-3.

---

### Task 1: `decimate` in `enchilada/resample.py`

**Goal:** Anti-aliased integer-factor rate reduction of time-domain `L1Data`, exported from `enchilada`, with the filter policy pinned by tests.

**Files:**
- Create: `src/enchilada/resample.py`
- Create: `tests/test_resample.py`
- Modify: `src/enchilada/__init__.py` (import and `__all__`)
- Modify: `tests/test_typing.py:237-251` (pinned `__all__` list)

**Acceptance Criteria:**
- [ ] `decimate(data, factor)` returns an `L1Data` with `num_time_samples == ceil(N / factor)`, `sample_rate_hz == fs / factor`, unchanged `start_time_gps`, channel order, `tdi_generation`, `physical_observable`, and the same `orbit_ephemeris` object.
- [ ] A tone at 1/50 of the new Nyquist, decimated by 8, matches its analytic value at `t0 + k*8*dt` within 2e-6 outside `edge_samples(8)` samples at each end.
- [ ] A tone at 1.5 times the new Nyquist comes out below 1e-6 in amplitude outside the edges.
- [ ] Frequency or WDM input raises `ValueError`; `factor` of `1`, `True` or `2.0` raises `ValueError`/`TypeError`/`TypeError`; a record shorter than the filter raises `ValueError` naming both lengths.
- [ ] `anti_alias_taps(factor)` has odd length between 70 and 90 times `factor`, unit DC gain, |H| within 1e-6 of 1 at half the new Nyquist and below 1e-6 at 1.5 times it.
- [ ] `from enchilada import decimate` works and `enchilada.__all__` contains `"decimate"` (pin test updated).

**Verify:** `.venv/bin/python -m pytest tests/test_resample.py tests/test_typing.py -q` → all pass; `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy` → clean.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resample.py`:

```python
"""decimate: anti-aliased rate reduction of time-domain L1Data."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy", reason="decimate needs scipy.signal")

from enchilada import L1Data, decimate  # noqa: E402
from enchilada.resample import anti_alias_taps, edge_samples  # noqa: E402

T0 = 97729889.327664  # Mojito's first L1 sample, TCB seconds after the LISA epoch
FS = 4.0


class StubOrbit:
    nominal_arm_length_m = 2.5e9
    transfer_frequency_hz = 0.019

    def positions(self, times_gps):
        raise NotImplementedError


def make_data(channel_data, *, orbit=None, fs=FS):
    return L1Data(
        channel_data,
        sample_rate_hz=fs,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=T0,
        orbit_ephemeris=orbit,
    )


def tone(frequency_hz, n, *, phase=0.0, fs=FS):
    return np.sin(2 * np.pi * frequency_hz * np.arange(n) / fs + phase)


def test_low_frequency_tone_is_aligned_with_the_coarse_grid():
    n, factor = 2**16, 8
    f = (FS / factor / 2) / 50
    out = decimate(make_data({"A": tone(f, n, phase=0.3)}), factor)
    k = np.arange(out.num_time_samples)
    # Output sample k sits at t0 + k * factor * dt: a one-input-sample
    # misalignment would show as about 8e-3 here.
    expected = np.sin(2 * np.pi * f * k * factor / FS + 0.3)
    edge = edge_samples(factor)
    np.testing.assert_allclose(
        out.channel_data["A"][edge:-edge], expected[edge:-edge], atol=2e-6
    )


def test_tone_above_the_new_nyquist_is_suppressed():
    n, factor = 2**16, 8
    f = 1.5 * (FS / factor / 2)
    out = decimate(make_data({"A": tone(f, n)}), factor)
    edge = edge_samples(factor)
    assert np.max(np.abs(out.channel_data["A"][edge:-edge])) < 1e-6


def test_output_grid_for_a_non_divisible_length():
    n, factor = 1003, 8
    out = decimate(make_data({"A": np.zeros(n), "E": np.zeros(n)}), factor)
    assert out.num_time_samples == math.ceil(n / factor) == 126
    assert out.sample_rate_hz == FS / factor
    assert out.start_time_gps == T0
    assert out.data_domain == "time"
    assert out.grid.channel_shape == (126,)


def test_conventions_order_and_orbit_carry_over():
    n = 2000
    orbit = StubOrbit()
    data = make_data({"E": np.zeros(n), "A": np.ones(n)}, orbit=orbit)
    out = decimate(data, 4)
    assert out.channel_names == ("E", "A")
    assert out.tdi_generation == "2.0"
    assert out.physical_observable == "fractional_frequency"
    assert out.orbit_ephemeris is orbit
    edge = edge_samples(4)
    np.testing.assert_allclose(out.channel_data["A"][edge:-edge], 1.0, atol=2e-6)


def test_refuses_frequency_and_wdm_input():
    data = make_data({"A": tone(0.01, 2**12)})
    with pytest.raises(ValueError, match="time-domain"):
        decimate(data.to_frequency(), 2)
    wdm = L1Data(
        {"A": np.zeros((5, 8))},
        sample_rate_hz=FS,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        data_domain="wdm",
    )
    with pytest.raises(ValueError, match="time-domain"):
        decimate(wdm, 2)


@pytest.mark.parametrize(
    "factor, error",
    [(1, ValueError), (0, ValueError), (True, TypeError), (2.0, TypeError)],
)
def test_refuses_bad_factors(factor, error):
    with pytest.raises(error):
        decimate(make_data({"A": np.zeros(4096)}), factor)


def test_refuses_a_record_shorter_than_the_filter():
    with pytest.raises(ValueError, match="shorter than"):
        decimate(make_data({"A": np.zeros(100)}), 8)


@pytest.mark.parametrize("factor", [2, 10])
def test_anti_alias_policy(factor):
    from scipy.signal import freqz

    taps = anti_alias_taps(factor)
    assert taps.size % 2 == 1
    assert 70 * factor <= taps.size <= 90 * factor
    assert taps.sum() == pytest.approx(1.0, abs=1e-9)
    new_nyquist = FS / factor / 2
    frequencies, response = freqz(
        taps, worN=np.array([0.5 * new_nyquist, 1.5 * new_nyquist]), fs=FS
    )
    assert abs(abs(response[0]) - 1.0) < 1e-6
    assert abs(response[1]) < 1e-6
    assert edge_samples(factor) == (taps.size // 2) // factor + 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_resample.py -q`
Expected: collection error, `ImportError: cannot import name 'decimate' from 'enchilada'`.

- [ ] **Step 3: Write the module**

Create `src/enchilada/resample.py`:

```python
"""Rate reduction of time-domain `L1Data` behind one fixed anti-alias policy.

`decimate` keeps the lower 80 % of the new band clean: a Kaiser-window FIR
low-pass with 120 dB stopband attenuation, passband edge at 0.8 times the new
Nyquist frequency and stopband edge at the new Nyquist, applied zero-phase by
polyphase filtering. The policy is fixed rather than exposed as parameters so
every campaign decimates the same way; a different policy is a different
function.

Imports scipy lazily; the core package stays NumPy-only.
"""

from __future__ import annotations

import numpy as np

from enchilada.data import L1Data

STOPBAND_ATTENUATION_DB = 120.0
"""Minimum attenuation above the new Nyquist frequency, in dB."""

PASSBAND_FRACTION = 0.8
"""Passband edge as a fraction of the new Nyquist frequency."""


def anti_alias_taps(factor: int) -> np.ndarray:
    """FIR coefficients :func:`decimate` uses for ``factor``.

    A Kaiser-window low-pass from :func:`scipy.signal.kaiserord` and
    :func:`scipy.signal.firwin`: transition band from
    ``PASSBAND_FRACTION * fs_new / 2`` to ``fs_new / 2`` with
    ``fs_new = fs / factor``, ``STOPBAND_ATTENUATION_DB`` attenuation, odd
    length so the polyphase filter is exactly zero-phase, unit DC gain. About
    ``78 * factor`` taps.
    """
    from scipy.signal import firwin, kaiserord  # lazy: optional dependency

    transition_width = (1.0 - PASSBAND_FRACTION) / factor  # of the input Nyquist
    numtaps, beta = kaiserord(STOPBAND_ATTENUATION_DB, transition_width)
    numtaps = int(numtaps) | 1
    cutoff = (1.0 + PASSBAND_FRACTION) / (2.0 * factor)  # transition-band centre
    return np.asarray(
        firwin(numtaps, cutoff, window=("kaiser", beta)), dtype=np.float64
    )


def edge_samples(factor: int) -> int:
    """Output samples at each end of a decimated record touched by the filter's
    zero padding: half the filter length, counted in output samples, plus one.
    About 39 for every ``factor``."""
    return (anti_alias_taps(factor).size // 2) // factor + 1


def decimate(data: L1Data, factor: int) -> L1Data:
    """Return ``data`` at ``sample_rate_hz / factor`` on a new, validated grid.

    Args:
        data: Time-domain `L1Data`.
        factor: Integer rate reduction, at least 2.

    Returns:
        A new `L1Data` with ``num_time_samples = ceil(N / factor)``,
        ``sample_rate_hz = fs / factor``, the same ``start_time_gps`` (output
        sample ``k`` sits at input sample ``k * factor`` because the polyphase
        filter is zero-phase), the same channel order, conventions and
        ``orbit_ephemeris`` object.

    The first and last :func:`edge_samples` output samples carry the filter's
    edge transient from zero padding. Nothing is trimmed, so the grid stays
    predictable; trim and re-slice if a campaign cares.

    Raises:
        ValueError: for frequency or WDM input, ``factor < 2``, or a record
            shorter than the anti-alias filter.
        TypeError: for a non-integer ``factor``.
    """
    if data.data_domain != "time":
        raise ValueError(
            f"decimate takes time-domain L1Data, got data_domain="
            f"{data.data_domain!r}; transform to 'time' first"
        )
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise TypeError(f"factor must be an integer, got {factor!r}")
    if factor < 2:
        raise ValueError(f"factor must be at least 2, got {factor}")
    from scipy.signal import resample_poly  # lazy: optional dependency

    factor = int(factor)
    taps = anti_alias_taps(factor)
    if data.num_time_samples < taps.size:
        raise ValueError(
            f"record of {data.num_time_samples} samples is shorter than the "
            f"{taps.size}-tap anti-alias filter for factor {factor}; decimate a "
            f"longer record or use a smaller factor"
        )
    channel_data = {
        name: np.asarray(
            resample_poly(data.channel_data[name], 1, factor, window=taps),
            dtype=np.float64,
        )
        for name in data.channel_names
    }
    return L1Data(
        channel_data,
        sample_rate_hz=data.sample_rate_hz / factor,
        tdi_generation=data.tdi_generation,
        physical_observable=data.physical_observable,
        channel_names=data.channel_names,
        start_time_gps=data.start_time_gps,
        orbit_ephemeris=data.orbit_ephemeris,
    )
```

- [ ] **Step 4: Export it**

In `src/enchilada/__init__.py` add the import in alphabetical position:

```python
from enchilada.orbits import NumericOrbit, Orbit
from enchilada.resample import decimate
from enchilada.translation import transform
```

and change `__all__` to end:

```python
    "WDMGrid",
    "__version__",
    "decimate",
    "replace",
    "transform",
]
```

In `tests/test_typing.py::test_public_surface_is_pinned` make the pinned list end the same way:

```python
        "WDMGrid",
        "__version__",
        "decimate",
        "replace",
        "transform",
    ]
```

- [ ] **Step 5: Run the tests and the linters**

Run: `.venv/bin/python -m pytest tests/test_resample.py tests/test_typing.py -q`
Expected: all pass (11 resample tests plus the typing file).

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy`
Expected: no findings, "Success: no issues found".

- [ ] **Step 6: Commit**

```bash
git add src/enchilada/resample.py tests/test_resample.py src/enchilada/__init__.py tests/test_typing.py
git commit -m "decimate: anti-aliased rate reduction of time-domain L1Data

Kaiser FIR (120 dB, passband to 0.8 of the new Nyquist), zero-phase
polyphase, start time unchanged, edge transient documented.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `load_mojito_l1` in `enchilada/mojito.py`

**Goal:** Read a window of TDI channels from a Mojito L1 file into a fractional-frequency `L1Data` with the file's orbit attached, refusing flagged samples and bad windows, tested on a synthetic file plus an opt-in real-file check.

**Files:**
- Create: `src/enchilada/mojito.py`
- Create: `tests/test_mojito.py`
- Modify: `tests/conftest.py` (append `write_mojito_file` and its constants)
- Modify: `src/enchilada/__init__.py` (import and `__all__`)
- Modify: `tests/test_typing.py:237-252` (pinned `__all__` list)

**Acceptance Criteria:**
- [ ] Default load returns channels `("A", "E", "T")` in that order with `tdi_generation == "2.0"`, `physical_observable == "fractional_frequency"`, values equal to the stored Hz divided by `laser_frequency`, `sample_rate_hz == fs`, and `start_time_gps == t0` for a zero offset.
- [ ] `channels=("Z", "X")` returns those two in that order; an unknown name raises `ValueError` listing `A, E, T, X, Y, Z`; a file without `tdis/T2` raises `ValueError` naming `tdis/T2`.
- [ ] `start_offset_s=10.1, duration_s=100.9` at `dt = 0.25` gives first sample 40, 403 samples, `start_time_gps == t0 + 40 * dt`; omitting `duration_s` reads to the end; a window starting or ending past the file, or with zero samples, raises `ValueError` whose message contains the file span.
- [ ] A file whose `fs` attribute disagrees with `dt` raises `ValueError` mentioning both.
- [ ] A window containing a flagged sample raises `ValueError` with the flagged count and the file index of the first; a window excluding it loads.
- [ ] `orbit=True` attaches a `NumericOrbit` with the table's `time_range_gps` and positions in the ecliptic frame; `orbit=False` gives `None`.
- [ ] With `ENCHILADA_MOJITO_L1` set, one hour of the real file gives 14400 samples at 4 Hz starting at the file `t0`, channel RMS between 1e-19 and 1e-17, `A == (Z - X)/sqrt(2)` to 1e-12 relative, orbit coverage, and `decimate(data, 10)` with 1440 samples at 0.4 Hz; the test skips otherwise.
- [ ] `from enchilada import load_mojito_l1` works and the `__all__` pin includes it.

**Verify:** `.venv/bin/python -m pytest tests/test_mojito.py tests/test_typing.py -q` → all pass (one skipped without the env var); `ENCHILADA_MOJITO_L1="/Users/aaron/Documents/lisa/mojito_data/mojito_light_731d_0.25s_L1_0_0_20251218T000909441368Z(1).h5" .venv/bin/python -m pytest tests/test_mojito.py -q -k real_file` → 1 passed; ruff and mypy clean.

**Steps:**

- [ ] **Step 1: Add the synthetic-file writer to `tests/conftest.py`**

Append at the end of `tests/conftest.py` (h5py is imported inside the function so the core-only CI lane can still import conftest):

```python
# ---- Mojito L1 fixture file -------------------------------------------------

MOJITO_LASER_FREQUENCY_HZ = 2.816e14
MOJITO_T0 = 97729889.327664  # first 4 Hz L1 sample, TCB seconds after the LISA epoch
MOJITO_DT = 0.25
MOJITO_ORBIT_DT = 500000.0
MOJITO_ORBIT_SIZE = 12
MOJITO_ORBIT_T0 = MOJITO_T0 - 2 * MOJITO_ORBIT_DT
MOJITO_ECLIPTIC_Z_M = 1.0e8
_J2000_OBLIQUITY_RAD = 0.40909280422232897
_AET_FROM_XYZ = np.array(
    [[-1.0, 0.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, 1.0]]
) / np.sqrt([2.0, 6.0, 3.0])[:, None]


def mojito_orbit_positions_equatorial(times):
    """Three spacecraft on a 1 au circle in the ecliptic plane at constant
    ecliptic z = MOJITO_ECLIPTIC_Z_M, written in equatorial coordinates (the
    frame Mojito stores), shaped (3, n, 3)."""
    au = 1.495978707e11
    omega = 2 * np.pi / 31557600.0
    phase = times * omega
    positions = np.empty((3, times.size, 3))
    c, s = np.cos(_J2000_OBLIQUITY_RAD), np.sin(_J2000_OBLIQUITY_RAD)
    for i in range(3):
        x = au * np.cos(phase + 2 * np.pi * i / 3)
        y = au * np.sin(phase + 2 * np.pi * i / 3)
        z = np.full_like(x, MOJITO_ECLIPTIC_Z_M)
        positions[i, :, 0] = x
        positions[i, :, 1] = c * y - s * z  # ecliptic -> equatorial
        positions[i, :, 2] = s * y + c * z
    return positions


def write_mojito_file(path, *, n=4096, flag_index=None, drop=(), fs=None):
    """Write a tiny file in the Mojito L1 layout and return (xyz, aet) in Hz.

    ``xyz`` and ``aet`` are (3, n) arrays in dataset order X, Y, Z and A, E, T;
    A/E/T are the orthonormal combination of X/Y/Z, as in the real product.
    ``flag_index`` sets one tdi_flags entry to 1; ``drop`` removes datasets
    (e.g. ``("T2",)``); ``fs`` overrides the sampling-rate attribute.
    """
    import h5py

    rng = np.random.default_rng(1)
    xyz = 6e-4 * rng.standard_normal((3, n))
    aet = _AET_FROM_XYZ @ xyz
    with h5py.File(path, "w") as f:
        f.attrs["laser_frequency"] = MOJITO_LASER_FREQUENCY_HZ
        f.attrs["lolipops_version"] = "0.4.5"
        f.attrs["pipeline_name"] = "mojito-signal"
        tdis = f.create_group("tdis")
        sampling = tdis.create_group("sampling")
        sampling.attrs["t0"] = MOJITO_T0
        sampling.attrs["dt"] = MOJITO_DT
        sampling.attrs["fs"] = 1.0 / MOJITO_DT if fs is None else fs
        sampling.attrs["size"] = n
        sampling.attrs["duration"] = n * MOJITO_DT
        for name, row in zip("XYZ", xyz, strict=True):
            tdis.create_dataset(f"{name}2", data=row)
        for name, row in zip("AET", aet, strict=True):
            tdis.create_dataset(f"{name}2", data=row)
        flags = np.zeros(n, dtype=np.int8)
        if flag_index is not None:
            flags[flag_index] = 1
        tdis.create_dataset("tdi_flags", data=flags)
        tdis.create_dataset("eta_flags", data=np.zeros(n, dtype=np.int8))
        for name in drop:
            del tdis[name]
        orbits = f.create_group("orbits")
        orbit_sampling = orbits.create_group("sampling")
        orbit_sampling.attrs["t0"] = MOJITO_ORBIT_T0
        orbit_sampling.attrs["dt"] = MOJITO_ORBIT_DT
        orbit_sampling.attrs["fs"] = 1.0 / MOJITO_ORBIT_DT
        orbit_sampling.attrs["size"] = MOJITO_ORBIT_SIZE
        orbit_sampling.attrs["duration"] = MOJITO_ORBIT_SIZE * MOJITO_ORBIT_DT
        times = MOJITO_ORBIT_T0 + np.arange(MOJITO_ORBIT_SIZE) * MOJITO_ORBIT_DT
        positions = mojito_orbit_positions_equatorial(times)
        for i in range(3):
            orbits.create_dataset(f"sc_position_{i + 1}", data=positions[i])
    return xyz, aet
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_mojito.py`:

```python
"""load_mojito_l1: a Mojito L1 HDF5 window as fractional-frequency L1Data."""

import os

import numpy as np
import pytest

pytest.importorskip("h5py", reason="the Mojito loader needs h5py")
pytest.importorskip("scipy", reason="the attached NumericOrbit needs scipy")

from conftest import (  # noqa: E402
    MOJITO_DT,
    MOJITO_ECLIPTIC_Z_M,
    MOJITO_LASER_FREQUENCY_HZ,
    MOJITO_ORBIT_DT,
    MOJITO_ORBIT_SIZE,
    MOJITO_ORBIT_T0,
    MOJITO_T0,
    write_mojito_file,
)

from enchilada import NumericOrbit, decimate, load_mojito_l1  # noqa: E402

N = 4096


@pytest.fixture
def mojito_file(tmp_path):
    path = tmp_path / "mojito.h5"
    xyz, aet = write_mojito_file(path, n=N)
    return path, xyz, aet


def test_default_channels_units_and_grid(mojito_file):
    path, xyz, aet = mojito_file
    data = load_mojito_l1(path)
    assert data.channel_names == ("A", "E", "T")
    assert data.tdi_generation == "2.0"
    assert data.physical_observable == "fractional_frequency"
    assert data.sample_rate_hz == 4.0
    assert data.num_time_samples == N
    assert data.start_time_gps == MOJITO_T0
    assert data.data_domain == "time"
    for row, name in zip(aet, "AET", strict=True):
        np.testing.assert_array_equal(
            data.channel_data[name], row / MOJITO_LASER_FREQUENCY_HZ
        )


def test_xyz_channels_in_the_requested_order(mojito_file):
    path, xyz, _ = mojito_file
    data = load_mojito_l1(path, channels=("Z", "X"), orbit=False)
    assert data.channel_names == ("Z", "X")
    np.testing.assert_array_equal(
        data.channel_data["Z"], xyz[2] / MOJITO_LASER_FREQUENCY_HZ
    )
    np.testing.assert_array_equal(
        data.channel_data["X"], xyz[0] / MOJITO_LASER_FREQUENCY_HZ
    )


def test_unknown_channel_is_refused(mojito_file):
    path, _, _ = mojito_file
    with pytest.raises(ValueError, match=r"A, E, T, X, Y, Z"):
        load_mojito_l1(path, channels=("A", "B"))


def test_missing_dataset_is_refused(tmp_path):
    path = tmp_path / "no_t.h5"
    write_mojito_file(path, drop=("T2",))
    with pytest.raises(ValueError, match="tdis/T2"):
        load_mojito_l1(path)


def test_window_rounds_the_start_and_floors_the_count(mojito_file):
    path, _, aet = mojito_file
    data = load_mojito_l1(path, start_offset_s=10.1, duration_s=100.9, orbit=False)
    # 10.1 / 0.25 = 40.4 -> 40; 100.9 / 0.25 = 403.6 -> 403
    assert data.num_time_samples == 403
    assert data.start_time_gps == MOJITO_T0 + 40 * MOJITO_DT
    np.testing.assert_array_equal(
        data.channel_data["A"], aet[0, 40:443] / MOJITO_LASER_FREQUENCY_HZ
    )


def test_omitted_duration_reads_to_the_end(mojito_file):
    path, _, aet = mojito_file
    data = load_mojito_l1(path, start_offset_s=1000.0, orbit=False)
    assert data.num_time_samples == N - 4000
    np.testing.assert_array_equal(
        data.channel_data["E"], aet[1, 4000:] / MOJITO_LASER_FREQUENCY_HZ
    )


@pytest.mark.parametrize(
    "window",
    [
        dict(start_offset_s=2000.0),  # starts past the end
        dict(start_offset_s=1000.0, duration_s=100.0),  # ends past the end
        dict(start_offset_s=0.0, duration_s=0.1),  # zero samples
        dict(start_offset_s=-1.0),  # before the first sample
    ],
)
def test_windows_outside_the_file_are_refused(mojito_file, window):
    path, _, _ = mojito_file
    last = MOJITO_T0 + (N - 1) * MOJITO_DT
    with pytest.raises(ValueError, match=rf"\[{MOJITO_T0}, {last}\]"):
        load_mojito_l1(path, orbit=False, **window)


def test_inconsistent_rate_attributes_are_refused(tmp_path):
    path = tmp_path / "bad_fs.h5"
    write_mojito_file(path, fs=3.9)
    with pytest.raises(ValueError, match=r"dt.*fs"):
        load_mojito_l1(path, orbit=False)


def test_flagged_samples_are_refused(tmp_path):
    path = tmp_path / "flagged.h5"
    write_mojito_file(path, flag_index=2000)
    with pytest.raises(ValueError, match=r"1 of 4096 .*sample 2000"):
        load_mojito_l1(path, orbit=False)
    with pytest.raises(ValueError, match="gap mask"):
        load_mojito_l1(path, start_offset_s=400.0, duration_s=200.0, orbit=False)
    data = load_mojito_l1(path, duration_s=400.0, orbit=False)
    assert data.num_time_samples == 1600


def test_orbit_is_attached_in_the_ecliptic_frame(mojito_file):
    path, _, _ = mojito_file
    data = load_mojito_l1(path)
    orbit = data.orbit_ephemeris
    assert isinstance(orbit, NumericOrbit)
    assert orbit.time_range_gps == (
        MOJITO_ORBIT_T0,
        MOJITO_ORBIT_T0 + (MOJITO_ORBIT_SIZE - 1) * MOJITO_ORBIT_DT,
    )
    nodes = MOJITO_ORBIT_T0 + np.arange(3) * MOJITO_ORBIT_DT
    _, _, z = orbit.positions(nodes)
    np.testing.assert_allclose(z, MOJITO_ECLIPTIC_Z_M, rtol=1e-6)
    assert load_mojito_l1(path, orbit=False).orbit_ephemeris is None


REAL_FILE = os.environ.get("ENCHILADA_MOJITO_L1")


@pytest.mark.skipif(not REAL_FILE, reason="set ENCHILADA_MOJITO_L1 to a Mojito L1 file")
def test_real_file_one_hour_window():
    import h5py

    with h5py.File(REAL_FILE, "r") as f:
        t0 = float(f["tdis/sampling"].attrs["t0"])
    data = load_mojito_l1(REAL_FILE, duration_s=3600.0)
    assert data.num_time_samples == 14400
    assert data.sample_rate_hz == 4.0
    assert data.start_time_gps == t0
    for name in data.channel_names:
        rms = float(np.sqrt(np.mean(data.channel_data[name] ** 2)))
        assert 1e-19 < rms < 1e-17, (name, rms)
    xyz = load_mojito_l1(REAL_FILE, channels=("X", "Y", "Z"), duration_s=3600.0, orbit=False)
    expected_a = (xyz.channel_data["Z"] - xyz.channel_data["X"]) / np.sqrt(2.0)
    np.testing.assert_allclose(data.channel_data["A"], expected_a, rtol=1e-12, atol=0.0)
    first, last = data.orbit_ephemeris.time_range_gps
    assert first <= data.t0 and last >= data.t0 + (data.N - 1) * data.dt
    coarse = decimate(data, 10)
    assert coarse.num_time_samples == 1440
    assert coarse.sample_rate_hz == 0.4
    assert coarse.start_time_gps == t0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mojito.py -q`
Expected: collection error, `ImportError: cannot import name 'load_mojito_l1' from 'enchilada'`.

- [ ] **Step 4: Write the module**

Create `src/enchilada/mojito.py`:

```python
"""Read DDPC Mojito L1 HDF5 files into `L1Data`.

Mojito is the LISA DDPC Common Dataset 1 (LISA-DDPC-SEG-TN-008). An L1 file
stores second-generation TDI time series under ``tdis/`` as ``X2``, ``Y2``,
``Z2``, ``A2``, ``E2`` and ``T2`` in total-frequency units (Hz); the time grid
in the attributes ``t0``, ``dt``, ``fs`` and ``size`` of ``tdis/sampling``;
quality flags in ``tdis/tdi_flags`` (0 usable, 1 not); the laser frequency in
the root attribute ``laser_frequency``; and a spacecraft ephemeris under
``orbits/`` in BCRS (equatorial) metres, in the layout
:meth:`enchilada.orbits.NumericOrbit.from_hdf5` reads by default.

Times are TCB seconds after the LISA reference epoch. The orbit table shares
that clock, and the loader carries them in ``start_time_gps`` unchanged. The
stored ``A2``/``E2``/``T2`` equal the orthonormal combination of
``X2``/``Y2``/``Z2`` that :meth:`enchilada.Covariance.in_basis` uses, so a
campaign may load either triple.

Imports h5py lazily; the ``mojito`` extra provides h5py and scipy.
"""

from __future__ import annotations

import math
import os

import numpy as np

from enchilada.data import L1Data
from enchilada.orbits import NumericOrbit

TDI_CHANNELS = ("A", "E", "T", "X", "Y", "Z")
"""Channel names the loader accepts; dataset ``tdis/<name>2`` holds each."""

TDI_GENERATION = "2.0"
"""Every Mojito TDI dataset is a second-generation combination."""

_RATE_TOLERANCE = 1e-9


def _resolve_window(
    t0: float, dt: float, size: int, start_offset_s: float, duration_s: float | None
) -> tuple[int, int]:
    """First sample index and sample count of the requested window.

    The start rounds to the nearest sample and the count rounds down, so a
    window never reaches past what was asked for.
    """
    if not math.isfinite(start_offset_s):
        raise ValueError(f"start_offset_s must be finite, got {start_offset_s!r}")
    if duration_s is not None and not math.isfinite(duration_s):
        raise ValueError(f"duration_s must be finite, got {duration_s!r}")
    first = int(round(start_offset_s / dt))
    count = size - first if duration_s is None else int(math.floor(duration_s / dt))
    if first < 0 or first >= size or count < 1 or first + count > size:
        last_time = t0 + (size - 1) * dt
        raise ValueError(
            f"requested window start_offset_s={start_offset_s}, "
            f"duration_s={duration_s} (samples {first} to {first + count - 1}) "
            f"does not fit the file, which holds {size} samples at dt = {dt} s "
            f"spanning [{t0}, {last_time}] s; offsets count from the first sample"
        )
    return first, count


def load_mojito_l1(
    path: str | os.PathLike[str],
    *,
    channels: tuple[str, ...] = ("A", "E", "T"),
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
    orbit: bool = True,
) -> L1Data:
    """Load a window of TDI channels from a Mojito L1 file.

    Args:
        path: The L1 HDF5 file.
        channels: Names from ``A E T X Y Z``, in the order the `L1Data` keeps;
            each reads dataset ``tdis/<name>2``.
        start_offset_s: Seconds after the file's first sample; rounds to the
            nearest sample.
        duration_s: Window length in seconds, rounded down to whole samples;
            None reads to the end of the file.
        orbit: Attach the file's ``orbits`` table as a `NumericOrbit`
            (BCRS positions rotated to the ecliptic frame).

    Returns:
        Time-domain `L1Data` in fractional frequency (the stored Hz divided by
        the root attribute ``laser_frequency``), ``tdi_generation="2.0"``,
        ``sample_rate_hz`` from the ``fs`` attribute, and ``start_time_gps =
        t0 + first_sample * dt`` on the file's TCB clock.

    Raises:
        ValueError: unknown or missing channel, a window outside the file,
            ``dt`` and ``fs`` attributes that disagree, or any nonzero
            ``tdi_flags`` value in the window (enchilada has no gap mask).
    """
    import h5py  # lazy: optional dependency

    channels = tuple(channels)
    unknown = [name for name in channels if name not in TDI_CHANNELS]
    if unknown:
        raise ValueError(
            f"unknown Mojito TDI channel(s) {unknown}; choose from "
            f"{', '.join(TDI_CHANNELS)}"
        )
    file_path = os.fspath(path)
    with h5py.File(file_path, "r") as file:
        sampling = file["tdis/sampling"].attrs
        t0 = float(sampling["t0"])
        dt = float(sampling["dt"])
        fs = float(sampling["fs"])
        size = int(sampling["size"])
        if abs(dt * fs - 1.0) > _RATE_TOLERANCE:
            raise ValueError(
                f"tdis/sampling attributes disagree: dt = {dt} s and fs = {fs} Hz "
                f"give dt * fs = {dt * fs}, expected 1"
            )
        first, count = _resolve_window(t0, dt, size, start_offset_s, duration_s)
        laser_frequency_hz = float(file.attrs["laser_frequency"])
        window = slice(first, first + count)
        flagged = np.flatnonzero(np.asarray(file["tdis/tdi_flags"][window]))
        if flagged.size:
            raise ValueError(
                f"{flagged.size} of {count} samples in the window carry a nonzero "
                f"tdi_flags value (first at file sample {first + int(flagged[0])}); "
                f"enchilada has no gap mask, so choose a window of usable samples"
            )
        channel_data: dict[str, np.ndarray] = {}
        for name in channels:
            dataset = f"tdis/{name}2"
            if dataset not in file:
                raise ValueError(
                    f"{file_path} has no dataset {dataset}; the tdis group holds "
                    f"{sorted(file['tdis'])}"
                )
            values = np.asarray(file[dataset][window], dtype=np.float64)
            values /= laser_frequency_hz  # Hz -> fractional frequency
            channel_data[name] = values
    ephemeris = (
        NumericOrbit.from_hdf5(
            file_path, group_path="orbits", coordinate_frame="equatorial"
        )
        if orbit
        else None
    )
    return L1Data(
        channel_data,
        sample_rate_hz=fs,
        tdi_generation=TDI_GENERATION,
        physical_observable="fractional_frequency",
        channel_names=channels,
        start_time_gps=t0 + first * dt,
        orbit_ephemeris=ephemeris,
    )
```

- [ ] **Step 5: Export it**

In `src/enchilada/__init__.py`:

```python
from enchilada.ledger import Ledger
from enchilada.mojito import load_mojito_l1
from enchilada.orbits import NumericOrbit, Orbit
```

and `__all__` ends:

```python
    "WDMGrid",
    "__version__",
    "decimate",
    "load_mojito_l1",
    "replace",
    "transform",
]
```

Mirror that ending in `tests/test_typing.py::test_public_surface_is_pinned`.

- [ ] **Step 6: Run the tests, the real-file test, and the linters**

Run: `.venv/bin/python -m pytest tests/test_mojito.py tests/test_typing.py -q`
Expected: all pass, 1 skipped (the real-file test).

Run: `ENCHILADA_MOJITO_L1="/Users/aaron/Documents/lisa/mojito_data/mojito_light_731d_0.25s_L1_0_0_20251218T000909441368Z(1).h5" .venv/bin/python -m pytest tests/test_mojito.py -q -k real_file`
Expected: 1 passed (reads about 60 MB; a few seconds).

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/enchilada/mojito.py tests/test_mojito.py tests/conftest.py src/enchilada/__init__.py tests/test_typing.py
git commit -m "load_mojito_l1: a Mojito L1 window as fractional-frequency L1Data

Window and channel selection, Hz divided by laser_frequency, flagged
samples refused, the file's orbit table attached in the ecliptic frame.
Synthetic-layout tests plus an opt-in real-file check.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Example script with a skip-safe test

**Goal:** `examples/load_mojito.py` loads one day, decimates and prints a summary, and its test runs on the synthetic fixture so no real file is needed.

**Files:**
- Create: `examples/load_mojito.py`
- Modify: `tests/test_examples.py` (append one test)

**Acceptance Criteria:**
- [ ] `main(path, duration_s=1000.0, factor=4)` on the synthetic file returns an `L1Data` with 1000 samples at 1.0 Hz and prints lines containing `native grid`, `decimated grid`, `rms` per channel and `orbit span`.
- [ ] Running the script without an argument and without `ENCHILADA_MOJITO_L1` exits with a usage message and a nonzero status.
- [ ] The test skips when h5py or scipy is absent.

**Verify:** `.venv/bin/python -m pytest tests/test_examples.py -q -k mojito` → 1 passed; `.venv/bin/python examples/load_mojito.py` → exits 1 with the usage line; ruff clean on `examples`.

**Steps:**

- [ ] **Step 1: Write the failing test**

Append to `tests/test_examples.py`:

```python
def test_load_mojito_example_reads_a_synthetic_file(tmp_path, capsys):
    pytest.importorskip("h5py", reason="the Mojito loader needs h5py")
    pytest.importorskip("scipy", reason="decimate needs scipy")
    from conftest import write_mojito_file

    path = tmp_path / "mojito.h5"
    write_mojito_file(path, n=4096)  # 1024 s at 4 Hz
    example = load_example("load_mojito")
    data = example.main(path, duration_s=1000.0, factor=4)
    assert data.num_time_samples == 1000
    assert data.sample_rate_hz == 1.0
    out = capsys.readouterr().out
    for fragment in ("native grid", "decimated grid", "rms", "orbit span"):
        assert fragment in out, out
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "load_mojito.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in os.environ.items() if k != "ENCHILADA_MOJITO_L1"},
    )
    assert proc.returncode != 0
    assert "usage" in (proc.stderr + proc.stdout).lower()
```

Add `import os` to the imports at the top of `tests/test_examples.py` (alphabetically before `import subprocess`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_examples.py -q -k mojito`
Expected: FAIL with `FileNotFoundError` for `examples/load_mojito.py`.

- [ ] **Step 3: Write the example**

Create `examples/load_mojito.py`:

```python
"""Load one day of Mojito L1 data into enchilada and decimate it.

Needs the `mojito` extra (h5py, scipy) and a Mojito L1 file:

    uv run --no-sync python examples/load_mojito.py /path/to/mojito_light_..._L1.h5

or set ENCHILADA_MOJITO_L1 to the file path. This example only reads data;
blocks and the Wheel live in the other examples.
"""

import os
import sys

import numpy as np

from enchilada import decimate, load_mojito_l1


def main(path, *, duration_s=86400.0, factor=10):
    """Load A, E, T for `duration_s` seconds, decimate by `factor`, print a summary."""
    native = load_mojito_l1(path, duration_s=duration_s)
    data = decimate(native, factor)
    print(f"native grid: {native.grid}")
    print(f"decimated grid: {data.grid} (t0 = {data.t0} s on the file's TCB clock)")
    for name in data.channel_names:
        rms = float(np.sqrt(np.mean(data.channel_data[name] ** 2)))
        print(f"  {name}: rms {rms:.3e} (fractional frequency)")
    first, last = data.orbit_ephemeris.time_range_gps
    print(f"orbit span: {first:.3f} to {last:.3f} s")
    return data


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ENCHILADA_MOJITO_L1")
    if not file_path:
        sys.exit("usage: load_mojito.py PATH  (or set ENCHILADA_MOJITO_L1)")
    main(file_path)
```

- [ ] **Step 4: Run the test and the linters**

Run: `.venv/bin/python -m pytest tests/test_examples.py -q -k mojito`
Expected: 1 passed.

Run: `.venv/bin/python -m ruff check src tests examples && .venv/bin/python -m ruff format --check src tests examples`
Expected: clean (wrap the `file_path = ...` line if ruff reports E501).

- [ ] **Step 5: Commit**

```bash
git add examples/load_mojito.py tests/test_examples.py
git commit -m "Example: load one day of Mojito L1 data and decimate it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Packaging, CI and documentation

**Goal:** The `mojito` extra exists and is locked, CI installs it where it installs `numeric-orbits`, and README and CHANGELOG describe the loader and `decimate`.

**Files:**
- Modify: `pyproject.toml:45-50` (optional dependencies)
- Modify: `uv.lock` (regenerate)
- Modify: `.github/workflows/ci.yml:35,123,156` and `.github/workflows/release.yml:41,105,163`
- Modify: `README.md` (install section after line 53; new section before `## Development`; the gap-mask bullet in Known limitations)
- Modify: `CHANGELOG.md:9-15` (0.4.0 Added)
- Modify: `tests/test_docs.py:51-56` (`_CALLABLES`)

**Acceptance Criteria:**
- [ ] `pyproject.toml` has `mojito = ["h5py>=3.0", "scipy>=1.8"]` and `uv.lock` lists the extra (`uv lock --check` passes).
- [ ] Every `uv sync ... --extra numeric-orbits` in `ci.yml` and `release.yml` also passes `--extra mojito`.
- [ ] README has a "Loading Mojito L1 data" section with a non-runnable Python block calling `load_mojito_l1(..., channels=..., start_offset_s=..., duration_s=...)` and `decimate(...)`, the install line, and the limitation sentence; `tests/test_docs.py` and `tests/test_packaging_docs.py` pass with `load_mojito_l1` and `decimate` added to `_CALLABLES`.
- [ ] CHANGELOG 0.4.0 Added lists `load_mojito_l1`, `decimate` and the `mojito` extra.
- [ ] Full suite passes with coverage at or above 95 %.

**Verify:** `MPLBACKEND=Agg .venv/bin/python -m pytest -q --cov=enchilada` → 0 failed, TOTAL ≥ 95 %; `uv lock --check` → no changes; ruff (incl. `examples benchmarks`) and mypy clean.

**Steps:**

- [ ] **Step 1: Declare the extra**

In `pyproject.toml`, after the `numeric-orbits` list and before `examples`:

```toml
mojito = [
    "h5py>=3.0",          # load_mojito_l1 reads the L1 file
    "scipy>=1.8",         # decimate (scipy.signal) and the attached NumericOrbit
]
```

- [ ] **Step 2: Regenerate the lockfile**

Run: `uv lock --offline` (both packages are already resolved in `uv.lock`). If that fails on network access, run `uv lock` with the sandbox disabled. Then `uv lock --check` must report the lockfile is up to date, and `git diff --stat uv.lock` should touch only the `enchilada` package's `optional-dependencies`, `requires-dist` and `provides-extras` entries.

- [ ] **Step 3: CI lanes**

Change the three lines in `.github/workflows/ci.yml` and the three in `.github/workflows/release.yml` from `--extra numeric-orbits` to `--extra numeric-orbits --extra mojito`:

```yaml
        run: uv sync --locked --extra numeric-orbits --extra mojito
```
```yaml
        run: uv sync --resolution lowest-direct --extra numeric-orbits --extra mojito
```
```yaml
          uv sync --extra numeric-orbits --extra mojito
```

Leave the `--no-extra numeric-orbits` lines as they are (they install no extras already).

- [ ] **Step 4: README**

In the Install section, replace the two-line sentence and code block that start "Loading tabulated spacecraft" with:

````markdown
Loading tabulated spacecraft ephemerides (`enchilada.orbits.NumericOrbit`)
needs the `numeric-orbits` extra; reading Mojito L1 files and decimating
(`enchilada.mojito`, `enchilada.resample`) needs the `mojito` extra:

```sh
uv sync --extra numeric-orbits   # adds h5py, scipy, lisaorbits
uv sync --extra mojito           # adds h5py, scipy (no lisaorbits)

# For the notebooks, install the orbit and example extras together:
uv sync --extra numeric-orbits --extra examples
uv run --no-sync jupyter lab examples/demo.ipynb
```
````

Insert this section before `## Development`:

````markdown
## Loading Mojito L1 data

`enchilada.mojito.load_mojito_l1` reads a DDPC Mojito L1 HDF5 file (Common
Dataset 1, Light and Heavy) into a time-domain `L1Data`, and
`enchilada.resample.decimate` reduces the 4 Hz record to a rate a campaign can
run at. Both need the `mojito` extra.

```python
from enchilada import decimate, load_mojito_l1

observed = load_mojito_l1(
    "mojito_light_731d_0.25s_L1_0_0_20251218T000909441368Z.h5",
    channels=("A", "E", "T"),  # any of A, E, T, X, Y, Z; dataset tdis/<name>2
    start_offset_s=0.0,  # seconds after the file's first sample
    duration_s=30 * 86400.0,  # None reads to the end of the file
)  # orbit=True (default) attaches the file's ephemeris
campaign_data = decimate(observed, 10)  # 4 Hz -> 0.4 Hz
```

Conventions the loader fixes:

- **Units.** The file stores TDI in total-frequency units (Hz). The loader
  divides by the root attribute `laser_frequency` (281.6 THz) and labels the
  result `physical_observable="fractional_frequency"`, so templates in the
  usual fractional-frequency convention need no scaling. `tdi_generation` is
  `"2.0"`.
- **Clock.** `start_time_gps` is `t0 + n0 * dt` from `tdis/sampling`, on the
  file's clock: TCB seconds after the LISA reference epoch, not GPS. The orbit
  table is on the same clock, so the span check and block response code agree.
- **Window.** The start offset rounds to the nearest sample and the duration
  rounds down to whole samples; a window outside the file is refused with the
  file's span in the message.
- **Flags.** Any nonzero `tdi_flags` value in the window is refused, with the
  count and the first flagged sample; enchilada has no gap mask (see
  [Known limitations](#known-limitations)). Mojito Light sets no flags.
- **Channels.** The stored A/E/T equal the orthonormal combination of X/Y/Z
  that `Covariance.in_basis` uses, so either triple converts consistently.
- **Orbit.** With `orbit=True` (default), `NumericOrbit.from_hdf5` reads the
  file's `orbits` group; its BCRS positions are rotated to the ecliptic frame.

`decimate(data, factor)` applies one fixed anti-alias policy: a Kaiser-window
FIR low-pass with 120 dB stopband attenuation, passband edge at 0.8 times the
new Nyquist frequency and stopband edge at the new Nyquist (about
`78 * factor` taps), applied zero-phase by polyphase filtering. The result has
`ceil(N / factor)` samples at `fs / factor` and the same `start_time_gps`,
because output sample `k` sits at input sample `k * factor`. The first and
last `enchilada.resample.edge_samples(factor)` output samples (about 39) carry
the filter's edge transient; nothing is trimmed. The DDPC's own 0.4 Hz bricks
start 50.5 s after the 4 Hz data, so decimated output does not line up sample
for sample with them. See
[`examples/load_mojito.py`](examples/load_mojito.py).
````

In Known limitations, extend the first bullet:

```markdown
- **No data-quality / gap mask.** Wheel requires complete finite observation arrays.
  Fill or trim gaps before constructing a campaign; `load_mojito_l1` refuses a
  window that contains flagged samples for this reason. A future data-gap interface
  must define how masking affects residual arithmetic, sampling grids, and
  windowing.
```

- [ ] **Step 5: README test table**

In `tests/test_docs.py`, extend the import and `_CALLABLES`:

```python
from enchilada import BlockState, L1Data, Wheel, decimate, load_mojito_l1
```
```python
_CALLABLES = {
    "BlockState": BlockState,
    "L1Data": L1Data,
    "Wheel": Wheel,
    "decimate": decimate,
    "load_mojito_l1": load_mojito_l1,
    "check_block": None,  # filled below; testing imports lazily
}
```

(The file's current line 19 is `from enchilada import BlockState, L1Data, Wheel`.)

- [ ] **Step 6: CHANGELOG**

Under `## [0.4.0] — Unreleased` → `### Added`, after the `Grid` bullet:

```markdown
- `enchilada.mojito.load_mojito_l1`: reads a DDPC Mojito L1 HDF5 file (TDI
  2.0 `X2`…`T2` stored in Hz) into a time-domain `L1Data` in fractional
  frequency on a chosen window, with the file's `tdis/sampling` clock in
  `start_time_gps` (TCB seconds after the LISA epoch) and its orbit table
  attached; a window with nonzero `tdi_flags` is refused.
- `enchilada.resample.decimate`: anti-aliased rate reduction of time-domain
  `L1Data` by an integer factor (Kaiser FIR, 120 dB, passband to 0.8 of the
  new Nyquist, zero-phase) keeping `start_time_gps`; `edge_samples` gives the
  transient length. Both functions are exported from `enchilada`.
- Optional extra `mojito` (h5py, scipy) for loading data without lisaorbits.
```

- [ ] **Step 7: Run everything**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest -q --cov=enchilada`
Expected: 0 failed; TOTAL coverage ≥ 95 %.

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy && uv lock --check`
Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml .github/workflows/release.yml README.md CHANGELOG.md tests/test_docs.py
git commit -m "mojito extra, CI lanes, README and CHANGELOG for the Mojito loader

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage.** §4 loader → Task 2 (window arithmetic, channel map, rate check, flags, units, orbit, errors). §5 decimate → Task 1 (policy numbers, alignment, edge transient, refusals, carry-over). §6 packaging → Task 4 (extra, lock, CI, exports in Tasks 1–2). §7 tests → Tasks 1–3 (synthetic fixture in conftest, real-file opt-in, example test). §8 example and docs → Tasks 3–4. §9 deferred → nothing to build.
- **Placeholders.** None; every code step is complete.
- **Type consistency.** `load_mojito_l1(path, *, channels, start_offset_s, duration_s, orbit)` and `decimate(data, factor)` are used with the same names in the tests, the example, the README block and `_CALLABLES`; `edge_samples` and `anti_alias_taps` match between module and tests; the `__all__` ending is identical in Tasks 1 and 2 (Task 2 adds `load_mojito_l1` between `decimate` and `replace`).
