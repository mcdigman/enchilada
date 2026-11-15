# Mojito L1 Loader and Decimation — Design

**Date:** 2026-11-14
**Status:** approved in conversation; implementation plan follows
**Builds on:** the 0.4.0 `Grid`/`Covariance` work on branch `covariance-simplification` (PR #6)

## 1. Goal

Give enchilada a faithful reader for the LISA DDPC Mojito L1 HDF5 product that
returns an `L1Data` ready for a `Wheel`, and a separate decimation step that
turns the native 4 Hz record into a rate a campaign can run at. Both live in
the package behind an optional extra, tested without the 33 GB file.

## 2. Decisions already made

Recorded so the plan and the implementers do not re-ask them.

| Decision | Choice |
|---|---|
| Target format | DDPC Common Dataset 1 Light ("Mojito Light") L1 file (LISA-DDPC-SEG-TN-008 v00_02); the same layout Mojito Heavy and later L1 products use |
| Units of the loaded data | fractional frequency: stored Hz divided by the file's root attribute `laser_frequency`; `physical_observable="fractional_frequency"` |
| Data volume | the loader reads a time window and channel subset at the native rate; decimation is a separate function |
| Placement | `src/enchilada/mojito.py` for the loader and `src/enchilada/resample.py` for `decimate`; both exported from `enchilada` |
| Dependencies | new extra `mojito = ["h5py>=3.0", "scipy>=1.8"]`; imports are lazy, the core stays NumPy-only |
| Anti-alias policy | fixed, documented, no knobs (section 5) |
| Deferred | noise-estimate reader, eta channels, brick stacking, gap handling (section 9) |

## 3. The file, as verified

Facts read from `mojito_light_731d_0.25s_L1_0_0_20251218T000909441368Z.h5`
(35.6 GB) and the DDPC technical note. The synthetic test fixture reproduces
this layout exactly.

Root attributes: `laser_frequency` (float, 2.816e14 Hz), `lolipops_version`
("0.4.5"), `pipeline_name`, `modulation_frequencies`.

Group `tdis/`:

| Member | Shape / dtype | Meaning |
|---|---|---|
| `sampling/` (attrs `t0`, `dt`, `fs`, `duration`, `size`) | t0 = 97 729 889.327 664 s, dt = 0.25 s, fs = 4 Hz, size = 252 465 200 | time grid of every TDI dataset; `duration = size * dt` |
| `X2`, `Y2`, `Z2` | `(size,)` float64, Hz | second-generation Michelson TDI, total-frequency units |
| `A2`, `E2`, `T2` | `(size,)` float64, Hz | quasi-orthogonal combinations; equal to A = (Z − X)/√2, E = (X − 2Y + Z)/√6, T = (X + Y + Z)/√3 of the stored X2/Y2/Z2 to 1e-15 relative, i.e. exactly enchilada's `in_basis` matrix |
| `eta_12` … `eta_32` | `(size,)` float64, Hz | single-link intermediary variables (not read) |
| `tdi_flags`, `eta_flags` | `(size,)` int8 | 0 = usable, 1 = do not use; all zero in Mojito Light |

Group `orbits/`: `sampling/` attrs (t0 = 61 171 239.327 664 s, dt = 500 000 s,
size = 274) and `sc_position_{1,2,3}` shaped `(274, 3)` in BCRS metres, plus
`sc_velocity_{1,2,3}`. This is the layout `NumericOrbit.from_hdf5` already
reads with its defaults (`group_path="orbits"`, `coordinate_frame="equatorial"`).
The table spans 61.2–197.7 Ms and covers the TDI span 97.7–160.8 Ms.

Other groups, not read by this work: `ltts/`, `noise_estimates/` (daily 3×3
cross-spectra for XYZ and AET on a 1000-point log grid from 1e-5 to 1 Hz, in
Hz²/Hz, plus a 6×6 for eta), `scet_wrt_tcb/`, `ttl/`.

Clock: all `t0` values are TCB seconds after the LISA reference epoch
(Newtonian time, no TPS). The orbit table and the TDI data share this clock.

Measured on the real file: every sample finite, no zeros, no flags set;
A2 RMS 6.2e-4 Hz, i.e. 2.2e-18 in fractional frequency; `dt * fs == 1.0`
exactly; one channel of 252 M samples reads in about one second.

## 4. Loader: `enchilada.mojito.load_mojito_l1`

```python
def load_mojito_l1(
    path: str | os.PathLike[str],
    *,
    channels: tuple[str, ...] = ("A", "E", "T"),
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
    orbit: bool = True,
) -> L1Data: ...
```

Behaviour, in the order the function performs it:

1. **Open and read the time grid.** `t0`, `dt`, `fs`, `size` from
   `tdis/sampling`. Refuse with `ValueError` if `dt * fs` differs from 1 by more
   than 1e-9 relative, naming both values. `sample_rate_hz` is `fs`.
2. **Resolve the window.** `n0 = round(start_offset_s / dt)`;
   `count = size - n0` when `duration_s` is None, else `floor(duration_s / dt)`.
   Refuse with `ValueError` when `n0 < 0`, `n0 >= size`, `count < 1`, or
   `n0 + count > size`; the message gives the requested window and the file
   span `[t0, t0 + (size - 1) * dt]` in seconds, computed from `size` and `dt`
   (not from the `duration` attribute, which is one sample longer than the last
   timestamp). `start_time_gps = t0 + n0 * dt`, multiplied exactly as
   `NumericOrbit.from_hdf5` builds its grid, so the orbit span check compares
   like with like.
3. **Resolve channels.** Each name must be one of `A E T X Y Z`; the dataset is
   `tdis/<name>2`. An unknown name raises `ValueError` listing the six; a name
   whose dataset is missing from the file raises `ValueError` naming the
   dataset path. Duplicates are refused by `L1Data` itself. `tdi_generation`
   is `"2.0"`.
4. **Check flags.** Every dataset about to be read (`tdis/tdi_flags` and each
   requested channel) must have shape `(size,)`; otherwise refuse with
   `ValueError` naming the dataset, its shape and `size`, because h5py clips
   an over-long slice silently and a short record would pass the flag gate
   unchecked. `laser_frequency` must be a positive finite number, else
   `ValueError`. Then read `tdis/tdi_flags[n0 : n0 + count]`. Any nonzero value
   raises `ValueError` with the count of flagged samples and the index of the
   first one (relative to the file), and says enchilada carries no gap mask.
   Nothing is returned partially.
5. **Read channels.** For each requested channel, read the window slice into a
   float64 array and divide by `laser_frequency` in place. Peak memory is one
   window per channel plus the dictionary being built (no second copy).
6. **Orbit.** With `orbit=True`, call
   `NumericOrbit.from_hdf5(path, group_path="orbits", coordinate_frame="equatorial")`
   and pass it as `orbit_ephemeris`. The equatorial-to-ecliptic rotation is the
   J2000 obliquity rotation already in `orbits.py`; verified on the real file
   to give the expected ±1.2e9 m ecliptic z-excursion of a spacecraft. With
   `orbit=False`, `orbit_ephemeris` is None.
7. **Return** `L1Data(channel_data, sample_rate_hz=fs, tdi_generation="2.0",
   physical_observable="fractional_frequency", channel_names=channels,
   start_time_gps=start_time_gps, orbit_ephemeris=orbit_or_none)`. The
   constructor's validation (lengths, dtype, orbit span) is the last gate.

Documented in the docstring and README: `start_time_gps` carries the file's
TCB clock, not GPS; the field is not renamed. The A/E/T datasets equal
`Covariance.in_basis`'s combination of X/Y/Z, so a campaign may load either
triple and convert covariances consistently.

Errors use the enchilada style: say what was given, what was expected, and
what to do.

## 5. Decimation: `enchilada.resample.decimate`

```python
def decimate(data: L1Data, factor: int) -> L1Data: ...
```

- **Input checks.** `data.data_domain` must be `"time"`, else `ValueError`
  ("decimate the time series; transform back first"). `factor` must be an
  `int` (not bool) of at least 2: `TypeError` for the wrong type, `ValueError`
  for a value below 2.
- **Filter policy (fixed).** A Kaiser-window FIR low-pass designed with
  `scipy.signal.kaiserord` / `scipy.signal.firwin` for 120 dB stopband
  attenuation, passband edge at `0.8 * fs_new / 2` and stopband edge at
  `fs_new / 2`, where `fs_new = fs / factor`. In normalised terms the
  transition width is `0.2 / factor` of the input Nyquist, which gives about
  `78 * factor` taps (odd length, symmetric, unit DC gain). Rationale: the OMS
  noise rises as f² above the LISA band, and scipy's default 53 dB Hamming
  filter lets that content fold back into the kept band at a level a noise
  block would see; 120 dB keeps the lower 80 % of the new band clean to 1e-6
  in power.
- **Application.** `scipy.signal.resample_poly(x, up=1, down=factor, window=taps)`
  per channel. This is zero-phase: output sample k is aligned with input
  sample `k * factor`, so `start_time_gps` is unchanged. Output length is
  `ceil(N / factor)`.
- **Edge transient.** Half the filter length, i.e. about `39 * factor` input
  samples or 40 output samples at each end, is affected by the zero padding.
  The docstring states this number. Nothing is trimmed, so the grid stays
  predictable; a campaign that cares trims and re-slices itself.
- **Minimum length.** If `N` is shorter than the filter length, refuse with
  `ValueError` giving both numbers: the polyphase filter cannot be applied to
  a record it does not span. Clean interior samples additionally need `N` of
  at least `2 * edge_samples(factor) * factor`; that is the caller's call.
- **Output.** A new `L1Data` with `sample_rate_hz = fs / factor`,
  `num_time_samples = ceil(N / factor)`, the same `start_time_gps`,
  `channel_names` order, `tdi_generation`, `physical_observable`, and the same
  `orbit_ephemeris` object by reference. Built through the constructor, so it
  is validated (including the orbit span, which can only shrink).
- **Caveat recorded in docs.** The DDPC's own 0.4 Hz bricks start 50.5 s after
  the 4 Hz `t0` (technical note, section 3.1), so output of `decimate` does not
  align sample-for-sample with those bricks.

## 6. Packaging and exports

- `pyproject.toml`: add `mojito = ["h5py>=3.0", "scipy>=1.8"]` under
  `[project.optional-dependencies]`, with a comment that `numeric-orbits`
  already provides both and this extra exists so a data-loading install need
  not pull lisaorbits.
- `uv.lock` regenerated for the new extra (`uv lock`; may need to run outside
  the sandbox).
- `.github/workflows/ci.yml`: every `uv sync ... --extra numeric-orbits` also
  passes `--extra mojito` (the coverage-gate lane, the lowest-direct lane, the
  sdist lane). The core-only lanes stay as they are and skip the new tests.
- `release.yml` mirrors the same `--extra` change where it installs extras.
- `src/enchilada/__init__.py`: export `decimate` and `load_mojito_l1`, add
  both to `__all__`.
- mypy: `h5py.*` and `scipy.*` are already `ignore_missing_imports`.

## 7. Tests

All new tests `pytest.importorskip` h5py (loader) or scipy (decimate) at
module import, so the core-only CI lanes skip them.

### 7.1 `tests/test_mojito.py`

Fixture `mojito_file(tmp_path)` writes a file with the layout of section 3 at
small scale: `laser_frequency = 2.816e14`, `tdis/sampling` attrs
(`t0 = 97729889.327664`, `dt = 0.25`, `fs = 4.0`, `size = N`, `duration = N*dt`)
with `N = 4096`, six `tdis/<ch>2` datasets where X/Y/Z are deterministic
pseudo-random Hz-scale series and A/E/T are the orthonormal combinations,
`tdis/tdi_flags` int8 zeros, and an `orbits/` group whose `sampling` attrs and
three `(n, 3)` position tables span the TDI window with margin. A second
fixture `mojito_file_flagged` sets one flag at a known index.

Cases:

- channels: default returns A, E, T in that order with `tdi_generation="2.0"`;
  `channels=("X","Y","Z")` returns those; an unknown name raises `ValueError`
  listing the six; a file missing `tdis/T2` raises `ValueError` naming the
  dataset.
- units: loaded values equal the stored values divided by `laser_frequency`;
  `physical_observable == "fractional_frequency"`.
- window: offset and duration resolve to `round` and `floor` of the sample
  counts; omitted duration reads to the end; `start_time_gps == t0 + n0 * dt`
  exactly; a window past the end raises `ValueError` whose message contains
  the file span; a zero-length window is refused.
- rate: `sample_rate_hz == fs`; a file with inconsistent `dt`/`fs` attrs is
  refused.
- consistency: a `size` attribute larger than the datasets, a `tdi_flags` or
  channel dataset shorter than `size`, and a zero `laser_frequency` are each
  refused with `ValueError` naming the offending quantity.
- flags: the flagged fixture raises `ValueError` mentioning the count and the
  first flagged index; a window that excludes the flagged sample loads.
- orbit: `orbit=True` attaches a `NumericOrbit` whose `time_range_gps` covers
  the data; `orbit=False` leaves None; positions are in the ecliptic frame
  (a table built in equatorial coordinates comes back rotated).
- real file, opt-in: when `ENCHILADA_MOJITO_L1` names a file, load one hour
  of A/E/T with the orbit and assert 14400 samples at 4 Hz, `start_time_gps`
  equal to the file's `t0`, every channel RMS between 1e-19 and 1e-17, A equal
  to (Z − X)/√2 of a second load of X/Y/Z to 1e-12 of the signal scale
  (`rtol=1e-12` with `atol=1e-12 * max|A|`, since a pure relative test fails
  at zero crossings by float64 rounding alone), the orbit span
  covering the data, and `decimate(data, 10)` giving 1440 samples at 0.4 Hz.
  Skips otherwise.

### 7.2 `tests/test_resample.py`

- alignment: a sinusoid at 1/50 of the new Nyquist, decimated by 8, matches
  the analytic value at `t0 + k * 8 * dt` to 1e-6 relative away from the
  stated edge transient (catches a phase shift or a wrong start time).
- suppression: a tone at 1.5 times the new Nyquist comes out with amplitude
  below 1e-6 of its input away from the edges.
- grid: `num_time_samples == ceil(N / factor)` for a non-divisible N,
  `sample_rate_hz == fs / factor`, `start_time_gps` unchanged.
- carry-over: channel order, `tdi_generation`, `physical_observable` and the
  orbit object identity are preserved.
- refusals: frequency-domain input, WDM input, `factor=1`, `factor=True`,
  `factor=2.0`, and a record shorter than the filter.

### 7.3 Example test

`tests/test_examples.py` gains a test that runs
`examples/load_mojito.py`'s `main(path, duration_s=1000.0, factor=4)` on the
  synthetic fixture (the one-day default exceeds the fixture) and checks
the printed grid line; skipped without h5py and scipy.

## 8. Example and documentation

- `examples/load_mojito.py`: `main(path, *, duration_s=86400.0, factor=10)`
  loads one day of A/E/T with the orbit, decimates, and prints the grid, each
  channel's RMS, and the orbit span; `__main__` takes the path from `argv[1]`
  or `ENCHILADA_MOJITO_L1`. Simulation stays out of it, per the examples rule.
- README: new section "Loading Mojito L1 data" after "Orbits": the extra, a
  snippet, units (Hz → fractional frequency via `laser_frequency`), the TCB
  clock carried in `start_time_gps`, the flag refusal, and `decimate` with
  the filter numbers and the 0.4 Hz brick caveat. "Known limitations" gains
  one sentence: flagged samples are refused by the Mojito loader because there
  is no gap mask.
- CHANGELOG 0.4.0 "Added": `load_mojito_l1`, `decimate`, the `mojito` extra.

## 9. Deferred

- A reader for `noise_estimates/{XYZ,AET,eta}` returning raw arrays (day
  times, the log-frequency grid, the cross-spectra in Hz²/Hz). Noise models
  belong to noise blocks; add this when a block asks for it.
- The eta single-link channels, LEGO brick stacking (summing bricks), and
  Mojito Heavy gap handling (which needs the gap-mask design the README
  already names as missing).
- TPS time frame: the file is TCB only.

## 10. Self-review

- No placeholders; every behaviour names its inputs, outputs and errors.
- Consistent names across sections: `load_mojito_l1`, `decimate`,
  `start_offset_s`, `duration_s`, `orbit`, `factor`, `mojito` extra.
- Scope is one plan: two small modules, tests, example, docs, packaging.
- Ambiguities resolved: window rounding (round start, floor count), clock
  label (TCB in `start_time_gps`, documented), filter policy (fixed numbers),
  trimming (none).
