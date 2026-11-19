# Forge: a workbench for building blocks against simulated Mojito data

Date: 2026-11-18, sequencing updated 2026-11-19. Branch: `forge`, stacked on
`block-initial-sample` (3831bfd, PR #9). Status: design approved in
conversation, implementation not started.

## 1. Goal

Give a block author one place to build a `Block` against the inputs a block
really receives: an `L1Data` conditional residual on a chosen grid and a
`Covariance` on that grid, loaded from a simulated Mojito-layout L1 file. The
author copies the `forge/` folder, fills in three marked methods of a skeleton
block, exercises them by hand in a notebook, passes `check_block`, and runs a
short campaign in a second notebook. Everything in the forge runs unchanged in
CI on a tiny synthetic Mojito-layout file, so the forge cannot drift from the
API the way untested templates do.

The forge is written against the `initial_sample` contract (spec
`2026-11-18-block-initial-sample-design.md`): a `Block` has `name`,
`initial_sample(conditional_residual, covariance, *, rng)` and
`sample(conditional_residual, covariance, current_block_state, *, rng)`, and
`Wheel.add(block, *, data_domain=None, ...)` calls `initial_sample` once at
registration. That migration is complete on branch `block-initial-sample`,
open as PR #9 against `main`. The forge is built stacked on it, in its own
git worktree; the forge PR targets `block-initial-sample` and is retargeted
to `main` once PR #9 merges (as PRs #6 to #8 were stacked).

## 2. Decisions

| Question | Decision |
| --- | --- |
| Where does the example data come from? | The author's own simulations, built on real Mojito data, which write Mojito-layout L1 HDF5 files. The forge reads them with `load_mojito_l1` and `decimate`. No synthetic generators and no stored fixture files ship with the forge; the test suite makes its own tiny file with the existing `write_mojito_file`. |
| Where does the noise covariance come from? | An analytic PSD: any callable `psd(f_hz)` (the simulation's noise model, `FixedLISANoise().psd` from the GB example when lisatools is installed, or the forge's `white_psd`), turned into a `Covariance` on the block's grid by `forge/data.py`. No PSD estimation, no noise block. |
| What does a forge session look like? | A skeleton to fill in: `forge/data.py` (shared loader and covariance builder), `forge/block.py` (the skeleton block), `forge/block.ipynb` (workbench with hand checks and `check_block`), `forge/wheel.ipynb` (short campaign). The author copies the folder per block. |
| Where does it live and how is it kept honest? | A top-level `forge/` folder in the enchilada repository with `tests/test_forge.py` driven by a synthetic Mojito-layout file, plus one case gated by a real file through `ENCHILADA_MOJITO_L1`. |
| Where does the block class live? | In `forge/block.py`, because notebooks cannot import each other and both notebooks need it; `block.ipynb` uses autoreload so edits to the module take effect cell by cell. |
| What is the placeholder model? | A sinusoid of one amplitude rendered natively on the frequency grid, a Gaussian prior on the amplitude, a conjugate Gaussian `sample`, so the skeleton runs green on noise-only data with no tuning and shows real numbers in every cell. |
| Time grid and coloured noise? | enchilada's time-grid covariance is one variance per sample, so `analytic_covariance` accepts only a white PSD there and refuses a coloured one with a message naming the frequency and WDM grids. |

## 3. Layout and sequencing

```
forge/
  README.md      purpose, the three steps, environment variables, what to replace, caveats
  data.py        load_window, white_psd, analytic_covariance
  block.py       ForgeBlock: the skeleton with three `# EDIT` methods
  block.ipynb    workbench: load, hand checks, check_block, closing checklist
  wheel.ipynb    short campaign with chain collection
tests/test_forge.py
```

The author copies `forge/` to a working folder per block and edits `block.py`
while running `block.ipynb`; `data.py` is imported by both notebooks, so data
loading and the covariance are written once. The forge sits beside
`examples/create_block.ipynb`, which stays the teaching tutorial; the forge
README points there for the reasoning behind the contract.

No change to `src/enchilada`. Both notebooks and `tests/test_forge.py` import
the forge modules from the `forge/` folder (the notebooks run with the folder
as their working directory; the test loads them with `importlib` the way
`tests/test_examples.py` loads example scripts).

## 4. `forge/data.py`

### `load_window`

```python
def load_window(
    path,
    *,
    factor: int,
    channels: tuple[str, ...] = ("A", "E", "T"),
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
    domain: str = "time",
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
) -> L1Data:
```

- Loads `channels` from the Mojito-layout file at `path` with `load_mojito_l1`
  (its defaults otherwise, so the orbit table is attached), decimates by
  `factor` with `enchilada.decimate` (`factor >= 2`, as `decimate` requires),
  and returns the window in `domain`.
- `domain="time"` or `"frequency"`: `duration_s` is required; the loaded
  window covers `duration_s` native seconds from `start_offset_s`, and a
  frequency window is `transform(decimated, "frequency")`.
- `domain="wdm"`: both division counts are required and must be positive
  even integers (validated through `WDMGrid` before any file I/O);
  `duration_s` must be `None`. The native duration is derived as
  `factor * num_frequency_divisions * num_time_divisions * dt`, with `dt`
  read from the file's `tdis/sampling` attribute (no samples are loaded, so
  a flagged sample outside the window or a file shorter than a minute does
  not matter), so the decimated record has exactly `Nf * Nt` samples and
  `transform(decimated, "wdm", num_frequency_divisions=Nf)` lands on the
  requested grid. Any other `domain` raises `ValueError`; `factor` is
  checked up front (an integer of at least 2, as `decimate` requires).
- The 40 filter-transient samples at each end of a decimated record
  (`enchilada.resample.edge_samples`) are documented in the README, not
  trimmed.

### `white_psd`

```python
def white_psd(level: float):
    """One-sided PSD that is `level` at every frequency."""
```

Returns a callable `f -> np.full(np.shape(f), level)`. White noise of
per-sample variance `sigma**2` at rate `fs` has `level = 2 * sigma**2 / fs`.

### `analytic_covariance`

```python
def analytic_covariance(data: L1Data, psd) -> Covariance:
```

`psd` is a callable `psd(f_hz) -> one-sided PSD` applied to every channel, or
a mapping from channel name to callable; a mapping missing one of
`data.channel_names` raises `ValueError`. A callable may return one scalar
(a white PSD written as a constant broadcasts); a non-finite or negative
value, a result of another shape, or a zero variance anywhere on the grid
raises `ValueError` naming the channel. Channels are independent (no cross
terms). The result is `Covariance.like(data, {channel: array})`, so it is on
`data`'s grid and validated by the constructor.

Per grid, with `N = data.num_time_samples`, `fs`, `df = data.df`, `Tobs`:

- Frequency: bin `k` of the `N // 2 + 1` rfft bins sits at `f_k = k * df`
  and gets `(Tobs / 2) * S(max(f_k, df))`. The floor keeps `S(0)` out of the
  evaluation; it affects only the DC bin.
- WDM: band spacing `dF = fs / (2 * Nf)`. Row `i >= 1` is band `i` for every
  column. Row 0 holds DC in its even columns and the Nyquist band
  `Nf * dF = fs / 2` in its odd columns, as `WDMGrid` documents. Each cell
  gets `S / 2` averaged across its band with the Meyer window's own weights,
  through `domania.wdm_covariance` (exact for stationary noise; about 1 ms
  for a day at 0.4 Hz), with `S` evaluated at `max(f, df)` as on the
  frequency grid. For white noise this is the per-cell variance domania's
  normalisation gives (`white_covariance` in `tests/conftest.py`: cell
  variance `sigma**2 / fs` with `S = 2 sigma**2 / fs`). The band-centre
  value `S(i * dF) / 2` was the first design; it is off by a factor of
  about 200 at a notch of a toy TDI2-shaped PSD on a coarse grid (`Nf = 64`
  at 0.4 Hz) and by 0.4 on an `f**-4` wall (checked against domania and a
  Monte Carlo), which is why the average is used.
- Time: the PSD is evaluated at the rfft frequencies `max(k * df, df)`; if
  the values are not all equal (`np.allclose` to the first) the function
  raises `ValueError("a coloured PSD has no per-sample covariance on the time
  grid; register the block on the frequency or WDM grid, or use white_psd")`.
  Otherwise every sample gets `S * fs / 2`.

## 5. `forge/block.py` and `forge/block.ipynb`

### `ForgeBlock`

```python
@dataclass
class ForgeBlock:
    name: str = "forge"
    frequency_hz: float = 1e-3
    prior_std: float = 1e-17
    start_amplitude: float | None = None

    @dataclass
    class State:
        num_sample_calls: int = 0
        last_posterior_mean: float | None = None
        last_posterior_std: float | None = None

    state: State = field(default_factory=State, init=False)
```

`__post_init__` checks the name is non-empty, `frequency_hz` and `prior_std`
are finite and positive, and `start_amplitude` is finite when given. The
`prior_std` default is wide next to the likelihood on Mojito-level data
(`1e-20` would give a prior precision of `1e40`, above the likelihood's
`6e38` on the test fixture, making the posterior the prior's). Three
methods and two checks carry an `# EDIT` marker and a one-line comment
saying what to replace them with:

- `_template(conditional_residual, amplitude) -> BlockState`: the signal on
  the block's grid. Placeholder: `_check_residual` first, then the time axis
  `t = start_time_gps + arange(N) / fs` from the residual's record metadata,
  `h = dt * rfft(sin(2 pi frequency_hz t))` as the per-channel basis (the same
  real sinusoid in every channel), and
  `conditional_residual.block_state({ch: amplitude * h}, parameters={"amplitude": amplitude})`.
  Replace with a real waveform, or re-render on another grid.
- `initial_sample(conditional_residual, covariance, *, rng) -> BlockState`:
  `start_amplitude` when configured, else `rng.normal(0.0, prior_std)`,
  rendered with `_template`. Replace the prior draw with your model's.
- `sample(conditional_residual, covariance, current_block_state, *, rng) -> BlockState`:
  the conjugate Gaussian draw. With `d_k` the residual bin, `h_k` the unit
  basis bin, `C_k` the covariance bin from `covariance.pair(ch, ch)`, and
  weights `w_k = 1` for the DC and Nyquist bins and `w_k = 2` otherwise (the
  one-sided rfft of real data counts interior bins twice; this is the
  `4 df |r|**2 / S` inner product of the GB example in covariance units),
  summed over channels:
  `precision = 1 / prior_std**2 + sum(w_k |h_k|**2 / C_k)`,
  `mean = sum(w_k Re(conj(h_k) d_k) / C_k) / precision`,
  `amplitude = rng.normal(mean, precision**-0.5)`. Records the mean and
  standard deviation and increments `num_sample_calls` on `self.state`,
  returns `_template(conditional_residual, amplitude)`. Replace with your
  sampler; keep the rule that it returns a complete state after acceptance
  and rejection alike.

Two checks are shared. `_check_residual(conditional_residual)`, called by
`_template` (so by `initial_sample` and `sample` alike), requires the
frequency grid (`ValueError` naming `data_domain='frequency'`) and
`frequency_hz` below Nyquist. `_check_covariance(conditional_residual,
covariance)`, called by `sample` only, requires a covariance (`ValueError`
"needs a noise covariance"), `covariance.check_compatible(conditional_residual)`
to pass, and zero cross terms between the residual's channels (`ValueError`
"independent channels"), the way `SineBlock` in `examples/toy_fit.py` checks.
A prior draw needs no covariance, so `initial_sample` accepts `None`. A
module-level `rfft_weights(num_time_samples)` gives the bin weights: 1 at DC
and at Nyquist when `num_time_samples` is even, 2 elsewhere, and
`inner_product(first, second, conditional_residual, covariance)`, built on
them, is the Whittle inner product `sum_ch sum_k w_k Re(conj(first_k)
second_k) / C_k` of two per-channel frequency-grid arrays. `sample` reads
`precision = 1 / prior_std**2 + <h|h>` and `mean = <h|d> / precision`, and
commits its diagnostics to `state` only after the new state is built; a
replacement likelihood reuses `inner_product` instead of re-deriving the
factor-of-two convention.

### `block.ipynb`

Cells, in order; code cells print one line each so the test can assert on them:

1. Markdown: what the forge is, the three steps, which cells to edit, pointer
   to the tutorial.
2. Code, settings: `%load_ext autoreload` / `%autoreload 2`; imports;
   `MOJITO_PATH = os.environ.get("ENCHILADA_MOJITO_L1")` with a `SystemExit`
   message when unset; `DURATION_S = float(os.environ.get("FORGE_DURATION_S", 86400.0))`;
   `FACTOR = int(os.environ.get("FORGE_FACTOR", 10))`; `DOMAIN = "frequency"`;
   `window = load_window(MOJITO_PATH, factor=FACTOR, duration_s=DURATION_S, channels=("A", "E"))`
   on the time grid; `LEVEL = 2 * mean(var(channel)) / window.fs` as the
   white level that matches the window, marked as the placeholder for the
   simulation's PSD with the lisatools alternative shown in a comment;
   `residual = transform(window, DOMAIN)`;
   `covariance = analytic_covariance(residual, white_psd(LEVEL))`; prints the
   grid summary (domain, channels, `N`, `fs`, `Tobs`).
3. Markdown: step 1, the model; what `_template`, `initial_sample`, `sample`
   do, what `_check_residual` and `_check_covariance` require, and what to
   replace.
4. Code: `from block import ForgeBlock`; `block = ForgeBlock(name="forge", frequency_hz=...)`;
   `assert isinstance(block, Block)`; prints the block.
5. Markdown: step 2, exercise by hand.
6. Code: `rng = np.random.default_rng(0)`;
   `start = block.initial_sample(residual, covariance, rng=rng)`; asserts the
   template keys equal the channel names, shapes equal the residual's, values
   finite, `parameters` has `"amplitude"`;
   `moved = block.sample(residual, covariance, start, rng=rng)`; asserts
   the sample counter advanced by one from its value before the cell (so
   the cell can be re-run); prints both amplitudes and the
   posterior mean and standard deviation.
7. Code (optional plot): `try: import matplotlib.pyplot as plt` and plot
   `|residual.channel_data["A"]|` and `|moved.template["A"]|` against the rfft
   frequencies on log axes; `except ImportError: pass`.
8. Markdown: step 3, the contract check.
9. Code: `check_block(deepcopy(block), residual, num_cycles=3, initial_covariance=covariance, random_seed=1)`;
   asserts the sample counter is unchanged (the copy ran, not the original);
   prints `check_block passed`.
10. Markdown: closing checklist (prior support, likelihood normalisation and
    the bin weights, domain assertions, independent instances per campaign,
    collect chains outside the block) and "continue in wheel.ipynb".

## 6. `forge/wheel.ipynb`

1. Markdown: what the campaign shows; where a second block goes.
2. Code, settings: the same environment variables and `load_window`,
   `LEVEL`, `residual`, `covariance` lines as `block.ipynb` cell 2 (five
   lines, duplicated on purpose so each notebook runs on its own), plus
   `TRUE_AMPLITUDE = None` for the author to fill in.
3. Code: `wheel = Wheel(residual, initial_covariance=covariance, random_seed=0)`;
   `block = ForgeBlock(name="forge", frequency_hz=...)`;
   `wheel.add(block, data_domain=DOMAIN)`; a commented second
   `wheel.add(...)` slot; prints the residual RMS after the start.
4. Code: `chain = []`; `on_cycle_complete` appends
   `wheel.ledger["forge"].parameters["amplitude"]`; `wheel.run(200, on_cycle_complete=...)`;
   drops the first `WARM_UP` cycles (0: the placeholder draws the posterior
   exactly; an MCMC sampler needs some) and prints `posterior amplitude:
   mean +/- std over N cycles` and, when
   `TRUE_AMPLITUDE` is set, the offset in units of the posterior standard
   deviation; prints the residual RMS after the run.
5. Code (optional plot): the trace of `chain`, guarded like `block.ipynb`
   cell 7.
6. Markdown: what to do next (replace the placeholder, register a noise
   block before signal blocks whose starts need its covariance, run
   independent campaigns, checkpointing is out of scope).

## 7. `tests/test_forge.py`

Skip-safe: `pytest.importorskip("h5py", reason=...)` at module level, like
`tests/test_mojito.py`. Modules are loaded with `importlib` from `forge/`.

- Fixture `simulated_file(tmp_path, monkeypatch)`: `write_mojito_file(tmp_path / "simulated.h5", n=...)`
  with `n = 4096` (1024 s at 4 Hz). The test settings are factor 4, a 600 s
  window (600 decimated samples at 1 Hz) and a WDM grid `Nf = 8`, `Nt = 16`
  (512 native samples, longer than the 315-tap anti-alias filter for factor
  4; an `8 x 8` grid would be too short). Sets `ENCHILADA_MOJITO_L1` to the
  file, `FORGE_DURATION_S` and `FORGE_FACTOR` to values that fit it. Returns
  the path.
- `load_window`: a time window has `fs = native / factor` and the expected
  sample count; a frequency window is on the frequency grid with the same
  record; a WDM window derives its duration, has
  `wdm_grid.array_shape == (Nf, Nt)` and equals the hand-run
  `load_mojito_l1` → `decimate` → `transform` pipeline bit for bit; a WDM
  window past a flagged sample and one on a file shorter than a minute both
  load; `duration_s` with `domain="wdm"`, a missing `duration_s` with
  `domain="time"`, an odd division count, and an unknown domain each raise
  `ValueError`, and a bad `factor` is refused before any I/O.
- `analytic_covariance`: on the frequency grid, bins equal
  `(Tobs / 2) * S(max(f_k, df))` for a known coloured callable; for
  `white_psd(2 * sigma**2 / fs)` the result equals conftest's
  `white_covariance(data, sigma**2)` on the time, frequency and WDM grids
  (`np.testing.assert_allclose` on the channel arrays), which pins the three
  normalisations against each other; on the WDM grid a steep red PSD gives
  one value per band, strictly inside the PSD's range over the band's
  support and falling with the band, and a PSD linear in `f` reproduces
  its band-centre value on the interior bands; a constant-returning PSD
  equals `white_psd` on every grid; a non-finite, negative, wrong-shaped or
  zero-variance PSD raises; a coloured PSD on the time grid raises with a
  message containing "frequency or WDM grid"; a per-channel mapping
  applies each callable to its channel and a mapping missing a channel raises.
- `ForgeBlock`: `isinstance(ForgeBlock(), Block)`; `check_block` passes on
  the simulated frequency window with the white covariance; `_check_residual`
  refuses a time-grid residual and a frequency at or above Nyquist, and
  `_check_covariance` refuses a `None` covariance, a covariance on another
  grid and a covariance with a cross term; `inner_product` on the simulated
  window with white noise equals the time-domain sum `sum_t s_t x_t /
  sigma**2` (Parseval); amplitude recovery: add `a_true * h` (with `h` the block's own
  basis at a frequency inside the decimation passband) to the simulated
  frequency window's channels, with `prior_std = 1e-17` so the data decide,
  run a `Wheel` with the matching white covariance for 300 cycles, and assert the chain mean after warm-up lies
  within four posterior standard deviations of `a_true`, within four
  standard errors of the block's recorded posterior mean, with the prior's
  precision below a thousandth of the posterior's, and that the chain
  standard deviation agrees with the block's recorded posterior standard
  deviation to within 25 %.
- Notebooks: every code cell of both notebooks compiles; each notebook's
  code cells run in a subprocess with the fixture's environment,
  `MPLBACKEND=Agg`, `MPLCONFIGDIR` under the test's tmp path and `cwd=forge/`,
  with `%`-magics stripped; the block notebook's stdout contains
  `check_block passed`, the wheel notebook's contains `posterior amplitude:`.
- Real file: `@pytest.mark.skipif(not os.environ.get("ENCHILADA_MOJITO_L1"))`;
  `load_window(real, factor=10, duration_s=3600.0, domain="frequency", channels=("A", "E"))`,
  `analytic_covariance` with `white_psd`, one `ForgeBlock().initial_sample`
  call; asserts the template shape matches the residual.

Gates: the full suite stays green with coverage at or above the existing
95 % floor (coverage measures `src/enchilada` only); ruff check and format
clean on `forge/*.py` and `tests/test_forge.py`; mypy unchanged (its `files`
is `src/enchilada`); the notebooks are never formatted.

## 8. Packaging and documentation

- `pyproject.toml`: add `forge/**/*.py`, `forge/**/*.ipynb` and
  `forge/**/*.md` to the sdist includes next to the `examples/**` entries;
  add `forge` to `[tool.ruff] src`; add `"forge/*.ipynb" = ["I001", "E501"]`
  to the per-file ignores.
- `forge/README.md`: what the forge is for; the three steps (fill in
  `block.py`, run `block.ipynb`, run `wheel.ipynb`); the environment
  variables (`ENCHILADA_MOJITO_L1`, `FORGE_DURATION_S`, `FORGE_FACTOR`); how to
  copy the folder per block; what to replace and what to keep; caveats:
  white-only covariance on the time grid, the 40 transient samples at each
  end of a decimated record, the anti-alias roll-off above 0.8 of the new
  Nyquist frequency (a native PSD overstates the cells above it and the
  white time-grid variance by about 12 %; WDM cells are band averages, so
  notches are handled), the placeholder's white level is a stand-in for the
  simulation's PSD (about 12 % below the in-band PSD for the same reason);
  a pointer
  to `examples/create_block.ipynb` for the contract's reasoning.
- Repository `README.md`: a short "Forge" paragraph after the block
  authoring section pointing at `forge/README.md`.
- `CHANGELOG.md`, `## [0.4.0] — Unreleased`, Added: one bullet for the forge.

## 9. Non-goals

- Reading injected truth (parameters or templates) from the simulation file:
  wait for the simulation's file layout; the notebooks compare against a
  value the author types in.
- Estimating a PSD from the data; a noise block; covariance across grids.
- A multi-block template beyond the commented second slot in `wheel.ipynb`.
- Changes to `examples/create_block.ipynb` or to `src/enchilada`.
- Checkpointing or restarting campaigns.
