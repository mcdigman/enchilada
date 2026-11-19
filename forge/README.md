# Forge

A workbench for building an enchilada `Block` against the inputs a block
really receives: a conditional residual on a chosen grid and a `Covariance`
on that grid, loaded from a simulated Mojito-layout L1 file. Copy this folder
once per block and fill in the skeleton. The tutorial that explains the
contract is [`examples/create_block.ipynb`](../examples/create_block.ipynb);
this folder assumes you have read it.

## The three steps

1. **Fill in `block.py`.** `ForgeBlock` has three methods marked `# EDIT`:
   `_template` renders the signal on the block's grid, `initial_sample`
   draws or renders the start, `sample` advances the state. Two checks,
   `_check_residual` and `_check_covariance`, say what the model needs from
   its inputs. The placeholder is a sinusoid of one amplitude on the
   frequency grid with a conjugate Gaussian sampler, so everything runs
   before you change a line.
2. **Run `block.ipynb`.** It loads a window of the simulated file, builds the
   covariance, calls `initial_sample` and `sample` by hand with asserts on
   the returned state, and runs `check_block`. Edits to `block.py` take
   effect cell by cell.
3. **Run `wheel.ipynb`.** A `Wheel` on the same window, the block registered
   on its grid, a chain collected through `on_cycle_complete`, the posterior
   against a truth you type in, and a trace plot.

## Setup

From the repository root, with h5py (either extra) and matplotlib (arrives
with `numeric-orbits`):

```sh
uv sync --extra numeric-orbits --extra examples
ENCHILADA_MOJITO_L1=/path/to/simulated_L1.h5 uv run --no-sync jupyter lab forge/block.ipynb
```

In a running kernel, `%env ENCHILADA_MOJITO_L1=/path/to/simulated_L1.h5` in a
cell sets it before the settings cell runs.

The notebooks read three environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `ENCHILADA_MOJITO_L1` | the Mojito-layout L1 file | required |
| `FORGE_DURATION_S` | native seconds to load | `86400` (one day) |
| `FORGE_FACTOR` | decimation factor | `10` (4 Hz to 0.4 Hz) |

`data.py` is shared by both notebooks. `load_window` goes through
`load_mojito_l1` and `decimate` and returns the window on the time,
frequency or WDM grid (for WDM, pass both division counts instead of
`duration_s`; the native duration is derived so the decimated record fits the
grid). `white_psd` and `analytic_covariance` turn a one-sided PSD, any
callable `psd(f_hz)` or one per channel, into a `Covariance` on that grid.

## What to replace and what to keep

Replace the model (`_template`, `initial_sample`, `sample`) and the checks'
expectations (the grid, what the likelihood needs from the covariance). Keep
the shape of the contract: a `BlockState` with a template on the input grid,
a complete state after a rejection too, draws from the supplied `rng`,
diagnostics on `block.state` rather than in the ledger, and a fresh instance
per campaign. Collect chains outside the block, as `wheel.ipynb` does. In
the notebooks, change the settings cells and the lines marked `# EDIT` (the
block's constructor, `TRUTH`); the asserts are the contract and name no
parameter of the placeholder. Keep `State.num_sample_calls`: the notebooks
assert on it.

To move the block to another grid, change `DOMAIN` in both notebooks (for
WDM also `NUM_FREQUENCY_DIVISIONS`, the row count; the decimated window
length divided by it must be an even integer), render `_template` on that
grid and update `_check_residual`. Every block receives the campaign's one
covariance, so every block registers on its grid (a covariance does not
translate between grids); another grid means moving `DOMAIN`, and with it
the covariance, for the whole campaign.

## Caveats

- **Time grid, white noise only.** enchilada's time-grid covariance is one
  variance per sample, so `analytic_covariance` accepts only a white PSD
  there and refuses a coloured one. Coloured noise needs the frequency or
  WDM grid.
- **Edge transient.** `decimate` leaves 40 filter-transient samples at each
  end of the record (`enchilada.resample.edge_samples`); nothing is trimmed,
  and on a WDM grid the transient leaks a few columns inward.
- **Anti-alias roll-off.** `decimate` passes the band fully only up to 0.8 of
  the new Nyquist frequency, so the PSD of the native noise overstates the
  cells above that (conservative) and the white time-grid variance by about
  14 %. A WDM cell's variance is the PSD averaged across its band
  (`domania.wdm_covariance`), so notches and steep walls are handled; row 0
  interleaves DC and Nyquist.
- **White stand-in.** The notebooks' `LEVEL` is a white PSD at the loaded
  window's own variance (about 12 % below the in-band PSD, for the reason
  above), a placeholder for the simulation's noise model, and a loud injected
  signal inflates it. On coloured noise the variance-based level is a band
  average dominated by the loudest band, and the diagonal covariance on an
  untapered window ignores leakage on steep spectra, so posterior widths
  mean little until `PSD` is the simulation's model. The GB example's
  `FixedLISANoise().psd` (lisatools, copied beside the notebooks) drops in
  as `PSD` when you have that stack.
