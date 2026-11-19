"""Load a window of a Mojito-layout L1 file and build its noise covariance.

Both forge notebooks import this module, so the data side of a block is
written once. `load_window` goes through enchilada's loader and decimator and
lands on the grid a block will be registered on; `analytic_covariance` turns
an analytic one-sided PSD into a `Covariance` on that grid. Nothing here is
specific to the placeholder block.
"""

import os
from collections.abc import Callable, Mapping

import domania
import numpy as np

from enchilada import (
    Covariance,
    L1Data,
    WDMGrid,
    decimate,
    load_mojito_l1,
    transform,
)

PSD = Callable[[np.ndarray], np.ndarray]
"""A one-sided PSD: frequencies in Hz (an array) -> PSD values of the same
shape, or one scalar for a white PSD."""


def _sampling_interval_s(path: str | os.PathLike[str]) -> float:
    """The file's native sampling interval, from the ``tdis/sampling``
    attribute `load_mojito_l1` reads, without loading any samples."""
    import h5py

    with h5py.File(path, "r") as file:
        if "tdis/sampling" not in file:
            raise ValueError(
                f"{os.fspath(path)} has no tdis/sampling group; is it a Mojito L1 file?"
            )
        return float(file["tdis/sampling"].attrs["dt"])


def load_window(
    path: str | os.PathLike[str],
    *,
    factor: int,
    channels: tuple[str, ...] = ("A", "E", "T"),
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
    domain: str = "time",
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
) -> L1Data:
    """Load `channels` from a Mojito-layout file, decimate by `factor` and
    return the window in `domain`.

    Time and frequency windows take `duration_s` (native seconds from
    `start_offset_s`). A WDM window takes both division counts instead and
    derives its native duration, `factor * Nf * Nt * dt`, so the decimated
    record has exactly `Nf * Nt` samples; both counts must be positive even
    integers. The first and last `enchilada.resample.edge_samples(factor)`
    samples (40) of the decimated record carry the anti-alias filter's edge
    transient; nothing is trimmed.
    """
    channels = tuple(channels)
    if isinstance(factor, bool) or not isinstance(factor, int | np.integer):
        raise TypeError(f"factor must be an integer, got {factor!r}")
    if factor < 2:
        raise ValueError(f"factor must be at least 2, got {factor}")
    if domain not in ("time", "frequency", "wdm"):
        raise ValueError(f"domain must be 'time', 'frequency' or 'wdm', got {domain!r}")
    if domain == "wdm":
        if num_frequency_divisions is None or num_time_divisions is None:
            raise ValueError(
                "a WDM window needs num_frequency_divisions and num_time_divisions"
            )
        if duration_s is not None:
            raise ValueError(
                "a WDM window derives its duration from the division counts; "
                "leave duration_s as None"
            )
        grid = WDMGrid(
            num_frequency_divisions=num_frequency_divisions,
            num_time_divisions=num_time_divisions,
        )
        # The loader floors duration_s / dt; the extra half sample keeps roundoff
        # from dropping the last sample.
        duration_s = (factor * grid.num_time_samples + 0.5) * _sampling_interval_s(path)
    elif duration_s is None:
        raise ValueError(f"a {domain} window needs duration_s")
    elif num_frequency_divisions is not None or num_time_divisions is not None:
        raise ValueError("division counts apply only to domain='wdm'")
    native = load_mojito_l1(
        path, channels=channels, start_offset_s=start_offset_s, duration_s=duration_s
    )
    data = decimate(native, factor)
    if domain == "time":
        return data
    if domain == "frequency":
        return transform(data, "frequency")
    return transform(
        data,
        "wdm",
        num_frequency_divisions=num_frequency_divisions,
        num_time_divisions=num_time_divisions,
    )


def white_psd(level: float) -> PSD:
    """One-sided PSD equal to `level` at every frequency.

    White noise of per-sample variance `sigma**2` at sample rate `fs` has
    `level = 2 * sigma**2 / fs`.
    """
    try:
        value = float(level)
    except (TypeError, ValueError):
        value = np.nan
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"level must be finite and positive, got {level!r}")
    level = value

    def psd(f: np.ndarray) -> np.ndarray:
        return np.full(np.shape(f), level, dtype=float)

    return psd


def _evaluate(psd: PSD, frequencies_hz: np.ndarray, name: str) -> np.ndarray:
    """`psd` at `frequencies_hz` as a float array of that shape; a scalar
    result (a white PSD written as a constant) broadcasts."""
    values = np.asarray(psd(frequencies_hz), dtype=float)
    try:
        values = np.array(np.broadcast_to(values, np.shape(frequencies_hz)))
    except ValueError:
        raise ValueError(
            f"psd for channel {name!r} returned shape {values.shape} for "
            f"frequencies of shape {np.shape(frequencies_hz)}"
        ) from None
    if not np.all(np.isfinite(values)):
        raise ValueError(f"psd for channel {name!r} is not finite at every frequency")
    if np.any(values < 0):
        raise ValueError(f"psd for channel {name!r} is negative at some frequency")
    return values


def _wdm_variance(psd: PSD, data: L1Data, name: str) -> np.ndarray:
    """Each cell's variance: the PSD averaged across its band with the WDM
    window's own weights (domania), so notches and steep walls are handled."""
    assert data.wdm_grid is not None
    rows, columns = data.wdm_grid.array_shape

    def model(f: np.ndarray, t: np.ndarray) -> np.ndarray:
        # domania hands over a 3-D frequency array; PSDs written for 1-D input
        # get a flat one.
        frequencies_hz = np.maximum(f, data.df)
        values = _evaluate(psd, frequencies_hz.ravel(), name)
        return values.reshape(frequencies_hz.shape)[None, None]

    return domania.wdm_covariance(model, Nf=rows, Nt=columns, dt=data.dt)[0, 0]


def analytic_covariance(data: L1Data, psd: PSD | Mapping[str, PSD]) -> Covariance:
    """A `Covariance` on `data`'s grid from a one-sided PSD, channels independent.

    `psd` is one callable for every channel or a mapping from channel name to
    callable; it may return one scalar. With `X = dt * rfft(x)` a stationary
    PSD `S` gives rfft-bin variance `(Tobs / 2) * S(f_k)`; `S` is evaluated at
    `max(f_k, df)`, so the DC bin takes the first bin's value. A WDM cell's
    variance is `S / 2` averaged across its band with the Meyer window's
    weights (`domania.wdm_covariance`, exact for stationary noise), with the
    same floor at `df`; row 0 holds DC in its even columns and the Nyquist
    band in its odd columns. The time grid has one variance per sample,
    `S * fs / 2`, so only a white PSD is accepted there.

    `decimate`'s anti-alias filter rolls off above 0.8 of the new Nyquist
    frequency, so the PSD of the native noise overstates the variance of the
    cells above that, and the white time-grid variance by about 14 % (the
    decimated variance is 12 % below `S * fs / 2`).
    """
    channels = tuple(data.channel_names)
    if callable(psd):
        per_channel = {name: psd for name in channels}
    else:
        missing = [name for name in channels if name not in psd]
        if missing:
            raise ValueError(f"psd mapping has no entry for channels {missing}")
        per_channel = {name: psd[name] for name in channels}
    pairs: dict[str, np.ndarray] = {}
    for name, fn in per_channel.items():
        if data.data_domain == "wdm":
            variance = _wdm_variance(fn, data, name)
        else:
            bins = np.maximum(np.fft.rfftfreq(data.N, data.dt), data.df)
            values = _evaluate(fn, bins, name)
            if data.data_domain == "frequency":
                variance = 0.5 * data.Tobs * values
            else:
                if not np.allclose(values, values[0], rtol=1e-9, atol=0.0):
                    raise ValueError(
                        f"psd for channel {name!r} is not white: a coloured PSD "
                        "has no per-sample covariance on the time grid; register "
                        "the block on the frequency or WDM grid, or use white_psd"
                    )
                variance = np.full(data.N, 0.5 * values[0] * data.fs)
        if np.any(variance <= 0):
            raise ValueError(
                f"psd for channel {name!r} gives a non-positive variance on the "
                f"{data.data_domain} grid"
            )
        pairs[name] = variance
    return Covariance.like(data, pairs)
