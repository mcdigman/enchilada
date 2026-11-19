"""Load a window of a Mojito-layout L1 file and build its noise covariance.

Both forge notebooks import this module, so the data side of a block is
written once. `load_window` goes through enchilada's loader and decimator and
lands on the grid a block will be registered on; `analytic_covariance` turns
an analytic one-sided PSD into a `Covariance` on that grid. Nothing here is
specific to the placeholder block.
"""

import os
from collections.abc import Callable, Mapping

import numpy as np

from enchilada import Covariance, L1Data, decimate, load_mojito_l1, transform

PSD = Callable[[np.ndarray], np.ndarray]
"""A one-sided PSD: frequencies in Hz (an array) -> PSD values (an array)."""

PROBE_DURATION_S = 60.0
"""Length of the one-channel probe load that reads the file's sampling interval."""


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
    record has exactly `Nf * Nt` samples; `num_time_divisions` must be even.
    The first and last `enchilada.resample.edge_samples(factor)` samples (40)
    of the decimated record carry the anti-alias filter's edge transient;
    nothing is trimmed.
    """
    channels = tuple(channels)
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
        if num_time_divisions % 2:
            raise ValueError(
                f"num_time_divisions must be even, got {num_time_divisions}"
            )
        probe = load_mojito_l1(
            path, channels=channels[:1], duration_s=PROBE_DURATION_S, orbit=False
        )
        count = factor * num_frequency_divisions * num_time_divisions
        # The loader floors duration_s / dt; the extra half sample keeps roundoff
        # from dropping the last sample.
        duration_s = (count + 0.5) * probe.dt
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
    if not np.isfinite(level) or level <= 0:
        raise ValueError(f"level must be finite and positive, got {level!r}")

    def psd(f: np.ndarray) -> np.ndarray:
        return np.full(np.shape(f), level, dtype=float)

    return psd


def analytic_covariance(data: L1Data, psd: PSD | Mapping[str, PSD]) -> Covariance:
    """A `Covariance` on `data`'s grid from a one-sided PSD, channels independent.

    `psd` is one callable for every channel or a mapping from channel name to
    callable. With `X = dt * rfft(x)` a stationary PSD `S` gives rfft-bin
    variance `(Tobs / 2) * S(f_k)`; `S` is evaluated at `max(f_k, df)`, so the
    DC bin takes the first bin's value. A WDM cell at band frequency `f` has
    variance `S(f) / 2`; row 0 holds DC (evaluated at the band spacing) in its
    even columns and the Nyquist band in its odd columns, and the PSD is
    treated as flat across a band. The time grid has one variance per sample,
    `S * fs / 2`, so only a white PSD is accepted there.
    """
    channels = tuple(data.channel_names)
    if callable(psd):
        per_channel = {name: psd for name in channels}
    else:
        missing = [name for name in channels if name not in psd]
        if missing:
            raise ValueError(f"psd mapping has no entry for channels {missing}")
        per_channel = {name: psd[name] for name in channels}
    bins = np.maximum(np.fft.rfftfreq(data.N, data.dt), data.df)
    pairs: dict[str, np.ndarray] = {}
    if data.data_domain == "frequency":
        for name, fn in per_channel.items():
            pairs[name] = 0.5 * data.Tobs * np.asarray(fn(bins), dtype=float)
    elif data.data_domain == "wdm":
        assert data.wdm_grid is not None
        rows, columns = data.wdm_grid.array_shape
        band = np.arange(rows, dtype=float)[:, None] * np.ones((1, columns))
        band[0, 0::2] = 1.0  # DC, evaluated at the band spacing
        band[0, 1::2] = rows  # the Nyquist band
        frequencies = band * data.fs / (2 * rows)
        for name, fn in per_channel.items():
            pairs[name] = 0.5 * np.asarray(fn(frequencies), dtype=float)
    else:
        for name, fn in per_channel.items():
            values = np.asarray(fn(bins), dtype=float)
            if not np.allclose(values, values[0], rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"psd for channel {name!r} is not white: a coloured PSD has "
                    "no per-sample covariance on the time grid; register the "
                    "block on the frequency or WDM grid, or use white_psd"
                )
            pairs[name] = np.full(data.N, 0.5 * values[0] * data.fs)
    return Covariance.like(data, pairs)
