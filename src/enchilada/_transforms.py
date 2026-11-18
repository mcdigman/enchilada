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
    if source_domain == "wdm" and source_wdm_grid is None:
        raise ValueError("WDM source requires wdm_grid")
    if target_domain == "wdm" and target_wdm_grid is None:
        raise ValueError("WDM target requires wdm_grid")
    sample_interval_s = 1.0 / sample_rate_hz
    channel_names = tuple(channel_data)
    stacked = np.stack([channel_data[name] for name in channel_names])
    if "wdm" in (source_domain, target_domain):
        # domania widens narrow dtypes itself; the cast keeps the dtype handed
        # to the seam explicit (complex128 from frequency, float64 otherwise).
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
