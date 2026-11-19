from collections.abc import Mapping

import numpy as np
import pytest

from enchilada import Covariance, L1Data


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_observed(
    rng, *, num_time_samples=64, channel_names=("A", "E", "T"), **overrides
):
    """A small, valid time-domain L1Data to build tests on."""
    fields = dict(
        channel_data={
            ch: rng.standard_normal(num_time_samples) for ch in channel_names
        },
        sample_rate_hz=0.5,
        num_time_samples=num_time_samples,
        channel_names=channel_names,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=0.0,
    )
    fields.update(overrides)
    return L1Data(**fields)


@pytest.fixture
def observed(rng):
    return make_observed(rng)


def const_block_state(residual, value):
    """A block state holding `value` in every sample of every channel."""
    return residual.block_state(
        {ch: np.full_like(arr, value) for ch, arr in residual.channel_data.items()}
    )


def starting_at(block, state):
    """Give `block` an `initial_sample` that returns `state` as it is.

    Test shorthand for a block whose start is a constant built by the test. The
    block keeps its identity, name and local state; `state` must be on the grid
    the block registers on, as a real `initial_sample` return would be.
    """

    def initial_sample(conditional_residual, covariance, *, rng):
        return state

    # object.__setattr__ also works on a frozen dataclass instance.
    object.__setattr__(block, "initial_sample", initial_sample)
    return block


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


# ---- Mojito L1 fixture file -------------------------------------------------

MOJITO_LASER_FREQUENCY_HZ = 2.816e14
MOJITO_T0 = 97729889.327664  # first 4 Hz L1 sample, TCB seconds after the LISA epoch
MOJITO_DT = 0.25
MOJITO_ORBIT_DT = 500000.0
MOJITO_ORBIT_SIZE = 12
MOJITO_ORBIT_T0 = MOJITO_T0 - 2 * MOJITO_ORBIT_DT
MOJITO_ECLIPTIC_Z_M = 1.0e8
_J2000_OBLIQUITY_RAD = 0.40909280422232897
_AET_FROM_XYZ = (
    np.array([[-1.0, 0.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, 1.0]])
    / np.sqrt([2.0, 6.0, 3.0])[:, None]
)


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


def write_mojito_file(
    path, *, n=4096, flag_index=None, drop=(), fs=None, size=None, laser_frequency=None
):
    """Write a tiny file in the Mojito L1 layout and return (xyz, aet) in Hz.

    ``xyz`` and ``aet`` are (3, n) arrays in dataset order X, Y, Z and A, E, T;
    A/E/T are the orthonormal combination of X/Y/Z, as in the real product.
    ``flag_index`` sets one tdi_flags entry to 1; ``drop`` removes datasets
    (e.g. ``("T2",)``); ``fs``, ``size`` and ``laser_frequency`` override the
    corresponding attributes (the datasets keep ``n`` samples).
    """
    import h5py

    rng = np.random.default_rng(1)
    xyz = 6e-4 * rng.standard_normal((3, n))
    aet = _AET_FROM_XYZ @ xyz
    with h5py.File(path, "w") as f:
        f.attrs["laser_frequency"] = (
            MOJITO_LASER_FREQUENCY_HZ if laser_frequency is None else laser_frequency
        )
        f.attrs["lolipops_version"] = "0.4.5"
        f.attrs["pipeline_name"] = "mojito-signal"
        tdis = f.create_group("tdis")
        sampling = tdis.create_group("sampling")
        sampling.attrs["t0"] = MOJITO_T0
        sampling.attrs["dt"] = MOJITO_DT
        sampling.attrs["fs"] = 1.0 / MOJITO_DT if fs is None else fs
        sampling.attrs["size"] = n if size is None else size
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
