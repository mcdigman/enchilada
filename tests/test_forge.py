"""The forge: data helpers, the skeleton block and the two workbench notebooks
run against a simulated Mojito-layout file."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py", reason="the forge loads Mojito-layout files with h5py")

from conftest import (  # noqa: E402
    MOJITO_DT,
    MOJITO_T0,
    white_covariance,
    write_mojito_file,
)
from enchilada import transform  # noqa: E402

FORGE = Path(__file__).resolve().parents[1] / "forge"
FILE_SAMPLES = 4096  # 1024 s at the fixture's 4 Hz
FACTOR = 4
DURATION_S = 600.0  # 2400 native samples -> 600 decimated samples at 1 Hz
WINDOW_SAMPLES = 600
NF, NT = 8, 16  # 512 native samples, longer than the 315-tap filter for factor 4


def load_forge(name):
    """Import forge/<name>.py the way test_examples.py imports example scripts."""
    spec = importlib.util.spec_from_file_location(name, FORGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coloured_psd(f):
    """A red one-sided PSD at Mojito scale."""
    return 1e-36 * (1.0 + (1e-3 / f) ** 2)


@pytest.fixture
def simulated_file(tmp_path, monkeypatch):
    """A tiny Mojito-layout file standing in for the simulation's output, with
    the forge's environment variables pointing at it."""
    path = tmp_path / "simulated.h5"
    write_mojito_file(path, n=FILE_SAMPLES)
    monkeypatch.setenv("ENCHILADA_MOJITO_L1", str(path))
    monkeypatch.setenv("FORGE_DURATION_S", str(DURATION_S))
    monkeypatch.setenv("FORGE_FACTOR", str(FACTOR))
    return path


@pytest.fixture
def forge_data():
    return load_forge("data")


def wdm_options():
    return dict(domain="wdm", num_frequency_divisions=NF, num_time_divisions=NT)


# ---- load_window -----------------------------------------------------------


def test_time_window_is_decimated_from_the_file(simulated_file, forge_data):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S
    )
    assert window.data_domain == "time"
    assert window.channel_names == ("A", "E", "T")
    assert window.num_time_samples == WINDOW_SAMPLES
    assert window.sample_rate_hz == 1.0 / (MOJITO_DT * FACTOR)
    assert window.start_time_gps == MOJITO_T0
    assert window.orbit_ephemeris is not None


def test_frequency_window_is_the_time_window_transformed(simulated_file, forge_data):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S, channels=("A", "E")
    )
    spectrum = forge_data.load_window(
        simulated_file,
        factor=FACTOR,
        duration_s=DURATION_S,
        channels=("A", "E"),
        domain="frequency",
    )
    expected = transform(window, "frequency")
    assert spectrum.data_domain == "frequency"
    assert spectrum.grid == expected.grid
    assert spectrum.start_time_gps == MOJITO_T0
    for name in ("A", "E"):
        np.testing.assert_array_equal(
            spectrum.channel_data[name], expected.channel_data[name]
        )


def test_wdm_window_derives_its_duration_and_lands_on_the_grid(
    simulated_file, forge_data
):
    window = forge_data.load_window(simulated_file, factor=FACTOR, **wdm_options())
    assert window.data_domain == "wdm"
    assert window.wdm_grid.array_shape == (NF, NT)
    assert window.num_time_samples == NF * NT
    assert window.sample_rate_hz == 1.0 / (MOJITO_DT * FACTOR)
    assert window.start_time_gps == MOJITO_T0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(wdm_options(), duration_s=100.0), "leave duration_s"),
        (
            dict(domain="wdm", num_frequency_divisions=NF),
            "num_frequency_divisions and num_time_divisions",
        ),
        (
            dict(domain="wdm", num_frequency_divisions=NF, num_time_divisions=15),
            "must be even",
        ),
        (dict(domain="time"), "needs duration_s"),
        (dict(domain="frequency"), "needs duration_s"),
        (
            dict(domain="time", duration_s=100.0, num_frequency_divisions=NF),
            "only to domain='wdm'",
        ),
        (dict(domain="laplace", duration_s=100.0), "domain must be"),
    ],
)
def test_load_window_refuses_inconsistent_requests(
    simulated_file, forge_data, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        forge_data.load_window(simulated_file, factor=FACTOR, **kwargs)


# ---- white_psd and analytic_covariance ---------------------------------------


def test_white_psd_is_flat_and_refuses_bad_levels(forge_data):
    psd = forge_data.white_psd(3.0)
    np.testing.assert_array_equal(psd(np.array([0.0, 1e-3, 0.5])), 3.0)
    assert psd(np.float64(0.1)).shape == ()
    for level in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            forge_data.white_psd(level)


def test_frequency_bins_carry_half_tobs_times_the_psd(simulated_file, forge_data):
    spectrum = forge_data.load_window(
        simulated_file,
        factor=FACTOR,
        duration_s=DURATION_S,
        channels=("A", "E"),
        domain="frequency",
    )
    covariance = forge_data.analytic_covariance(spectrum, coloured_psd)
    bins = np.arange(spectrum.N // 2 + 1) * spectrum.df
    expected = 0.5 * spectrum.Tobs * coloured_psd(np.maximum(bins, spectrum.df))
    for name in ("A", "E"):
        np.testing.assert_allclose(covariance.pair(name, name), expected, rtol=1e-12)
    # The DC bin takes the first bin's value instead of psd(0).
    assert covariance.A[0] == covariance.A[1]
    np.testing.assert_array_equal(covariance.AE, 0.0)


def test_wdm_cells_take_half_the_psd_at_their_band_frequency(
    simulated_file, forge_data
):
    window = forge_data.load_window(simulated_file, factor=FACTOR, **wdm_options())
    cells = forge_data.analytic_covariance(window, coloured_psd).A
    band_spacing_hz = window.fs / (2 * NF)
    for band in range(1, NF):
        np.testing.assert_allclose(
            cells[band], 0.5 * coloured_psd(band * band_spacing_hz), rtol=1e-12
        )
    np.testing.assert_allclose(
        cells[0, 0::2], 0.5 * coloured_psd(band_spacing_hz), rtol=1e-12
    )
    np.testing.assert_allclose(
        cells[0, 1::2], 0.5 * coloured_psd(NF * band_spacing_hz), rtol=1e-12
    )


@pytest.mark.parametrize("domain", ["time", "frequency", "wdm"])
def test_white_psd_matches_the_white_covariance_convention(
    simulated_file, forge_data, domain
):
    options = (
        wdm_options() if domain == "wdm" else dict(domain=domain, duration_s=DURATION_S)
    )
    window = forge_data.load_window(simulated_file, factor=FACTOR, **options)
    sigma2 = 1.3e-36
    covariance = forge_data.analytic_covariance(
        window, forge_data.white_psd(2 * sigma2 / window.fs)
    )
    reference = white_covariance(window, sigma2)
    for name in window.channel_names:
        np.testing.assert_allclose(
            covariance.pair(name, name), reference.pair(name, name), rtol=1e-12
        )


def test_coloured_psd_is_refused_on_the_time_grid(simulated_file, forge_data):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S
    )
    with pytest.raises(ValueError, match="frequency or WDM grid"):
        forge_data.analytic_covariance(window, coloured_psd)
    # A tilt of one part in a million across the band is still coloured.
    with pytest.raises(ValueError, match="frequency or WDM grid"):
        forge_data.analytic_covariance(window, lambda f: 1e-36 * (1.0 + 1e-6 * f))


def test_per_channel_psd_mapping(simulated_file, forge_data):
    spectrum = forge_data.load_window(
        simulated_file,
        factor=FACTOR,
        duration_s=DURATION_S,
        channels=("A", "E"),
        domain="frequency",
    )
    covariance = forge_data.analytic_covariance(
        spectrum, {"A": forge_data.white_psd(2e-36), "E": forge_data.white_psd(4e-36)}
    )
    np.testing.assert_allclose(covariance.E, 2 * covariance.A, rtol=1e-12)
    with pytest.raises(ValueError, match=r"no entry for channels \['E'\]"):
        forge_data.analytic_covariance(spectrum, {"A": forge_data.white_psd(2e-36)})
