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
from enchilada import decimate, load_mojito_l1, transform  # noqa: E402

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


def steep_psd(f):
    """A red PSD whose wall sits inside the test grid's first bands."""
    return 1e-36 * (1.0 + (0.1 / f) ** 2)


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


def load_grid(forge_data, path, domain):
    """The simulated window on `domain`, all three channels."""
    options = (
        wdm_options() if domain == "wdm" else dict(domain=domain, duration_s=DURATION_S)
    )
    return forge_data.load_window(path, factor=FACTOR, **options)


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
    # Exactly factor * Nf * Nt native samples went in: the same pipeline by hand.
    native = load_mojito_l1(simulated_file, duration_s=FACTOR * NF * NT * MOJITO_DT)
    assert native.num_time_samples == FACTOR * NF * NT
    expected = transform(
        decimate(native, FACTOR),
        "wdm",
        num_frequency_divisions=NF,
        num_time_divisions=NT,
    )
    for name in ("A", "E", "T"):
        np.testing.assert_array_equal(
            window.channel_data[name], expected.channel_data[name]
        )


def test_wdm_window_reads_only_the_requested_samples(tmp_path, forge_data):
    # A flagged sample before the window must not matter, so the native
    # sampling interval comes from the file's attributes, not a probe load.
    flagged = tmp_path / "flagged.h5"
    write_mojito_file(flagged, n=FILE_SAMPLES, flag_index=10)
    window = forge_data.load_window(
        flagged, factor=FACTOR, start_offset_s=100.0, **wdm_options()
    )
    assert window.num_time_samples == NF * NT
    assert window.start_time_gps == MOJITO_T0 + 100.0
    # A file shorter than a minute still serves a window that fits it.
    short = tmp_path / "short.h5"
    write_mojito_file(short, n=200)
    window = forge_data.load_window(
        short, factor=2, domain="wdm", num_frequency_divisions=8, num_time_divisions=10
    )
    assert window.wdm_grid.array_shape == (8, 10)


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
            "positive even integer",
        ),
        (
            dict(domain="wdm", num_frequency_divisions=7, num_time_divisions=NT),
            "positive even integer",
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


def test_load_window_validates_the_factor_before_reading(tmp_path, forge_data):
    missing = tmp_path / "missing.h5"
    with pytest.raises(ValueError, match="at least 2"):
        forge_data.load_window(missing, factor=1, duration_s=DURATION_S)
    for factor in (2.0, True):
        with pytest.raises(TypeError, match="integer"):
            forge_data.load_window(missing, factor=factor, duration_s=DURATION_S)


# ---- white_psd and analytic_covariance ---------------------------------------


def test_white_psd_is_flat_and_refuses_bad_levels(forge_data):
    psd = forge_data.white_psd(3.0)
    np.testing.assert_array_equal(psd(np.array([0.0, 1e-3, 0.5])), 3.0)
    assert psd(np.float64(0.1)).shape == ()
    for level in (0.0, -1.0, np.inf, np.nan, "abc", np.array([1.0, 2.0])):
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


def test_wdm_cells_average_the_psd_across_each_band(simulated_file, forge_data):
    window = load_grid(forge_data, simulated_file, "wdm")
    cells = forge_data.analytic_covariance(window, steep_psd).A
    band_spacing_hz = window.fs / (2 * NF)
    # Stationary noise: one value per band. Row 0 interleaves DC and Nyquist.
    for band in range(1, NF):
        assert np.ptp(cells[band]) == 0.0
    assert np.ptp(cells[0, 0::2]) == 0.0 and np.ptp(cells[0, 1::2]) == 0.0

    # Each band's value is an average of S / 2 over that band's support, so it
    # lies strictly between the PSD's extremes there and falls with the band.
    def bounds(low_hz, high_hz):
        grid = np.linspace(max(low_hz, window.df), high_hz, 1001)
        return 0.5 * steep_psd(grid).min(), 0.5 * steep_psd(grid).max()

    values = [cells[0, 0]] + [cells[band, 0] for band in range(1, NF)] + [cells[0, 1]]
    for band, value in enumerate(values):
        low, high = bounds((band - 1) * band_spacing_hz, (band + 1) * band_spacing_hz)
        assert low < value < high, (band, value, low, high)
    assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True))

    # A PSD linear in f averages to its band-centre value on interior bands.
    def linear_psd(f):
        return 1e-36 * (1.0 + 2.0 * f)

    linear = forge_data.analytic_covariance(window, linear_psd).A
    for band in range(1, NF):
        np.testing.assert_allclose(
            linear[band], 0.5 * linear_psd(band * band_spacing_hz), rtol=1e-9
        )
    # A PSD that vanishes at every band centre averages to half its peak over
    # a band (the Meyer window's partition of unity), so a band-centre
    # evaluation would give zero here.
    notched = forge_data.analytic_covariance(
        window, lambda f: 1e-36 * np.sin(np.pi * f / band_spacing_hz) ** 2
    ).A
    np.testing.assert_allclose(notched[1:], 0.25e-36, rtol=1e-12)
    # A PSD written for one-dimensional input works on the WDM path too.
    flat = forge_data.analytic_covariance(window, lambda f: np.full(len(f), 2e-36))
    np.testing.assert_array_equal(flat.A, 1e-36)


@pytest.mark.parametrize("domain", ["time", "frequency", "wdm"])
def test_white_psd_matches_the_white_covariance_convention(
    simulated_file, forge_data, domain
):
    window = load_grid(forge_data, simulated_file, domain)
    sigma2 = 1.3e-36
    covariance = forge_data.analytic_covariance(
        window, forge_data.white_psd(2 * sigma2 / window.fs)
    )
    reference = white_covariance(window, sigma2)
    for name in window.channel_names:
        np.testing.assert_allclose(
            covariance.pair(name, name), reference.pair(name, name), rtol=1e-12
        )


@pytest.mark.parametrize("domain", ["time", "frequency", "wdm"])
def test_a_constant_psd_broadcasts_like_white_psd(simulated_file, forge_data, domain):
    window = load_grid(forge_data, simulated_file, domain)
    constant = forge_data.analytic_covariance(window, lambda f: 2e-36)
    white = forge_data.analytic_covariance(window, forge_data.white_psd(2e-36))
    for name in window.channel_names:
        np.testing.assert_array_equal(constant.pair(name, name), white.pair(name, name))


def test_psd_values_must_be_finite_and_positive(simulated_file, forge_data):
    spectrum = load_grid(forge_data, simulated_file, "frequency")
    with pytest.raises(ValueError, match="not finite"):
        forge_data.analytic_covariance(spectrum, lambda f: np.nan)
    with pytest.raises(ValueError, match="non-positive"):
        forge_data.analytic_covariance(spectrum, lambda f: 0.0)
    with pytest.raises(ValueError, match="returned shape"):
        forge_data.analytic_covariance(spectrum, lambda f: np.ones(3))
    with pytest.raises(ValueError, match="negative"):
        forge_data.analytic_covariance(spectrum, lambda f: -1e-36)
    window = load_grid(forge_data, simulated_file, "wdm")
    with pytest.raises(ValueError, match="non-positive"):
        forge_data.analytic_covariance(window, lambda f: 0.0)


def test_coloured_psd_is_refused_on_the_time_grid(simulated_file, forge_data):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S
    )
    with pytest.raises(ValueError, match="frequency or WDM grid"):
        forge_data.analytic_covariance(window, coloured_psd)
    # A tilt of one part in a million across the band (f up to 0.5 Hz) is
    # still coloured.
    with pytest.raises(ValueError, match="frequency or WDM grid"):
        forge_data.analytic_covariance(window, lambda f: 1e-36 * (1.0 + 2e-6 * f))


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
