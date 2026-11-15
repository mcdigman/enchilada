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
