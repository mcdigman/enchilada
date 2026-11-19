"""The two-noise-block example recovers an excess variance above a known floor."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def noise_floor():
    spec = importlib.util.spec_from_file_location(
        "noise_floor", EXAMPLES / "noise_floor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_excess_variance_is_recovered_above_the_floor(noise_floor):
    result = noise_floor.run_example(n_cycles=200, burn_in=50, seed=0)
    mean, std = result["posterior_mean"], result["posterior_std"]
    truth = noise_floor.TRUTH["excess"]
    assert std > 0.0
    assert mean == pytest.approx(truth, abs=max(5 * std, 0.05)), (
        f"excess: posterior {mean} +/- {std} vs truth {truth}"
    )
    wheel = result["wheel"]
    assert list(wheel.ledger) == ["floor", "excess"]
    np.testing.assert_array_equal(
        wheel.ledger["floor"].covariance.A, noise_floor.TRUTH["floor"]
    )
    np.testing.assert_allclose(
        wheel.covariance.A,
        noise_floor.TRUTH["floor"] + wheel.ledger["excess"].covariance.A,
    )
    assert wheel.ledger["excess"].template is None
    assert wheel.ledger["excess"].parameters["variance"] == pytest.approx(
        float(wheel.ledger["excess"].covariance.A[0])
    )


def test_the_excess_block_reads_the_floor_it_is_handed(noise_floor):
    from enchilada import BlockState, Covariance

    observed = noise_floor.make_observed(seed=1)
    floor = Covariance.like(
        observed, {ch: np.full(observed.N, 0.25) for ch in observed.channel_names}
    )
    block = noise_floor.ExcessNoiseBlock("excess")
    rng = np.random.default_rng(0)
    start = block.initial_sample(observed, floor, rng=rng)
    assert isinstance(start, BlockState)
    assert start.template is None
    np.testing.assert_array_equal(start.covariance.A, start.parameters["variance"])
    later = block.sample(observed, floor, start, rng=rng)
    np.testing.assert_array_equal(later.covariance.A, later.parameters["variance"])
    with pytest.raises(ValueError, match="floor"):
        block.initial_sample(observed, None, rng=rng)
    with pytest.raises(ValueError, match="time-domain"):
        block.sample(observed.to_frequency(), None, start, rng=rng)
