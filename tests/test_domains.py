"""WDM grids describe domania's folded coefficient layout without a transform."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_wdm_grid_is_keyword_only_and_immutable():
    from enchilada.domains import WDMGrid

    with pytest.raises(TypeError):
        WDMGrid(4, 8)
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    with pytest.raises(FrozenInstanceError):
        grid.num_frequency_divisions = 8
    assert grid.array_shape == (4, 8)
    assert grid.num_time_samples == 32


@pytest.mark.parametrize("name", ["num_frequency_divisions", "num_time_divisions"])
@pytest.mark.parametrize("value", [0, -2, 3, True, False, 4.0, np.float64(4.0)])
def test_wdm_grid_rejects_invalid_divisions(name, value):
    from enchilada.domains import WDMGrid

    arguments = {"num_frequency_divisions": 4, "num_time_divisions": 8, name: value}
    with pytest.raises(ValueError, match=name):
        WDMGrid(**arguments)


def test_numpy_integer_grid_counts_are_normalised_and_every_cell_is_data():
    from enchilada.domains import Grid, WDMGrid

    grid = WDMGrid(num_frequency_divisions=np.int64(4), num_time_divisions=np.int32(8))
    assert type(grid.num_frequency_divisions) is int
    assert type(grid.num_time_divisions) is int
    assert grid.array_shape == (4, 8)
    assert int(np.prod(grid.array_shape)) == grid.num_time_samples
    assert not hasattr(grid, "active_mask")
    assert not hasattr(Grid("wdm", 32, 1.0, grid), "active_mask")


@pytest.mark.parametrize(
    "arguments",
    [
        {"num_frequency_divisions": 4},
        {"num_time_divisions": 8},
        {"num_frequency_divisions": 4, "num_time_divisions": 8},
    ],
)
def test_resolve_wdm_grid_derives_only_the_missing_division(arguments):
    from enchilada.domains import WDMGrid, resolve_wdm_grid

    assert resolve_wdm_grid(32, **arguments) == WDMGrid(
        num_frequency_divisions=4, num_time_divisions=8
    )


@pytest.mark.parametrize(
    "num_time_samples,arguments",
    [
        (32, {}),
        (32, {"num_frequency_divisions": 4, "num_time_divisions": 4}),
        (32, {"num_frequency_divisions": 6}),
        (12, {"num_frequency_divisions": 4}),
        (32, {"num_frequency_divisions": True}),
        (32, {"num_time_divisions": 4.0}),
        (0, {"num_frequency_divisions": 4}),
        (-32, {"num_frequency_divisions": 4}),
        (True, {"num_frequency_divisions": 4}),
        (32.0, {"num_frequency_divisions": 4}),
    ],
)
def test_resolve_wdm_grid_rejects_ambiguous_or_incompatible_counts(
    num_time_samples, arguments
):
    from enchilada.domains import resolve_wdm_grid

    with pytest.raises(ValueError):
        resolve_wdm_grid(num_time_samples, **arguments)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wdm_coefficients_keep_values_and_input_buffers(dtype):
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    coefficients = np.arange(grid.num_time_samples, dtype=dtype).reshape(
        grid.array_shape
    )
    original = coefficients.copy()
    check_wdm_coefficients({"A": coefficients}, grid, "data")
    np.testing.assert_array_equal(coefficients, original)


def test_row_zero_odd_slots_carry_nyquist_data_and_are_accepted():
    """In the folded layout row 0 interleaves DC (even slots) with Nyquist (odd
    slots); there is no structural zero to enforce."""
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    coefficients = np.zeros(grid.array_shape)
    coefficients[0, 1::2] = np.nextafter(0.0, 1.0)
    check_wdm_coefficients({"A": coefficients}, grid, "data")


@pytest.mark.parametrize(
    "coefficients,error,match",
    [
        (np.zeros((5, 8)), ValueError, "shape"),
        (np.zeros((4, 7)), ValueError, "shape"),
        (np.zeros((2, 16)), ValueError, "shape"),  # well formed, but not this grid
        (np.zeros(32), ValueError, "shape"),
        (np.zeros((4, 8), complex), TypeError, "real"),
        (np.zeros((4, 8), int), TypeError, "floating"),
        ([[0.0] * 8] * 4, TypeError, "numpy array"),
    ],
)
def test_wdm_coefficients_reject_data_off_the_grid(coefficients, error, match):
    from enchilada.domains import WDMGrid, check_wdm_coefficients

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=8)
    with pytest.raises(error, match=match):
        check_wdm_coefficients({"A": coefficients}, grid, "data")
