"""The persistent sum trees behind Wheel's residuals and its noise sum."""

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import Covariance, L1Data
from enchilada._sum_tree import NoiseSum, SignalSum


def data():
    return L1Data(
        {"A": np.zeros(8), "E": np.zeros(8)},
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="strain",
    )


class SignalLeaves:
    """Templates: dicts of channel arrays; the sum is read on channel A."""

    tree = SignalSum
    add = staticmethod(SignalSum.with_signal)

    @staticmethod
    def leaf(level):
        return {"A": np.full(8, level), "E": np.full(8, 2 * level)}

    @staticmethod
    def value(total):
        return None if total is None else float(total["A"][0])


class NoiseLeaves:
    """Covariances: white noise of `level` on one grid; the sum is read on A."""

    tree = NoiseSum
    add = staticmethod(NoiseSum.with_covariance)

    @staticmethod
    def leaf(level):
        return white_covariance(data(), level)

    @staticmethod
    def value(total):
        return None if total is None else float(total.A[0])


@pytest.fixture(params=[SignalLeaves, NoiseLeaves], ids=["signal", "noise"])
def leaves(request):
    return request.param


def test_the_empty_tree_sums_to_none(leaves):
    tree = leaves.tree()
    assert tree.total is None
    assert tree.excluding(0) is None
    assert tree.excluding(7) is None


def test_one_leaf_is_its_own_total_and_excluding_it_leaves_nothing(leaves):
    tree = leaves.add(leaves.tree(), 0, leaves.leaf(1.0))
    assert leaves.value(tree.total) == 1.0
    assert tree.excluding(0) is None
    # A slot beyond the capacity was never filled, so excluding it drops nothing.
    assert leaves.value(tree.excluding(1)) == 1.0


def test_three_leaves_sum_and_each_exclusion_drops_one(leaves):
    tree = leaves.tree()
    for index, level in enumerate((1.0, 2.0, 4.0)):
        tree = leaves.add(tree, index, leaves.leaf(level))
    assert leaves.value(tree.total) == 7.0
    assert leaves.value(tree.excluding(0)) == 6.0
    assert leaves.value(tree.excluding(1)) == 5.0
    assert leaves.value(tree.excluding(2)) == 3.0
    assert leaves.value(tree.excluding(3)) == 7.0  # capacity 4, slot 3 empty


def test_replacing_a_slot_and_withdrawing_it(leaves):
    tree = leaves.tree()
    for index, level in enumerate((1.0, 2.0, 4.0)):
        tree = leaves.add(tree, index, leaves.leaf(level))
    replaced = leaves.add(tree, 1, leaves.leaf(10.0))
    assert leaves.value(replaced.total) == 15.0
    assert leaves.value(replaced.excluding(1)) == 5.0
    withdrawn = leaves.add(replaced, 1, None)
    assert leaves.value(withdrawn.total) == 5.0
    assert leaves.value(withdrawn.excluding(1)) == 5.0
    assert leaves.value(withdrawn.excluding(0)) == 4.0


def test_a_discarded_candidate_leaves_the_accepted_tree_unchanged(leaves):
    accepted = leaves.add(leaves.tree(), 0, leaves.leaf(1.0))
    accepted = leaves.add(accepted, 1, leaves.leaf(2.0))
    candidate = leaves.add(accepted, 1, leaves.leaf(100.0))
    candidate = leaves.add(candidate, 5, leaves.leaf(1000.0))  # grows the candidate
    assert leaves.value(candidate.total) == 1101.0
    assert leaves.value(candidate.excluding(5)) == 101.0
    assert leaves.value(accepted.total) == 3.0
    assert leaves.value(accepted.excluding(0)) == 2.0
    assert leaves.value(accepted.excluding(1)) == 1.0
    assert leaves.value(accepted.excluding(5)) == 3.0


def test_sparse_slots_skip_the_empty_leaves(leaves):
    tree = leaves.add(leaves.tree(), 6, leaves.leaf(3.0))
    assert leaves.value(tree.total) == 3.0
    assert tree.excluding(6) is None
    assert leaves.value(tree.excluding(0)) == 3.0
    assert leaves.value(tree.excluding(7)) == 3.0


def test_signal_sum_promotes_float32_leaves_to_float64():
    tree = SignalSum().with_signal(0, {"A": np.full(4, 1e8, np.float32)})
    tree = tree.with_signal(1, {"A": np.full(4, 1.0, np.float32)})
    tree = tree.with_signal(2, {"A": np.full(4, -1e8, np.float32)})
    assert tree.total["A"].dtype == np.float64
    np.testing.assert_array_equal(tree.total["A"], 1.0)


def test_noise_sum_totals_are_validated_read_only_covariances():
    tree = NoiseSum().with_covariance(0, white_covariance(data(), 1.0))
    tree = tree.with_covariance(1, white_covariance(data(), 2.0))
    assert type(tree.total) is Covariance
    assert not tree.total.blocks.flags.writeable
    np.testing.assert_array_equal(tree.total.E, 3.0)
    np.testing.assert_array_equal(tree.excluding(0).E, 2.0)
    assert tree.total.grid == data().grid
