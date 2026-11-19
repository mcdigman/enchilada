"""Persistent pairwise sums of internally owned, validated block publications.

Replacing a leaf copies only its path to the root. An uncommitted candidate can
therefore be discarded without changing accepted state. Leaves are never
mutated here, and an absent publication needs no zero element. Wheel keeps both
trees private: `SignalSum` adds templates and `NoiseSum` adds covariances.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from enchilada.covariance import Covariance

Signal = dict[str, np.ndarray] | None
Noise = Covariance | None


def _combine_signals(left: Signal, right: Signal) -> Signal:
    if left is None:
        return right
    if right is None:
        return left
    # Do not round a subtree of float32 signals before it meets a float64
    # observation or another wider signal elsewhere in the tree.
    with np.errstate(over="ignore", invalid="ignore"):
        return {
            name: np.add(
                values,
                right[name],
                dtype=np.result_type(values, right[name], np.float64),
            )
            for name, values in left.items()
        }


def _combine_covariances(left: Noise, right: Noise) -> Noise:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


@dataclass(frozen=True, eq=False)
class _Node[T]:
    total: T
    left: "_Node[T] | None"
    right: "_Node[T] | None"


@dataclass(frozen=True, eq=False)
class _SumTree[T]:
    """A persistent aggregate indexed by nonnegative block registration positions.

    `combine(left, right)` adds two partial sums and `empty` is the sum of
    nothing. Both trees use None for an absent leaf and let `combine` skip it.
    """

    combine: Callable[[T, T], T]
    empty: T
    _root: _Node[T] | None = None
    _capacity: int = 1

    def _total(self, node: _Node[T] | None) -> T:
        return self.empty if node is None else node.total

    @property
    def total(self) -> T:
        return self._total(self._root)

    def with_leaf(self, index: int, leaf: T) -> "_SumTree[T]":
        root, capacity = self._root, self._capacity
        while index >= capacity:
            root = _Node(self._total(root), root, None)
            capacity *= 2
        return replace(
            self, _root=self._replace(root, index, capacity, leaf), _capacity=capacity
        )

    def _replace(
        self, node: _Node[T] | None, index: int, capacity: int, leaf: T
    ) -> _Node[T]:
        if capacity == 1:
            return _Node(leaf, None, None)
        midpoint = capacity // 2
        left = None if node is None else node.left
        right = None if node is None else node.right
        if index < midpoint:
            left = self._replace(left, index, midpoint, leaf)
        else:
            right = self._replace(right, index - midpoint, midpoint, leaf)
        return _Node(self.combine(self._total(left), self._total(right)), left, right)

    def excluding(self, index: int) -> T:
        if index >= self._capacity:
            return self.total  # that slot was never filled
        return self._excluding(self._root, index, self._capacity)

    def _excluding(self, node: _Node[T] | None, index: int, capacity: int) -> T:
        if node is None or capacity == 1:
            return self.empty
        midpoint = capacity // 2
        if index < midpoint:
            return self.combine(
                self._excluding(node.left, index, midpoint), self._total(node.right)
            )
        return self.combine(
            self._total(node.left),
            self._excluding(node.right, index - midpoint, midpoint),
        )


@dataclass(frozen=True, eq=False)
class SignalSum:
    """A private aggregate of block templates by registration position."""

    _tree: _SumTree[Signal] = field(
        default_factory=lambda: _SumTree(_combine_signals, None)
    )

    @property
    def total(self) -> Signal:
        return self._tree.total

    def with_signal(self, index: int, signal: Signal) -> "SignalSum":
        return SignalSum(self._tree.with_leaf(index, signal))

    def excluding(self, index: int) -> Signal:
        return self._tree.excluding(index)


@dataclass(frozen=True, eq=False)
class NoiseSum:
    """A private aggregate of noise blocks' covariances by registration position.

    Every leaf is a validated Covariance on one grid with one channel order;
    Wheel guarantees that before storing one, so `Covariance.__add__` never
    refuses a sum here.
    """

    _tree: _SumTree[Noise] = field(
        default_factory=lambda: _SumTree(_combine_covariances, None)
    )

    @property
    def total(self) -> Noise:
        return self._tree.total

    def with_covariance(self, index: int, covariance: Noise) -> "NoiseSum":
        return NoiseSum(self._tree.with_leaf(index, covariance))

    def excluding(self, index: int) -> Noise:
        return self._tree.excluding(index)
