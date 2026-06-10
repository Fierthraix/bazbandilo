#!/usr/bin/env python3


def logistic(x0: float, μ: float = 4) -> float:
    assert -1 <= x0 <= 1
    return (μ / 2) * (1 - x0**2) - 1


def cubic(x0: float, μ: float = 3) -> float:
    assert -1 <= x0 <= 1
    return μ * x0**3 - (1 - μ) * x0


def tent(x0: float, μ: float = 1) -> float:
    assert -1 <= x0 <= 1
    return 2 * μ * (1 - abs(x0)) - 1


def bernoulli(x0: float, μ: float = 2) -> float:
    assert -1 <= x0 <= 1
    if x0 <= 0:
        return μ * x0 + 1
    return μ * x0 - 1
