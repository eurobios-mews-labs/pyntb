# Copyright 2023 Eurobios Mews Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Analytical root finding with array inputs (test)."""

import numpy as np
import pytest

from pyntb.polynomial import solve_p2_v, _cardan_v, solve_p3_v

_nprs = 3141592654

def _maxabs(x):
    return np.max(np.abs(x))


@pytest.fixture
def sizetol():
    return 99, 1.0E-10


def    test_solve_p2_distinct_real_roots(sizetol):
    np.random.seed(_nprs)
    size, tol = sizetol

    r1 = np.random.randn(size)
    r2 = np.random.randn(size)

    def fun(x):
        return (x - r1) * (x - r2)

    x1, x2 = solve_p2_v(1., -(r1 + r2), r1 * r2)

    assert np.logical_and.reduce((_maxabs(fun(x1)) < tol,
                                  _maxabs(fun(x2)) < tol,
                                  np.max(np.minimum(np.abs(r1 - x1),
                                                    np.abs(r1 - x2))) < tol,
                                  np.max(np.minimum(np.abs(r2 - x1),
                                                    np.abs(r2 - x2))) < tol,
                                  ))


def test_solve_p2_double_real_root(sizetol):
    np.random.seed(_nprs)
    size, tol = sizetol

    rd = np.random.randn(size)

    def fun(x):
        return (x - rd)**2

    x1, x2 = solve_p2_v(1., -2 * rd, rd**2)

    assert np.logical_and.reduce((_maxabs(fun(x1)) < tol,
                                  _maxabs(fun(x2)) < tol,
                                  _maxabs(x1 - rd) < tol,
                                  _maxabs(x2 - rd) < tol,
                                  ))


def test_solve_p2_complex_roots(sizetol):
    np.random.seed(_nprs)
    size, tol = sizetol

    r1 = np.random.randn(size) + 1j * np.random.randn(size)
    r2 = np.random.randn(size) + 1j * np.random.randn(size)

    def fun(x):
        return (x - r1) * (x - r2)

    x1, x2 = solve_p2_v(1., -(r1 + r2), r1 * r2)

    assert np.logical_and.reduce((_maxabs(fun(x1)) < tol,
                                  _maxabs(fun(x2)) < tol,
                                  np.max(np.minimum(np.abs(r1 - x1),
                                                    np.abs(r1 - x2))) < tol
                                  ))


def test_solve_cardan(sizetol):
    np.random.seed(_nprs)
    size, tol = sizetol

    p = np.random.randn(size)
    q = np.random.randn(size)

    def fun(x):
        return x**3 + p * x + q

    x1, x2, x3 = _cardan_v(p, q)

    assert np.logical_and.reduce((_maxabs(fun(x1)) < tol,
                                  _maxabs(fun(x2)) < tol,
                                  _maxabs(fun(x3)) < tol,
                                  ))


def test_solve_p3(sizetol):
    np.random.seed(_nprs)
    size, tol = sizetol

    a = np.random.randn(size)
    b = np.random.randn(size)
    c = np.random.randn(size)
    d = np.random.randn(size)

    def fun(x):
        return a * x**3 + b * x**2 + c * x + d

    x1, x2, x3 = solve_p3_v(a, b, c, d)

    assert np.logical_and.reduce((_maxabs(fun(x1)) < tol,
                                  _maxabs(fun(x2)) < tol,
                                  _maxabs(fun(x3)) < tol,
                                  ))
