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

"""Misc. numeric functions (test)."""

import numpy as np

from pyntb.optimize import bisect_v, fixed_point, qnewt2d_v


def test_bisect():
    size = 99
    tol = 1.0E-09

    c = np.linspace(1, size, size) + np.random.randn(size)
    c = np.abs(c)

    def fun(x):
        return x**2 - c

    r0 = np.sqrt(c)

    x0, err = bisect_v(fun, 0., np.sqrt(np.max(c)) * 1.1, (size,), print_err=True,
                       tol=tol, maxiter=99)

    assert np.max(np.abs(x0 - r0) <= np.minimum(tol, err))


def test_fixed_point():
    size = 99
    tol = 1.0E-09

    c = 1 + np.sqrt(np.linspace(1, 10, size))

    def fun(x):
        return np.sin(c * x)

    x0 = fixed_point(fun, 0.5, xtol=tol, maxiter=999)

    assert np.max(np.abs(x0 - fun(x0))) < tol


def test_qnewt2d():
    size = 99
    tol = 1.0E-12

    a = np.abs(1 + np.random.randn(size))
    b = np.abs(1 + np.random.randn(size))

    def f1(x, y):
        return y - a * x**2

    def f2(x, y):
        return y - b * x**3

    xg = np.ones((size,))
    yg = np.ones((size,))
    x, y, count, err = qnewt2d_v(f1, f2, xg, yg, rtol=tol, maxiter=999, dx=1.0E-09, dy=1.0E-09)

    assert np.logical_and(np.max(np.abs(f1(x, y))) < tol,
                          np.max(np.abs(f2(x, y))) < tol)
