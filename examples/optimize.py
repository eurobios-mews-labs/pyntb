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

"""Misc. numeric functions (examples)."""

import matplotlib.pyplot as plt
import numpy as np

from pyntb.optimize import bisect_v, fixed_point, qnewt2d_v


def example_bisect_v(n=10):
    """Simple example for bisection on arrays.

    Here we have the function fun(x) = x**2 - c where c is an array of
    shape (n,). The bisection is run then we plot both results and errors
    against analytics results.
    """
    c = np.linspace(1, n, n)

    def fun(x):
        return x**2 - c

    x0, err = bisect_v(fun, 0., np.sqrt(n) + 1., (n,), print_err=True)

    plt.figure()
    plt.title('Bisection results')
    plt.plot(c, x0, 'o', label='bisect_1d results')
    plt.plot(c, np.sqrt(c), label='analytic')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.title('Bisection error')
    plt.plot(c, err, 'o', label='from bisect_1d')
    plt.plot(c, np.abs(x0 - np.sqrt(c)), label='from analytic')
    plt.grid(True)
    plt.legend()

    return


def example_fixed_point(n=8, put_nan=True):
    """Simple example for fixed_point.

    Here we try to find the fixed point of sin(x*c) where c is an array. We can
    choose to put a nan in the array, and in that case the call to the
    fixed_point from scipy.optimize will throw an error; the modified
    fixed_point provided by pyntb correct this.
    """

    c = 1 + np.sqrt(np.linspace(1, n, n))
    if put_nan:
        c[n // 2] = np.nan
    x = np.linspace(0., np.pi, 101)

    def fun(x):
        return np.sin(c * x)

    from scipy.optimize import fixed_point as fixed_point_scipy

    try:
        x0 = fixed_point_scipy(fun, 0.5)
    except RuntimeError:
        print("Error in scipy (RuntimeError due to nans in input array), "
              "using fixed_point from pyntb")
        x0 = fixed_point(fun, 0.5)

    plt.figure()
    plt.plot(x, x, ls='--', c='gray')
    for i in range(n):
        plt.plot(x, np.sin(c[i] * x))
    plt.plot(x0, x0, ls=None, marker='o', c='gray')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.title('Fixed point results')

    return


def example_qnewt2d_v(n=10):
    """Simple example for two-dimensional quasi-Newton with arrays.

    Here we're solving the system [y -a*x**2, y-b*sqrt(x)] = [0, 0] for different
    values of a and b.

    """

    a = np.abs(np.random.randn(n))
    b = np.abs(np.random.randn(n))

    def f1(x, y):
        return y - a * x**2

    def f2(x, y):
        return y - b * np.sqrt(x)

    xg = np.ones((n,))
    yg = np.ones((n,))
    x, y, count, err = qnewt2d_v(f1, f2, xg, yg)

    plt.figure()
    plt.semilogy(err, ls='None', marker='o', label='qnewt2d_v err')
    plt.semilogy(np.abs(f1(x, y)), ls='None', marker='.', label='real err on f1')
    plt.semilogy(np.abs(f2(x, y)), ls='None', marker='.', label='real err on f2')
    plt.grid(True)
    plt.legend()
    plt.title('Errors on qnewt2d_v')

    return


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    example_bisect_v()
    example_fixed_point()
    example_qnewt2d_v()
