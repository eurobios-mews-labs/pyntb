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

"""Analytical root finding with array inputs (examples)."""

import matplotlib.pyplot as plt
import numpy as np

from pyntb.polynomial import solve_p2_v, _cardan_v, solve_p3_v


def example_solve_p2_v(n=10):
    """
    Simple example for solving several 2nd order polynomial roots at one time.
    """
    a = np.random.randn(n)
    b = np.random.randn(n)
    c = np.random.randn(n)

    def fun(x):
        """Function to zero."""
        return a * x**2 + b * x + c

    x1, x2 = solve_p2_v(a, b, c)

    c1 = fun(x1)
    c2 = fun(x2)

    plt.figure()
    plt.semilogy(np.abs(c1), 'o', label='first root')
    plt.semilogy(np.abs(c2), 'o', label='second root')
    plt.grid(True)
    plt.legend()
    plt.title('Error on root finding (solve_p2_v)')

    return


def example_solve__cardan_v(n=10):
    """...
    """

    p = np.random.randn(n)
    q = np.random.randn(n)

    def fun(x):
        """Function to zero."""
        return x**3 + p * x + q

    x1, x2, x3 = _cardan_v(p, q)

    plt.figure()
    plt.semilogy(np.abs(fun(x1)), 'o', label='first root')
    plt.semilogy(np.abs(fun(x2)), 'o', label='second root')
    plt.semilogy(np.abs(fun(x3)), 'o', label='third root')
    plt.grid(True)
    plt.legend()
    plt.title('Error on root finding (_cardan_v)')

    return


def example_solve_p3_v(n=10):
    """
    Simple example for solving several 3rd order polynomial roots at one time.
    """
    a = np.random.randn(n)
    b = np.random.randn(n)
    c = np.random.randn(n)
    d = np.random.randn(n)

    def fun(x):
        """Function to zero."""
        return a * x**3 + b * x**2 + c * x + d

    x1, x2, x3 = solve_p3_v(a, b, c, d)

    plt.figure()
    plt.semilogy(np.abs(fun(x1)), 'o', label='first root')
    plt.semilogy(np.abs(fun(x2)), 'o', label='second root')
    plt.semilogy(np.abs(fun(x3)), 'o', label='third root')
    plt.grid(True)
    plt.legend()
    plt.title('Error on root finding (solve_p3_v)')

    return


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    example_solve_p2_v()
    example_solve__cardan_v()
    example_solve_p3_v()
