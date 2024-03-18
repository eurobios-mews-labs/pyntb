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

"""Misc. numeric functions."""

from __future__ import annotations  # Type annotations for Python 3.7 and 3.8

import numpy as np
from scipy._lib._util import _asarray_validated
from scipy._lib._util import _lazywhere


def bisect_v(fun: callable, a: float, b: float, shape: tuple[int, ...], tol=1.0E-06,
             maxiter=128, print_err=False) -> tuple[np.ndarray, np.ndarray]:
    """Bisection method to find a zero of a continuous function [a, b] -> R,
    such that f(a) < 0 < f(b).

    The method is vectorized, in the sense that it can in a single call find
    the zeros of several independent real-valued functions with the same input
    range [a, b].
    For this purpose, the `fun` argument should be a Python function taking as
    input a Numpy array of values in [a, b] and returning a Numpy array
    containing the evaluation of each function for the corresponding input.
    This is most efficient if the outputs values of all these functions are
    computed using vectorized Numpy operations as in the example below.

    Parameters
    ----------
    fun: Python function taking a Numpy array and returning a Numpy array of the same shape.
    a : lower bound of [a, b] interval.
    b : upper bound of [a, b] interval.
    shape : shape of the inputs and ouputs of `fun` and thus of the ouputs of `bisect_v`.
    tol : absolute tolerance.
    maxiter : maximum number of iterations.
    print_err : print or not max error and number of iterations at the end.

    Returns
    -------
    x: Numpy array of the same shape as the inputs and outputs of `fun`
        The zero of each function.
    err: Numpy array of the same shape as the inputs and outputs of `fun`
        The convergence error.

    Example
    -------
    >>> c = np.array([1.0, 4.0, 9.0, 16.0])
    >>> def f(x):
    ...     return x**2 - c
    >>> x0, err = bisect_v(f, a=0.0, b=10.0, shape=(4,), tol=1e-10)
    >>> x0
    array([1., 2., 3., 4.])

    """
    a_ = a * np.ones(shape)
    b_ = b * np.ones(shape)

    err = np.abs(b - a)
    count = 1
    while np.nanmax(err) > tol and count <= maxiter:
        x = 0.5 * (a_ + b_)
        y = fun(x)
        i = y < 0
        a_[i] = x[i]
        b_[~i] = x[~i]
        err = np.abs(b_ - a_)
        count += 1
    x = 0.5 * (a_ + b_)
    x[np.isnan(fun(x))] = np.nan
    if print_err:
        print(f"Bisection max err (abs) : {np.max(err):.2E}; count={count}")
    return x, err


def _del2(p0, p1, d):
    return p0 - np.square(p1 - p0) / d

def _relerr(actual, desired):
    return (actual - desired) / desired

def _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel):
    """Almost copied from scipy.optimize._minpack_py.py.

    Changed
        if np.all(np.abs(relerr) < xtol)
    into
        if np.nanmax(np.abs(relerr)) < xtol

    Misc correction to match pylint.

    """
    p0 = x0
    for _ in range(maxiter):
        p1 = func(p0, *args)
        if use_accel:
            p2 = func(p1, *args)
            d = p2 - 2.0 * p1 + p0
            p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
        else:
            p = p1
        relerr = _lazywhere(p0 != 0, (p, p0), f=_relerr, fillvalue=p)
        if np.nanmax(np.abs(relerr)) < xtol:
            return p
        p0 = p
    msg = f"Failed to converge after {maxiter} iterations, value is {p}"
    raise RuntimeError(msg)


def fixed_point(func, x0, args=(), xtol=1e-8, maxiter=500, method='del2'):
    """
    Find a fixed point of the function (copied from
    scipy.optimize._minpack_py.py in order to handle nans).

    Given a function of one or more variables and a starting point, find a
    fixed point of the function: i.e., where ``func(x0) == x0``.

    Parameters
    ----------
    func : callable
        Function to evaluate.
    x0 : array_like
        Fixed point of function.
    args : tuple, optional
        Extra arguments to `func`.
    xtol : float, optional
        Convergence tolerance, defaults to 1e-08.
    maxiter : int, optional
        Maximum number of iterations, defaults to 500.
    method : {"del2", "iteration"}, optional
        Method of finding the fixed-point, defaults to "del2",
        which uses Steffensen's Method with Aitken's ``Del^2``
        convergence acceleration [1]_. The "iteration" method simply iterates
        the function until convergence is detected, without attempting to
        accelerate the convergence.

    """
    use_accel = {'del2': True, 'iteration': False}[method]
    x0 = _asarray_validated(x0, as_inexact=True)
    return _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel)


def qnewt2d_v(f1: callable, f2: callable, x0: np.ndarray, y0: np.ndarray,
              rtol=1.0E-12, maxiter=64, dx=1.0E-03, dy=1.0E-03) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-dimensional quasi-Newton with arrays.

    Apply a 2D quasi newton on a large number of case at the same time, ie solve
    the system [f1(x, y), f2(x, y)] = [0., 0.] in n cases.

    Derivatives are estimated with a second-order centered estimation (ie f1 and
    f2 are evaluated four times at each iteration).

    All return values are arrays of the same size as inputs x0 and y0.

    Parameters
    ----------
    f1 : first component of a 2d function of two variables
    f2 : second component of a 2d function of two variables
    x0 : first component of the initial guess
    y0 : second component of the initial guess
    rtol : relative tolerance
    maxiter : max number of iteration
    dx : delta for evaluating derivative regarding first component
    dy : delta for evaluating derivative regarding second component

    Returns
    -------
    x: first component of solution
    y: second component of solution
    count: number of iterations when exiting the function
    err: relative error when exiting the function

    """
    err = 1.
    count = 0
    x = x0.copy()
    y = y0.copy()
    while err > rtol and count < maxiter:
        F1 = f1(x, y)
        F2 = f2(x, y)
        Ja = 0.5 * (f1(x + dx, y) - f1(x - dx, y)) / dx
        Jb = 0.5 * (f1(x, y + dy) - f1(x, y - dy)) / dy
        Jc = 0.5 * (f2(x + dx, y) - f2(x - dx, y)) / dx
        Jd = 0.5 * (f2(x, y + dy) - f2(x, y - dy)) / dy
        di = 1. / (Ja * Jd - Jb * Jc)
        ex = di * (Jd * F1 - Jb * F2)
        ey = di * (Ja * F2 - Jc * F1)
        x -= ex
        y -= ey
        err = max(np.nanmax(np.abs(ex / x)), np.nanmax(np.abs(ey / y)))
        count += 1
    return x, y, count, np.maximum(np.abs(ex / x), np.abs(ey / y))
