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

"""Analytical root finding with array inputs."""

from __future__ import annotations  # Type annotations for Python 3.7 and 3.8

from typing import Union

import numpy as np


def solve_p2_v(a: Union[float, np.ndarray], b: Union[float, np.ndarray],
               c: Union[float, np.ndarray]) \
        -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Roots of second order polynomial with array as coefficients.

    Solve the equation a*x**2 + b*x + c = 0 when coefficients are arrays. If
    arrays are used, they must have the same size. Coefficient a must not be
    zero.

    Parameters
    ----------
    a :
    b :
    c :

    Returns
    -------
    x1: first root (may be complex)
    x2: second root (may be complex)

    """
    delta = b**2 - 4. * a * c
    i2a = 0.5 / a
    sqd = np.sqrt(delta + 0J)
    x1 = (-b + sqd) * i2a
    x2 = (-b - sqd) * i2a

    if np.all(np.imag(x1) == 0.):
        x1 = np.real(x1)

    if np.all(np.imag(x2) == 0.):
        x2 = np.real(x2)

    return x1, x2


def _cardan_v(p_: Union[float, np.ndarray], q_: Union[float, np.ndarray]) \
        -> tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Roots of specific third order polynomial with array as coefficients.

    Solve the equation x**3 + p*x + q = 0 when coefficients are arrays. If
    arrays are used, they must have the same size.

    Parameters
    ----------
    p_ :
    q_ :

    Returns
    -------
    x1: first root (may be complex)
    x2: second root (may be complex)
    x3: third root (may be complex)

    """
    p = np.asarray(p_)
    q = np.asarray(q_)

    delta = 4. * p**3 + 27. * q**2
    sp = np.sqrt(delta / 27.)
    sm = np.sqrt(delta / 27. + 0J)

    u = np.where(delta >= 0.,
                 np.cbrt(0.5 * (-q + sp)),
                 np.power(0.5 * (-q + sm), 1 / 3))

    v = np.where(delta >= 0.,
                 np.cbrt(0.5 * (-q - sp)),
                 np.power(0.5 * (-q - sm), 1 / 3))

    j1 = 0.5 * (-1. + np.sqrt(3) * 1J)
    j2 = j1.conjugate()
    x1 = u + v
    x2 = j1 * u + j2 * v
    x3 = j2 * u + j1 * v

    return np.real(x1), x2, x3


def solve_p3_v(a: Union[float, np.ndarray], b: Union[float, np.ndarray],
               c: Union[float, np.ndarray], d: Union[float, np.ndarray]) \
        -> tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Roots of second order polynomial with array as coefficients.

    Solve the equation a*x**3 + b*x**2 + c*x + d = 0 when coefficients are
    arrays. If arrays are used, they must have the same size. Coefficient a must
    not be zero.

    Parameters
    ----------
    a :
    b :
    c :
    d :

    Returns
    -------
    x1: first root (may be complex)
    x2: second root (may be complex)
    x3: third root (may be complex)

    """
    p = - (b / a)**2 / 3. + c / a
    q = (b / (27. * a)) * (2. * (b / a)**2 - 9. * c / a) + d / a
    z1, z2, z3 = _cardan_v(p, q)

    s = b / (3. * a)
    x1 = z1 - s
    x2 = z2 - s
    x3 = z3 - s

    return x1, x2, x3
