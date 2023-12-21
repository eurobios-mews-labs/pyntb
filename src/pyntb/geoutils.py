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

"""Geodata utility functions."""

import numpy as np


def haversine(lat1, lon1, lat2, lon2, r=6.371E+06):
    """Haversine distance.

    Estimate distance between to gps points using the haversine formula (see
    https://en.wikipedia.org/wiki/Haversine_formula). Note that since the Earth
    is not a perfect sphere, this formula is not exact, and tends to be more
    wrong near the poles. The default value is the Earth's mean radius given in
    meters, you can change it to any value to fit your problem.

    If inputs are numpy arrays, they must all have the same shape.

    Parameters
    ----------
    lat1 : float or numpy array
        Latitude of first point in degrees.
    lon1 : float or numpy array
        Longitude of first point in degrees.
    lat2 : float or numpy array
        Latitude of second point in degrees.
    lon2 : float or numpy array
        Longitude of second point in degrees.
    r : float
        Reference radius, ie the Earth's mean radius in meters in our case. The
        default is 6.371E+06.

    Returns
    -------
    Distance between input points.

    """
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(0.5 * dp)**2 + np.cos(p1) * np.cos(p2) * np.sin(0.5 * dl)**2
    c = np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 2 * r * c
