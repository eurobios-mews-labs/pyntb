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

"""Geodata utility functions (test)."""

import numpy as np
from sklearn.metrics.pairwise import haversine_distances

from pyntb.geoutils import haversine

_nprs = 3141592654


def test_haversine():
    np.random.seed(_nprs)
    size = 99
    tol = 1.0E-12

    lat1 = -90 + 180 * np.random.rand(size)
    lat2 = -90 + 180 * np.random.rand(size)
    lon1 = -180 + 360 * np.random.rand(size)
    lon2 = -180 + 360 * np.random.rand(size)

    x = np.deg2rad(np.column_stack((lat1, lon1)))
    y = np.deg2rad(np.column_stack((lat2, lon2)))

    da = np.diag(haversine_distances(x, y))
    db = haversine(lat1, lon1, lat2, lon2, r=1)

    assert np.max(np.abs(db - da) <= tol)
