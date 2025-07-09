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

from pyntb.geoutils import haversine


def example_haversine():
    """Simple example for haversine distance.

    Here we have the function fun(x) = x**2 - c where c is an array of
    shape (n,). The bisection is run then we plot both results and errors
    against analytics results.
    """

    paris = (48.866667, 2.333333)
    brest = (48.3905283, -4.4860088)

    distance = haversine(paris[0], paris[1], brest[0], brest[1])
    distance /= 1000.0

    print(f"The distance between Paris and Brest is {distance:.3f} km.")

    return


if __name__ == "__main__":
    example_haversine()
