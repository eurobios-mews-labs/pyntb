# Eurobios-Mews-Labs Toolbox

A toolbox to factorize some code utilities across various projects.

## Install

The package requires Python 3.7 or above and is available on PyPI:
```
python3 -m pip install pyntb
```

The optional dependencies used in the examples in the Github repository can be installed at the same time by typing instead:
```
python3 -m pip install pyntb[examples]
```

## Content

- `geoutils`
  - implementation of haversine distance which works on numpy arrays with an option to change the sphere radius
- `optimize` 
  - a bisection algorithm that work on numpy array inputs
  - a copy of `scipy.optimize.fixed_point` that deals with nan values
  - a 2D, quasi-newton algorithm that works with numpy array inputs
- `polynomial`
  - find roots of 2nd and 3rd order polymonials with numpy array inputs 

## Examples

See `examples` directory for more details.
