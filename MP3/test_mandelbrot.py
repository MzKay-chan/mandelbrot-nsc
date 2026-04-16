"""
Tests for the Mandelbrot kernels (MP3 M1).

Run from MP3/:
    pytest -v
    pytest --cov=mandelbrot -v
"""

import numpy as np
import pytest

from mandelbrot import (
    mandelbrot_chunk,
    mandelbrot_grid_python,
    mandelbrot_pixel,
    mandelbrot_pixel_numba,
    mandelbrot_serial,
)


# Analytically justified pixel cases — derived from the iteration, no plot read-off.
#  0 + 0j   -> z stays at 0, in set, returns max_iter
#  -1 + 0j  -> period-2 cycle 0 <-> -1, in set, returns max_iter
#  5 + 0j   -> z_1 = 5, |z|^2 = 25 > 4, returns 1
#  -2.5+0j  -> z_1 = -2.5, |z|^2 = 6.25 > 4, returns 1
#  2 + 0j   -> z_1 = 2 (|z|^2 = 4, NOT > 4 strict), z_2 = 6, returns 2
KNOWN_PIXEL_CASES = [
    (0 + 0j, 100, 100),
    (-1 + 0j, 100, 100),
    (5 + 0j, 100, 1),
    (-2.5 + 0j, 100, 1),
    (2 + 0j, 100, 2),
]

IMPLEMENTATIONS = [mandelbrot_pixel, mandelbrot_pixel_numba]


@pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=["python", "numba"])
@pytest.mark.parametrize("c, max_iter, expected", KNOWN_PIXEL_CASES)
def test_pixel_known_values(impl, c, max_iter, expected):
    # both kernels must hit the analytically expected escape count
    assert impl(c, max_iter) == expected


def test_pixel_python_matches_numba_random():
    # cross-check on a seeded random sweep — catches any JIT-vs-Python drift
    rng = np.random.default_rng(seed=42)
    points = rng.uniform(-2.0, 2.0, size=(50, 2))
    max_iter = 100
    for re, im in points:
        c = complex(re, im)
        assert mandelbrot_pixel(c, max_iter) == mandelbrot_pixel_numba(c, max_iter)


def test_chunk_shape_and_dtype():
    # chunk must give back the requested band shape with int32 escape counts
    N = 16
    out = mandelbrot_chunk(2, 7, N, -2.0, 1.0, -1.5, 1.5, max_iter=50)
    assert out.shape == (5, N)
    assert out.dtype == np.int32


def test_serial_matches_python_oracle():
    # full Numba grid must match the pure-Python pixel-by-pixel oracle exactly
    N = 32
    bounds = dict(x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5)
    max_iter = 100

    expected = mandelbrot_grid_python(N=N, max_iter=max_iter, **bounds)
    actual = mandelbrot_serial(N=N, max_iter=max_iter, **bounds)
    np.testing.assert_array_equal(actual, expected)


def test_serial_origin_pixel_in_set():
    # odd N so the centre pixel lands exactly on c = 0+0j (origin is in set)
    N = 33
    max_iter = 100
    grid = mandelbrot_serial(N, -1.0, 1.0, -1.0, 1.0, max_iter=max_iter)
    centre = N // 2
    assert grid[centre, centre] == max_iter
