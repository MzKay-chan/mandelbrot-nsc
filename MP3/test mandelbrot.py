import pytest
from numba import njit
import numpy as np

#Mandelbrot pixel

def mandelbrot_pixel(c, max_iter=100):
    """blah"""
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter


#Mandelbrot pixel numba
@njit
def mandelbrot_point_numba(c, max_iter=100):
    """blah"""
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter


#Mandelbrot chunk
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter=100):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

#Mandelbrot serial

#Mandelbrot parallel

# Test cases for mandelbrot_pixel
KNOWN_CASES = [
    (0+0j, 100, 100), # origin: never escapes
    (5.0+0j, 100, 1), # far outside, escapes on iteration 1
    (-2.5+0j, 100, 1), # left tip of set
]
IMPLEMENTATIONS = [mandelbrot_pixel, mandelbrot_point_numba]
@pytest.mark.parametrize("impl", IMPLEMENTATIONS) # independent axis
@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES) # bundled
def test_pixel_all(impl, c, max_iter, expected):
    assert impl(c, max_iter) == expected # 2 x 3 = 6 tests