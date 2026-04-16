from typing import Callable
import numpy as np
from numba import njit


def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    """
    Escape iteration count for one complex point (pure Python).

    Iterates z_{n+1} = z_n^2 + c from z_0 = 0 until |z|^2 > 4 or max_iter.

    Parameters
    ----------
    c : complex
        Point to test for set membership.
    max_iter : int
        Iteration cap; returned if the orbit never escapes.

    Returns
    -------
    int
        First n where |z_n|^2 > 4, or max_iter.
    """
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter


@njit(cache=True)
def mandelbrot_pixel_numba(c: complex, max_iter: int) -> int:
    """
    Numba-compiled twin of mandelbrot_pixel.

    Same behaviour, same result — separate function so tests can cross-check
    the two implementations against each other.

    Parameters
    ----------
    c : complex
        Point to test for set membership.
    max_iter : int
        Iteration cap; returned if the orbit never escapes.

    Returns
    -------
    int
        First n where |z_n|^2 > 4, or max_iter.
    """
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter


@njit(cache=True)
def mandelbrot_chunk(row_start: int, row_end: int, N: int,
                     x_min: float, x_max: float,
                     y_min: float, y_max: float,
                     max_iter: int) -> np.ndarray:
    """
    Escape counts for a band of rows of an N x N grid.

    The full image samples [x_min, x_max] x [y_min, y_max] on N x N pixels;
    this only fills rows [row_start, row_end). Splitting like this is what
    lets parallel workers each take a disjoint band.

    Parameters
    ----------
    row_start, row_end : int
        Half-open row range, 0 <= row_start < row_end <= N.
    N : int
        Side length of the full grid.
    x_min, x_max : float
        Real-axis.
    y_min, y_max : float
        Imag-axis.
    max_iter : int
        Iteration cap per pixel.

    Returns
    -------
    np.ndarray
    """
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            c = complex(x_min + col*dx, c_imag)
            out[r, col] = mandelbrot_pixel_numba(c, max_iter)
    return out


def mandelbrot_serial(N: int, x_min: float, x_max: float,
                      y_min: float, y_max: float,
                      max_iter: int = 100) -> np.ndarray:
    """
    Full N x N grid, single process. Wraps mandelbrot_chunk over all rows.

    Used as the correctness oracle for parallel runs and as the baseline
    for speedup measurements.

    Parameters
    ----------
    N : int
        Side length.
    x_min, x_max, y_min, y_max : float
        Grid extent.
    max_iter : int
        Iteration cap per pixel.

    Returns
    -------
    np.ndarray
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def mandelbrot_grid_python(N: int, x_min: float, x_max: float,
                           y_min: float, y_max: float,
                           max_iter: int,
                           kernel: Callable[[complex, int], int] = mandelbrot_pixel,
                           ) -> np.ndarray:
    """
    Pure-Python double loop that calls a per-pixel kernel directly.

    Independent reference path — does not go through mandelbrot_chunk, so a
    mismatch with mandelbrot_serial pinpoints which path is wrong.

    Parameters
    ----------
    N : int
        Side length.
    x_min, x_max, y_min, y_max : float
        Grid extent.
    max_iter : int
        Iteration cap per pixel.
    kernel : callable, optional
        (c, max_iter) -> int. Defaults to mandelbrot_pixel; pass
        mandelbrot_pixel_numba to cross-check the JIT version.

    Returns
    -------
    np.ndarray
    """
    out = np.empty((N, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(N):
        c_imag = y_min + r * dy
        for col in range(N):
            out[r, col] = kernel(complex(x_min + col*dx, c_imag), max_iter)
    return out
