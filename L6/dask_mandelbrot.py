from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, numpy as np, time, statistics
from numba import njit
import matplotlib.pyplot as plt
from pathlib import Path


def mandelbrot_dask(N, x_min, x_max, y_min, y_max, 
                    max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
        row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def chunk_sweep(client, n_workers, chunk_mults, n, x_min, x_max, y_min, y_max, max_iter, t_serial):
    chunk_counts, wall_times, lifs = [], [], []

    for mult in chunk_mults:
        n_chunks = mult * n_workers
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                            max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        speedup = t_serial / t_par

        chunk_counts.append(n_chunks)
        wall_times.append(t_par)
        lifs.append(lif)
        print(f"n_chunks={n_chunks} (x{mult}):"
              f"{t_par}s speedup={speedup}x LIF={lif}")
    best_idx = lifs.index(min(lifs))
    print(f"\n Sweet spot: {chunk_counts[best_idx]} chunks"
          f"(LIF={lifs[best_idx], {wall_times[best_idx]}})")
    return chunk_counts, wall_times, lifs


def plot_chunk_sweep(chunk_counts, wall_times, lifs, t_serial, n_workers):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(chunk_counts, wall_times, 'o-', label='Dask parallel (median)')
    ax.axhline(t_serial, linestyle='--', color='gray',
               label=f'Serial baseline ({t_serial:.3f}s)')

    best_idx = lifs.index(min(lifs))
    ax.axvline(chunk_counts[best_idx], linestyle=':', color='green', alpha=0.7,
               label=f'Sweet spot (n={chunk_counts[best_idx]}, LIF={lifs[best_idx]:.2f})')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('n_chunks (log₂ scale)')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Dask Chunk Sweep — {n_workers} workers')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(__file__).parent / 'dask_chunk_sweep.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    N, max_iter = 8192, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    client  = Client("tcp://10.92.1.226:8786")
    n_workers = len(client.scheduler_info()['workers'])
    print(f"Cluster: {n_workers} workers")

    # warm up numpa
    client.run(lambda: mandelbrot_chunk(
        0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    print("JIT warm-up done")

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial baseline: {t_serial:.4f}s")

    chunk_mults = [1, 2, 4, 8, 16]
    chunk_counts, wall_times, lifs = chunk_sweep(
        client, n_workers, chunk_mults,
        N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, t_serial)

    plot_chunk_sweep(chunk_counts, wall_times, lifs, t_serial, n_workers)

    client.close()
