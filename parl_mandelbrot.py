import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path

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

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
    max_iter=100, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0

    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    # if caller passes an open pool, use it directly (no startup cost)
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))

    # otherwise create our own pool + warm up
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny) # warm-up: load JIT cache in workers
        parts = p.map(_worker, chunks)

    return np.vstack(parts)

if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # --- M2: verify parallel matches serial ---
    #warm up numba jit
    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    serial_result = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    parallel_result = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=4)

    if np.array_equal(serial_result, parallel_result):
        print("Serial and parallel results match!")
    else:
        diff = np.abs(serial_result.astype(np.int64) - parallel_result.astype(np.int64))
        print(f"Results dont match! max diff: {diff.max()}, mismatched pixels: {(diff>0).sum()}")

    # save image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(parallel_result, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    out = Path(__file__).parent / 'mandelbrot.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')

    # --- M3: benchmark sweep ---
    # Serial baseline (Numba already warm from above)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"\nSerial baseline: {t_serial:.4f}s")

    workers_list = []
    speedups_list = []

    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks) # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        workers_list.append(n_workers)
        speedups_list.append(speedup)
        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")

    # Back-solve implied serial fraction using Amdahl's law
    # s = (1/S_peak - 1/p_peak) / (1 - 1/p_peak)
    best_idx = speedups_list.index(max(speedups_list))
    p_peak = workers_list[best_idx]
    s_peak = speedups_list[best_idx]
    if p_peak > 1:
        implied_s = (1/s_peak - 1/p_peak) / (1 - 1/p_peak)
        print(f"\nPeak speedup: {s_peak:.2f}x at p={p_peak}")
        print(f"Implied serial fraction (Amdahl): {implied_s:.4f} ({implied_s*100:.1f}%)")

    #Speedup plot
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(workers_list, speedups_list, 'o-', label='Measured speedup')
    ax2.plot(workers_list, workers_list, '--', color='gray', label='Ideal (linear)')
    ax2.axvline(os.cpu_count(), linestyle=':', color='red', alpha=0.5,
                label=f'Logical cores ({os.cpu_count()})')
    ax2.set_xlabel('Number of workers')
    ax2.set_ylabel('Speedup')
    ax2.set_title(f'Parallel Mandelbrot Speedup ({N}x{N})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.savefig(Path(__file__).parent / 'speedup_plot.png', dpi=150)
    plt.close(fig2)
    print(f"Speedup plot saved")

    # --- L05 M1/M2: chunk count sweep with LIF ---
    # use the best worker count from the sweep above
    best_n_workers = workers_list[best_idx]
    print(f"\n--- Chunk sweep (n_workers={best_n_workers}) ---")

    chunk_mults = [1, 2, 4, 8, 16]
    chunk_counts = []
    chunk_times = []
    chunk_lifs = []

    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    for mult in chunk_mults:
        n_chunks = mult * best_n_workers
        with Pool(processes=best_n_workers) as pool:
            pool.map(_worker, tiny) # warm up jit cache
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter,
                    n_workers=best_n_workers, n_chunks=n_chunks, pool=pool)
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = best_n_workers * t_par / t_serial - 1
        speedup = t_serial / t_par
        chunk_counts.append(n_chunks)
        chunk_times.append(t_par)
        chunk_lifs.append(lif)
        print(f"{n_chunks:4d} chunks: {t_par:.4f}s, speedup={speedup:.2f}x, LIF={lif:.2f}")

    # find sweet spot (lowest LIF)
    best_chunk_idx = chunk_lifs.index(min(chunk_lifs))
    print(f"\nSweet spot: {chunk_counts[best_chunk_idx]} chunks "
          f"(LIF={chunk_lifs[best_chunk_idx]:.2f})")

    # chunk sweep plot
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.plot(chunk_counts, chunk_times, 'o-', label='Parallel time')
    ax3.axhline(t_serial, linestyle='--', color='gray', label=f'Serial ({t_serial:.4f}s)')
    ax3.set_xlabel('n_chunks')
    ax3.set_ylabel('Wall time (s)')
    ax3.set_title(f'Chunk count sweep (n_workers={best_n_workers})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.savefig(Path(__file__).parent / 'chunk_sweep_plot.png', dpi=150)
    plt.close(fig3)
    print("Chunk sweep plot saved")

    # --- L05 M3: comprehensive comparison table ---
    # re-use t_serial from above as numba baseline
    # pull naive + numpy times from mandelbrot.py if you have them,
    # otherwise just print what we can measure here
    print(f"\n--- Comprehensive speedup table (1024x1024) ---")
    best_chunk = chunk_counts[best_chunk_idx]
    best_time = chunk_times[best_chunk_idx]
    print(f"{'Implementation':<20} {'Time (s)':>10} {'Speedup':>10}")
    print("-" * 42)
    print(f"{'Numba (serial)':<20} {t_serial:>10.4f} {'1.00x':>10}")
    print(f"{'Parallel (opt.)':<20} {best_time:>10.4f} {t_serial/best_time:>9.2f}x")

    # amdahl back-solve with the chunk-optimised result
    best_speedup = t_serial / best_time
    if best_n_workers > 1:
        implied_s2 = (1/best_speedup - 1/best_n_workers) / (1 - 1/best_n_workers)
        print(f"\nWith chunking — peak speedup: {best_speedup:.2f}x at p={best_n_workers}")
        print(f"Implied serial fraction: {implied_s2:.4f} ({implied_s2*100:.1f}%)")
        print(f"(Compare to L04 without chunking: {implied_s:.4f} ({implied_s*100:.1f}%))")
