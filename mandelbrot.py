import numpy as np
import matplotlib.pyplot as plt
import time, statistics 
from numba import njit
import cProfile, pstats


class mandelbrot:
    
    #This is the old benchmarking implementation. Check cProfile for current implementation
    def benchmark(self, func, args, n_runs=3):
        "Time func, returns median of n_runs"
        times =[]
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = func(*args)
            times.append(time.perf_counter() - t0)
        median_t = statistics.median(times)
        print(f"For function {func}:"
            f"Median: {median_t :.4f}s "
            f"(min={min(times):.4f}, max={max(times):.4f}")
        return median_t

    # @profile
    def mandelbrot_naive (self, xmin , xmax , ymin , ymax , width , height , max_iter =100) :
        x = np . linspace ( xmin , xmax , width )
        y = np . linspace ( ymin , ymax , height )
        result = np . zeros (( height , width ) , dtype = int )
        for i in range ( height ) :
            for j in range ( width ) :
                c = x [ j ] + 1j * y [ i ]
                z = 0
            for n in range ( max_iter ) :
                if abs ( z ) > 2:
                    result [i , j ] = n
                    break
                z = z * z + c
            else :
                result [i , j ] = max_iter
        return result


    def complex_matrix(self, xmin,xmax,ymin,ymax, density):
        """
        Function that creates the matrix of complex numbers with some density between the max and min values for x,y
        
        This is optimized using numpy

        Returns = vectors with with calues for height and width

        """
        re = np.linspace(xmin,xmax, density )
        im = np.linspace(ymin,ymax, density )
        return re[np.newaxis, :] + im[:, np.newaxis] * 1j
        
    # @profile
    def mandelbrot_numpy(self, c, max_iter):
        z = np.zeros_like(c)
        iteration_counts = np.zeros(c.shape)
        
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2  # Points that haven't escaped yet
            z[mask] = z[mask] ** 2 + c[mask]
            iteration_counts[mask] += 1 
        
        return iteration_counts

    @njit
    def mandelbrot_numba_naive(self, xmin, xmax, ymin, ymax, width, height, max_iter):
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        result = np.zeros((height, width), dtype = np.int32)
        for i in range(height):
            for j in range(width):
                c = x[j] + 1j * y[i]
                z = 0j
                n = 0
                while n < max_iter and \
                    z.real*z.real+z.imag*z.imag <= 4.0:
                    z = z*z + c; n +=1
                result[i,j] = n
        return result

    @njit
    def mandelbrot_point_numba(self, c, max_iter=100):
        z = 0j
        for n in range(max_iter):
            if z.real*z.real + z.imag*z.imag > 4.0:
                return n
            z = z*z + c
        return max_iter
    
    def mandelbrot_hybrid(self, xmin, xmax, ymin, ymax, height, width, max_iter=100):
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        result = np.zeros((height, width), dtype = np.int32)
        for i in range(height):
            for j in range(width):
                c = x[j] + 1j * y[i]
                result[i,j] = mb.mandelbrot_point_numba(c, max_iter)


if __name__ == "__main__":

    #Init class
    mb = mandelbrot()
    
    #Matrix generater:
    c = mb.complex_matrix(-2, 1, -1.5, 1.5, density=1024)

    #Naive benchmarking
    #t_naive, iterations = mb.benchmark(mb.mandelbrot_naive, (-2, 1, -1.5, 1.5, 1024, 1024, 100), 5)
    #t, iterations = mb.benchmark(mb.mandelbrot_numpy, (c, 100), n_runs=5) #Runs mandelbrot algo 5 times 
    
    #Uncomment for Line_profile
    # mb.mandelbrot_numpy(c,100)
    # mb.mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100)

    #UnComment for cProfile use
    # cProfile.run('mb.mandelbrot_numpy (c,100)', 'numpy_profile.prof')
    # cProfile.run('mb.mandelbrot_naive (-2, 1, -1.5, 1.5, 1024, 1024, 100)', 'naive_profile.prof')
    #
    # for name in ('numpy_profile.prof', 'naive_profile.prof'):
    #     stats = pstats.Stats(name)
    #     stats.sort_stats('cumulative')
    #     stats.print_stats(10)

    # Plot with colormap
    #plt.figure(figsize=(10, 8))
    #plt.imshow(iterations, extent=[-2, 0.5, -1.5, 1.5], 
    #           cmap='hot', origin='lower', interpolation='bilinear')
    #plt.colorbar(label='Iterations to escape')
    #plt.title('Mandelbrot Set')
    #plt.xlabel('Real')
    #plt.ylabel('Imaginary')
    #plt.savefig('mandelbrot_colormap.png', dpi=300)
    #plt.show()


    #With my current implementation i cannot use numbas njit, as it can not handle python objects
    # Numba implementations
    def mandelbrot_naive (xmin , xmax , ymin , ymax , width , height , max_iter =100) :
        x = np . linspace ( xmin , xmax , width )
        y = np . linspace ( ymin , ymax , height )
        result = np . zeros (( height , width ) , dtype = int )
        for i in range ( height ) :
            for j in range ( width ) :
                c = x [ j ] + 1j * y [ i ]
                z = 0
            for n in range ( max_iter ) :
                if abs ( z ) > 2:
                    result [i , j ] = n
                    break
                z = z * z + c
            else :
                result [i , j ] = max_iter
        return result

    def mandelbrot_numpy(c, max_iter):
        z = np.zeros_like(c)
        iteration_counts = np.zeros(c.shape)
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2  # Points that haven't escaped yet
            z[mask] = z[mask] ** 2 + c[mask]
            iteration_counts[mask] += 1 
        
        return iteration_counts

    @njit
    def mandelbrot_numba_naive(xmin, xmax, ymin, ymax, width, height, max_iter):
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        result = np.zeros((height, width), dtype = np.int32)
        for i in range(height):
            for j in range(width):
                c = x[j] + 1j * y[i]
                z = 0j
                n = 0
                while n < max_iter and \
                    z.real*z.real+z.imag*z.imag <= 4.0:
                    z = z*z + c; n +=1
                result[i,j] = n
        return result

    @njit
    def mandelbrot_point_numba(c, max_iter=100):
        z = 0j
        for n in range(max_iter):
            if z.real*z.real + z.imag*z.imag > 4.0:
                return n
            z = z*z + c
        return max_iter
    
    def mandelbrot_hybrid(xmin, xmax, ymin, ymax, height, width, max_iter=100):
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        result = np.zeros((height, width), dtype = np.int32)
        for i in range(height):
            for j in range(width):
                c = x[j] + 1j * y[i]
                result[i,j] = mandelbrot_point_numba(c, max_iter)

    @njit
    def mandelbrot_numba_typed(xmin, xmax, ymin, ymax, height, width, max_iter=100, dtype=np.float64):
        x = np.linspace(xmin, xmax, width).astype(dtype)
        y = np.linspace(ymin, ymax, height).astype(dtype)
        result = np.zeros((height, width), dtype = np.int32)
        for i in range(height):
            for j in range(width):
                c = x[j] + 1j * y[i]
                result[i,j] = mandelbrot_point_numba(c, max_iter)
        return result

    def mandelbrot_numba_typed16(xmin, xmax, ymin, ymax, height, width, max_iter=100, dtype=np.float64):
        x = np.linspace(xmin, xmax, width).astype(dtype)
        y = np.linspace(ymin, ymax, height).astype(dtype)
        result = np.zeros((height, width), dtype = np.int32)
        for i in range(height):
            for j in range(width):
                c = x[j] + 1j * y[i]
                result[i,j] = mandelbrot_point_numba(c, max_iter)
        return result

    #Numba benchmarking for both implementations
    mandelbrot_numba_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100)
    mandelbrot_hybrid(-2, 1, -1.5, 1.5, 1024, 1024, 100)

    t_naive = mb.benchmark(mandelbrot_naive, (-2, 1, -1.5, 1.5, 1024, 1024, 100), 5)
    t_numpy = mb.benchmark(mandelbrot_numpy, (c, 100))
    t_full = mb.benchmark(mandelbrot_numba_naive, (-2, 1, -1.5, 1.5, 1024, 1024, 100), 5)
    t_hybrid = mb.benchmark(mandelbrot_hybrid, (-2, 1, -1.5, 1.5, 1024, 1024, 100), 5)

    print(f" Naive:     {t_naive:.3f}s")
    print(f" Numpy:       {t_numpy:.3f}s    ({t_naive/t_numpy:1f}x)")
    print(f" Fully compiled:        {t_full:.3f}s       ({t_naive/t_full:.3f}x)")

    #Numba data type testing:
    for dtype in [np.float16, np.float32, np.float64]:
        t0 = time.perf_counter()
        if dtype == np.float16:
            mandelbrot_numba_typed16(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=dtype)
        else:
            mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=dtype)
        print(f"{dtype.__name__}: {time.perf_counter()-t0:.3f}s")

    r16 = mandelbrot_numba_typed16(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=np.float16)
    r32 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=np.float32)
    r64 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, result, title in zip(axes, [r16, r32, r64], ['float16', 'float32', 'float64 (ref)']):
        ax.imshow(result, cmap='hot')
        ax.set_title(title); ax.axis('off')
    plt.savefig('precision_comparison.png', dpi=150)
    print(f"Max diff float32 vs float 64: {np.abs(r32-r64).max()}")
    print(f"Max diff float16 vs float 64: {np.abs(r16-r64).max()}")
