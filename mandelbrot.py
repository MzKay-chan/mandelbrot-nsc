import numpy as np
import matplotlib.pyplot as plt
import time, statistics 
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
        return median_t, result

    @profile
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
        
    @profile
    def mandelbrot_numpy(self, c, max_iter):
        z = np.zeros_like(c)
        iteration_counts = np.zeros(c.shape)
        
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2  # Points that haven't escaped yet
            z[mask] = z[mask] ** 2 + c[mask]
            iteration_counts[mask] += 1 
        
        return iteration_counts


    def mandelbrot_numba(self, c, max_iter):
        pass
        return

if __name__ == "__main__":

    #Init class
    mb = mandelbrot()
    
    #Matrix generater:
    c = mb.complex_matrix(-2, 1, -1.5, 1.5, density=1024)

    #Naive benchmarking
    #t_naive, iterations = mb.benchmark(mb.mandelbrot_naive, (-2, 1, -1.5, 1.5, 1024, 1024, 100), 5)
    #t, iterations = mb.benchmark(mb.mandelbrot_numpy, (c, 100), n_runs=5) #Runs mandelbrot algo 5 times 
    
    #Uncomment for Line_profile
    mb.mandelbrot_numpy(c,100)
    mb.mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100)

    #UnComment for cProfile use
    # cprofile.run('mb.mandelbrot_numpy (c,100)', 'numpy_profile.prof')
    # cprofile.run('mb.mandelbrot_naive (-2, 1, -1.5, 1.5, 1024, 1024, 100)', 'naive_profile.prof')
    #
    # for name in ('numpy_profile.prof', 'naive_profile.prof'):
    #     stats = pstats.stats(name)
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
