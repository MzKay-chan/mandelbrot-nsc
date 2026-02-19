import numpy as np
import matplotlib.pyplot as plt
import time, statistics 
class mandelbrot:
    
    #TODO: I need to make the naive complex matrix with not numpy
    def benchmark(self, func, c, max_iter, n_runs=3):
        "Time func, returns median of n_runs"
        times =[]
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = func(c, max_iter)
            times.append(time.perf_counter() - t0)
        median_t = statistics.median(times)
        print(f"Median: {median_t :.4f}s "
            f"(min={min(times):.4f}, max={max(times):.4f}")
        return median_t, result

    def complex_matrix(self, xmin,xmax,ymin,ymax, density):
        """
        Function that creates the matrix of complex numbers with some density between the max and min values for x,y
        
        This is optimized using numpy

        Returns = vectors with with calues for height and width

        """
        re = np.linspace(xmin,xmax, density )
        im = np.linspace(ymin,ymax, density )
        return re[np.newaxis, :] + im[:, np.newaxis] * 1j
        

    def escape_time(self, c, max_iter):
        z = np.zeros_like(c)
        iteration_counts = np.zeros(c.shape)
        
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2  # Points that haven't escaped yet
            z[mask] = z[mask] ** 2 + c[mask]
            iteration_counts[mask] += 1 
        
        return iteration_counts


if __name__ == "__main__":
    mb = mandelbrot()
    c = mb.complex_matrix(-2, 1, -1.5, 1.5, density=1024)
    t, iterations = mb.benchmark(mb.escape_time, c, max_iter=100, n_runs=5) #Runs my mandelbrot algo 5 times 

    # Plot with colormap
    plt.figure(figsize=(10, 8))
    plt.imshow(iterations, extent=[-2, 0.5, -1.5, 1.5], 
               cmap='hot', origin='lower', interpolation='bilinear')
    plt.colorbar(label='Iterations to escape')
    plt.title('Mandelbrot Set')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.savefig('mandelbrot_colormap.png', dpi=300)
    plt.show()
