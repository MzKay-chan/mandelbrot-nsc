import numpy as np
import matplotlib.pyplot as plt
import time 
class mandelbrot:
    
    #TODO: I need to make the naive complex matrix with not numpy

    def complex_matrix(self, xmin,xmax,ymin,ymax, density):
        """
        Function that creates the matrix of complex numbers with some density between the max and min values for x,y
        
        This is optimized using numpy

        Returns = vectors with with calues for height and width

        """
        re = np.linspace(xmin,xmax, int((xmax-xmin) * density) )
        im = np.linspace(ymin,ymax, int((ymax-ymin) * density) )
        return re[np.newaxis, :] + im[:, np.newaxis] * 1j


    def is_stable(self, c, max_iter):
        """
        Function that goes over each point c in grid
        """
        z = 0
        for _ in range(max_iter):
            z = z ** 2 + c
        return abs(z) <= 2

        
    def escape_time(self, c, max_iter):
        z = np.zeros_like(c)
        iteration_counts = np.zeros(c.shape)
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2  # Points that haven't escaped yet
            z[mask] = z[mask] ** 2 + c[mask]
            iteration_counts[mask] = i
        
        return iteration_counts


if __name__ == "__main__":
    mb = mandelbrot()
    time_start = time.time()
    c = mb.complex_matrix(-2, 0.5, -1.5, 1.5, density=1024)
    iterations = mb.escape_time(c, max_iter=100)
    time_stop = time.time()
    elapsed_time = time_stop - time_start
    print(elapsed_time)
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
