import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
import math

@dataclass
class MandelbrotConfig:
    """Configuration for Mandelbrot set visualization"""
    width: int = 800
    height: int = 600
    max_iter: int = 256
    xmin: float = -2.5
    xmax: float = 1.0
    ymin: float = -1.25
    ymax: float = 1.25


def create_complex_matrix(width, height, xmin, xmax, ymin, ymax):
    """
    Create a complex number matrix representing the complex plane region.
    
    Args:
        width: Width of the image in pixels
        height: Height of the image in pixels
        xmin, xmax: Real axis bounds
        ymin, ymax: Imaginary axis bounds
    
    Returns:
        2D numpy array of complex numbers
    """
    # Create linearly spaced arrays for real and imaginary components
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    
    # Create meshgrid and combine into complex matrix
    real_grid, imag_grid = np.meshgrid(real, imag)
    complex_matrix = real_grid + 1j * imag_grid
    
    return complex_matrix


def mandelbrot_set_membership(complex_matrix, max_iter):
    """
    Determine which complex numbers belong to the Mandelbrot set.
    Uses vectorized operations with complex numbers.
    
    Args:
        complex_matrix: 2D array of complex numbers to test
        max_iter: Maximum iterations before considering a point in the set
    
    Returns:
        2D array of iteration counts (max_iter means in the set)
    """
    # Initialize z to zeros and c to the complex matrix
    c = complex_matrix
    z = np.zeros_like(c)
    
    # Array to store iteration counts
    iter_counts = np.zeros(c.shape, dtype=int)
    
    # Array to track which points are still being calculated
    mask = np.ones(c.shape, dtype=bool)
    
    for i in range(max_iter):
        # Only calculate for points that haven't escaped
        z[mask] = z[mask] ** 2 + c[mask]
        
        # Check which points have escaped (|z| > 2)
        escaped = np.abs(z) > 2
        
        # Update iteration counts for newly escaped points
        newly_escaped = escaped & mask
        iter_counts[newly_escaped] = i
        
        # Update mask
        mask = mask & ~escaped
        
        # If all points have escaped, break early
        if not mask.any():
            break
    
    # Points that never escaped are set to max_iter
    iter_counts[mask] = max_iter
    
    return iter_counts


def naive_escape_time(cx, cy, max_iter):
    """
    Unoptimized naive escape time algorithm using real numbers only.
    Simulates complex number operations: (x + yi)^2 = (x^2 - y^2) + (2xy)i
    
    Args:
        cx: Real component of complex number c
        cy: Imaginary component of complex number c
        max_iter: Maximum iterations
    
    Returns:
        Tuple of (iteration count, final zx, final zy) for smooth coloring
    """
    x = 0.0
    y = 0.0
    
    for i in range(max_iter):
        # Check escape condition: x^2 + y^2 > 4 (equivalent to |z| > 2)
        x2 = x * x
        y2 = y * y
        
        if x2 + y2 > 4.0:
            return i, x, y
        
        # Complex multiplication: (x + yi)^2 + (cx + cyi)
        # Real part: x^2 - y^2 + cx
        # Imaginary part: 2xy + cy
        xtemp = x2 - y2 + cx
        y = 2.0 * x * y + cy
        x = xtemp
    
    return max_iter, x, y


def compute_mandelbrot_naive(width, height, xmin, xmax, ymin, ymax, max_iter):
    """
    Compute Mandelbrot set using naive escape time algorithm.
    
    Returns:
        2D array of iteration counts with smooth coloring values
    """
    result = np.zeros((height, width))
    
    for row in range(height):
        cy = ymin + (ymax - ymin) * row / (height - 1)
        
        for col in range(width):
            cx = xmin + (xmax - xmin) * col / (width - 1)
            
            iter_count, final_x, final_y = naive_escape_time(cx, cy, max_iter)
            
            # Apply normalized iteration count for smooth coloring
            if iter_count < max_iter:
                # Calculate the magnitude at escape
                magnitude = math.sqrt(final_x * final_x + final_y * final_y)
                
                # Normalized iteration count formula
                # This removes banding by adding a fractional component
                smooth_iter = iter_count + 1 - math.log(math.log(magnitude)) / math.log(2)
                result[row, col] = smooth_iter
            else:
                result[row, col] = max_iter
    
    return result


def create_color_mapping(iter_data, max_iter):
    """
    Create smooth color mapping from iteration data.
    
    Args:
        iter_data: 2D array of (smooth) iteration counts
        max_iter: Maximum iteration value
    
    Returns:
        RGB image array
    """
    # Normalize the data
    normalized = np.copy(iter_data)
    
    # Set points in the set to 0
    in_set = (iter_data >= max_iter)
    normalized[in_set] = 0
    
    # Normalize to [0, 1] range for points outside the set
    outside_set = ~in_set
    if outside_set.any():
        max_val = np.max(normalized[outside_set])
        if max_val > 0:
            normalized[outside_set] = normalized[outside_set] / max_val
    
    # Create RGB channels with smooth gradients
    height, width = iter_data.shape
    image = np.zeros((height, width, 3))
    
    # Create a smooth color gradient using sine waves with different phases
    image[:, :, 0] = np.sin(normalized * math.pi * 2 + 0) ** 2  # Red
    image[:, :, 1] = np.sin(normalized * math.pi * 2 + 2) ** 2  # Green
    image[:, :, 2] = np.sin(normalized * math.pi * 2 + 4) ** 2  # Blue
    
    # Set points in the set to black
    image[in_set] = 0
    
    return image


def main():
    """Generate Mandelbrot set visualization"""
    
    # Configuration
    config = MandelbrotConfig(
        width=1200,
        height=900,
        max_iter=256,
        xmin=-2.5,
        xmax=1.0,
        ymin=-1.25,
        ymax=1.25
    )
    
    print("Generating Mandelbrot Set Visualization...")
    print(f"Resolution: {config.width}x{config.height}")
    print(f"Max iterations: {config.max_iter}")
    print(f"Region: [{config.xmin}, {config.xmax}] x [{config.ymin}, {config.ymax}]")
    
    # Method 1: Using complex matrix (faster, for comparison)
    print("\n1. Computing using complex matrix method...")
    complex_matrix = create_complex_matrix(
        config.width, config.height,
        config.xmin, config.xmax,
        config.ymin, config.ymax
    )
    iter_counts_fast = mandelbrot_set_membership(complex_matrix, config.max_iter)
    
    # Method 2: Using naive escape time algorithm with smooth coloring
    print("2. Computing using naive escape time algorithm with smooth coloring...")
    iter_counts_smooth = compute_mandelbrot_naive(
        config.width, config.height,
        config.xmin, config.xmax,
        config.ymin, config.ymax,
        config.max_iter
    )
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Standard method (shows banding)
    ax1 = axes[0]
    im1 = ax1.imshow(iter_counts_fast, extent=[config.xmin, config.xmax, config.ymin, config.ymax],
                     cmap='hot', origin='lower', interpolation='bilinear')
    ax1.set_title('Standard Method (with color bands)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Real axis')
    ax1.set_ylabel('Imaginary axis')
    plt.colorbar(im1, ax=ax1, label='Iterations to escape')
    
    # Plot 2: Smooth coloring
    ax2 = axes[1]
    smooth_image = create_color_mapping(iter_counts_smooth, config.max_iter)
    ax2.imshow(smooth_image, extent=[config.xmin, config.xmax, config.ymin, config.ymax],
               origin='lower', interpolation='bilinear')
    ax2.set_title('Naive Method with Smooth Gradient', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Real axis')
    ax2.set_ylabel('Imaginary axis')
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = 'mandelbrot_set.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to '{output_filename}'")
    
    # Save individual smooth colored version using PIL
    smooth_output = 'mandelbrot_smooth.png'
    smooth_image_uint8 = (smooth_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(smooth_image_uint8)
    pil_image.save(smooth_output)
    print(f"✓ Saved smooth version to '{smooth_output}'")
    
    plt.show()
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()