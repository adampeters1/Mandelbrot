"""
Mandelbrot Set Visualization

This module provides tools for generating and visualizing the Mandelbrot set,
including both complex number operations and a naive escape time algorithm
using real number arithmetic, with smooth coloring via normalized iteration count.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
import math
from PIL import Image


@dataclass
class MandelbrotConfig:
    """Configuration parameters for Mandelbrot set visualization."""
    x_min: float = -2.5
    x_max: float = 1.0
    y_min: float = -1.25
    y_max: float = 1.25
    width: int = 800
    height: int = 600
    max_iterations: int = 256


def create_complex_matrix(config: MandelbrotConfig) -> np.ndarray:
    """
    Create a complex matrix representing points in the complex plane.
    
    This function generates a 2D grid of complex numbers spanning the
    specified region of the complex plane, where each point c = x + yi
    will be tested for membership in the Mandelbrot set.
    
    Args:
        config: MandelbrotConfig object containing the bounds and resolution.
        
    Returns:
        A 2D numpy array of complex numbers representing the complex plane.
    """
    # Create linearly spaced arrays for real and imaginary parts
    real_values = np.linspace(config.x_min, config.x_max, config.width)
    imag_values = np.linspace(config.y_max, config.y_min, config.height)  # Inverted for correct orientation
    
    # Create 2D meshgrid
    real_grid, imag_grid = np.meshgrid(real_values, imag_values)
    
    # Combine into complex matrix: c = real + imag * i
    complex_matrix = real_grid + 1j * imag_grid
    
    return complex_matrix


def compute_mandelbrot_membership(complex_matrix: np.ndarray, 
                                   max_iterations: int) -> np.ndarray:
    """
    Determine which points in the complex matrix belong to the Mandelbrot set.
    
    The Mandelbrot set consists of all complex numbers c for which the iteration
    z_{n+1} = z_n^2 + c (starting with z_0 = 0) does not diverge to infinity.
    A point is considered to have escaped if |z| > 2.
    
    Args:
        complex_matrix: 2D array of complex numbers to test.
        max_iterations: Maximum number of iterations before assuming convergence.
        
    Returns:
        A 2D numpy array of iteration counts (max_iterations = likely in set).
    """
    # Initialize arrays
    height, width = complex_matrix.shape
    iterations = np.zeros((height, width), dtype=np.int32)
    z = np.zeros_like(complex_matrix, dtype=np.complex128)
    c = complex_matrix.copy()
    
    # Track which points haven't escaped yet
    not_escaped = np.ones((height, width), dtype=bool)
    
    for i in range(max_iterations):
        # Apply iteration: z = z^2 + c
        z[not_escaped] = z[not_escaped] ** 2 + c[not_escaped]
        
        # Check escape condition: |z| > 2
        escaped = not_escaped & (np.abs(z) > 2)
        
        # Record iteration count for newly escaped points
        iterations[escaped] = i
        
        # Update mask
        not_escaped[escaped] = False
    
    # Points that never escaped get max_iterations
    iterations[not_escaped] = max_iterations
    
    return iterations


def naive_escape_time_algorithm(x0: float, y0: float, 
                                 max_iterations: int) -> tuple[int, float, float]:
    """
    Unoptimized naive escape time algorithm using real number arithmetic.
    
    This function simulates complex number operations using two real numbers
    to determine if a point escapes the Mandelbrot set. Instead of using
    complex multiplication directly, we expand (a + bi)^2 = (a^2 - b^2) + (2ab)i
    
    The iteration is:
        x_{n+1} = x_n^2 - y_n^2 + x_0
        y_{n+1} = 2 * x_n * y_n + y_0
    
    Escape condition: x^2 + y^2 > 4 (equivalent to |z| > 2)
    
    Args:
        x0: Real part of the complex number c.
        y0: Imaginary part of the complex number c.
        max_iterations: Maximum iterations before assuming point is in set.
        
    Returns:
        Tuple of (iteration_count, final_x, final_y) for smooth coloring.
    """
    x = 0.0
    y = 0.0
    iteration = 0
    
    # Naive loop without any optimizations
    while iteration < max_iterations:
        # Compute x^2 and y^2 separately (naive approach, not caching)
        x_squared = x * x
        y_squared = y * y
        
        # Check escape condition: x^2 + y^2 > 4
        if x_squared + y_squared > 4.0:
            break
        
        # Compute next iteration using real arithmetic
        # (x + yi)^2 = x^2 - y^2 + 2xyi
        # Adding c = x0 + y0*i gives:
        # new_x = x^2 - y^2 + x0
        # new_y = 2xy + y0
        
        new_x = x_squared - y_squared + x0
        new_y = 2.0 * x * y + y0
        
        x = new_x
        y = new_y
        iteration += 1
    
    return iteration, x, y


def compute_normalized_iteration_count(iteration: int, x: float, y: float, 
                                        max_iterations: int) -> float:
    """
    Compute normalized iteration count for smooth coloring.
    
    The normalized iteration count eliminates the banding effect by using
    the fractional escape value. The formula is:
    
        nu = iteration + 1 - log2(log2(|z|))
    
    where |z| is the magnitude at escape. This produces a continuous
    value that smoothly transitions between iteration bands.
    
    Args:
        iteration: The discrete iteration count at which the point escaped.
        x: Final x coordinate (real part of z).
        y: Final y coordinate (imaginary part of z).
        max_iterations: Maximum iterations (used for points in the set).
        
    Returns:
        Normalized iteration count as a float for smooth coloring.
    """
    # Points that didn't escape are in the set
    if iteration >= max_iterations:
        return float(max_iterations)
    
    # Calculate magnitude squared: |z|^2 = x^2 + y^2
    magnitude_squared = x * x + y * y
    
    # Avoid log of zero or negative (shouldn't happen but safety check)
    if magnitude_squared <= 1.0:
        return float(iteration)
    
    # Calculate magnitude
    magnitude = math.sqrt(magnitude_squared)
    
    # Normalized iteration count formula
    # nu = n + 1 - log2(log2(|z|))
    # We use log2(log2(|z|)) = log(log(|z|)) / log(2) / log(2)
    
    # Avoid log of values <= 1
    if magnitude <= 1.0:
        return float(iteration)
    
    log_magnitude = math.log(magnitude)
    
    if log_magnitude <= 0:
        return float(iteration)
    
    # The formula: iteration + 1 - log2(log2(|z|))
    # log2(x) = log(x) / log(2)
    log2_log_magnitude = math.log(log_magnitude) / math.log(2.0)
    
    normalized = iteration + 1.0 - log2_log_magnitude
    
    return max(0.0, normalized)


def generate_smooth_mandelbrot(config: MandelbrotConfig) -> np.ndarray:
    """
    Generate Mandelbrot set with smooth coloring using naive escape time algorithm.
    
    This function iterates over every pixel, using the naive real-number
    escape time algorithm and normalized iteration count for smooth gradients.
    
    Args:
        config: MandelbrotConfig object with visualization parameters.
        
    Returns:
        2D numpy array of normalized iteration counts.
    """
    # Initialize output array
    smooth_iterations = np.zeros((config.height, config.width), dtype=np.float64)
    
    # Calculate pixel dimensions
    pixel_width = (config.x_max - config.x_min) / config.width
    pixel_height = (config.y_max - config.y_min) / config.height
    
    # Iterate over every pixel (naive, unoptimized approach)
    for py in range(config.height):
        # Calculate imaginary component (y-axis is inverted for image coordinates)
        y0 = config.y_max - py * pixel_height
        
        for px in range(config.width):
            # Calculate real component
            x0 = config.x_min + px * pixel_width
            
            # Use naive escape time algorithm
            iteration, final_x, final_y = naive_escape_time_algorithm(
                x0, y0, config.max_iterations
            )
            
            # Compute normalized iteration count for smooth coloring
            smooth_value = compute_normalized_iteration_count(
                iteration, final_x, final_y, config.max_iterations
            )
            
            smooth_iterations[py, px] = smooth_value
        
        # Progress indicator for long computations
        if (py + 1) % 100 == 0:
            print(f"Processing row {py + 1}/{config.height}")
    
    return smooth_iterations


def create_custom_colormap() -> LinearSegmentedColormap:
    """
    Create a custom colormap for aesthetically pleasing Mandelbrot visualization.
    
    Returns:
        A matplotlib LinearSegmentedColormap object.
    """
    # Define color stops for a rich, smooth gradient
    colors = [
        (0.0, 0.0, 0.0),      # Black (in set)
        (0.0, 0.0, 0.3),      # Dark blue
        (0.0, 0.2, 0.5),      # Blue
        (0.0, 0.5, 0.8),      # Light blue
        (0.0, 0.8, 0.8),      # Cyan
        (0.2, 1.0, 0.5),      # Green-cyan
        (0.5, 1.0, 0.0),      # Yellow-green
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.6, 0.0),      # Orange
        (1.0, 0.0, 0.0),      # Red
        (0.6, 0.0, 0.2),      # Dark red
        (0.3, 0.0, 0.3),      # Purple
        (0.0, 0.0, 0.0),      # Back to black for smooth looping
    ]
    
    return LinearSegmentedColormap.from_list('mandelbrot', colors, N=2048)


def visualize_mandelbrot(smooth_iterations: np.ndarray, 
                         config: MandelbrotConfig,
                         save_path: str = "mandelbrot.png") -> None:
    """
    Create and save the final Mandelbrot visualization.
    
    Args:
        smooth_iterations: 2D array of normalized iteration counts.
        config: Configuration used for the computation.
        save_path: File path to save the output image.
    """
    # Create figure with specified size
    fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Normalize the iteration counts for coloring
    # Points in the set (max_iterations) will be colored black
    display_data = smooth_iterations.copy()
    
    # Create mask for points in the set
    in_set_mask = smooth_iterations >= config.max_iterations - 1
    
    # Normalize external points to [0, 1] range for colormap
    external_max = np.max(smooth_iterations[~in_set_mask]) if np.any(~in_set_mask) else 1
    display_data = display_data / external_max
    
    # Set points in the set to 0 (black in our colormap)
    display_data[in_set_mask] = 0
    
    # Apply logarithmic scaling for better visual distribution
    display_data = np.log1p(display_data * 10) / np.log1p(10)
    
    # Display the image
    img = ax.imshow(
        display_data,
        extent=[config.x_min, config.x_max, config.y_min, config.y_max],
        cmap=cmap,
        interpolation='bilinear',
        aspect='equal'
    )
    
    # Add labels and title
    ax.set_xlabel('Real (Re)', fontsize=12)
    ax.set_ylabel('Imaginary (Im)', fontsize=12)
    ax.set_title('The Mandelbrot Set\n$z_{n+1} = z_n^2 + c$', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Iteration Count', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save using matplotlib
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Image saved to: {save_path}")
    
    # Also save using PIL for demonstration
    save_with_pillow(smooth_iterations, config, "mandelbrot_pil.png")
    
    # Display
    plt.show()


def save_with_pillow(smooth_iterations: np.ndarray, 
                     config: MandelbrotConfig,
                     save_path: str) -> None:
    """
    Save the Mandelbrot visualization using PIL/Pillow.
    
    Args:
        smooth_iterations: 2D array of normalized iteration counts.
        config: Configuration used for the computation.
        save_path: File path to save the output image.
    """
    height, width = smooth_iterations.shape
    
    # Create RGB image array
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Normalize iteration counts
    max_val = np.max(smooth_iterations[smooth_iterations < config.max_iterations])
    
    for y in range(height):
        for x in range(width):
            value = smooth_iterations[y, x]
            
            if value >= config.max_iterations - 1:
                # Points in the set are black
                rgb_array[y, x] = [0, 0, 0]
            else:
                # Normalize to [0, 1]
                t = value / max_val
                
                # Apply smooth coloring using sine functions for RGB
                # This creates a pleasing gradient without banding
                r = int(255 * (0.5 + 0.5 * math.sin(3.0 * math.pi * t)))
                g = int(255 * (0.5 + 0.5 * math.sin(3.0 * math.pi * t + 2.094)))  # +2π/3
                b = int(255 * (0.5 + 0.5 * math.sin(3.0 * math.pi * t + 4.189)))  # +4π/3
                
                rgb_array[y, x] = [r, g, b]
    
    # Create and save PIL Image
    img = Image.fromarray(rgb_array, mode='RGB')
    img.save(save_path)
    print(f"PIL image saved to: {save_path}")


def main():
    """Main function to generate and display the Mandelbrot set visualization."""
    print("=" * 60)
    print("Mandelbrot Set Visualization")
    print("=" * 60)
    
    # Create configuration for the "initial" state view
    # This shows the classic full Mandelbrot set
    config = MandelbrotConfig(
        x_min=-2.5,      # Standard view bounds
        x_max=1.0,
        y_min=-1.25,
        y_max=1.25,
        width=800,       # Resolution
        height=600,
        max_iterations=256  # Iteration depth
    )
    
    print(f"\nConfiguration:")
    print(f"  Region: [{config.x_min}, {config.x_max}] x [{config.y_min}, {config.y_max}]")
    print(f"  Resolution: {config.width} x {config.height}")
    print(f"  Max iterations: {config.max_iterations}")
    
    # Step 1: Create the complex matrix
    print("\n[Step 1] Creating complex matrix...")
    complex_matrix = create_complex_matrix(config)
    print(f"  Complex matrix shape: {complex_matrix.shape}")
    print(f"  Sample point (center): {complex_matrix[config.height//2, config.width//2]:.4f}")
    
    # Step 2: Compute membership using complex operations (for verification)
    print("\n[Step 2] Computing Mandelbrot membership (complex method)...")
    iterations_complex = compute_mandelbrot_membership(complex_matrix, config.max_iterations)
    points_in_set = np.sum(iterations_complex == config.max_iterations)
    print(f"  Points likely in set: {points_in_set} ({100*points_in_set/(config.width*config.height):.2f}%)")
    
    # Step 3: Generate smooth visualization using naive escape time algorithm
    print("\n[Step 3] Generating smooth visualization (naive real-number algorithm)...")
    print("  This may take a moment due to the unoptimized approach...")
    smooth_iterations = generate_smooth_mandelbrot(config)
    print("  Smooth iteration computation complete!")
    
    # Step 4: Create and save visualization
    print("\n[Step 4] Creating visualization...")
    visualize_mandelbrot(smooth_iterations, config)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()