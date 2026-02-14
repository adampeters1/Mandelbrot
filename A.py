"""
Interactive Mandelbrot Set Viewer

A PyQt6-based interactive viewer with threaded computation, scroll-wheel zooming
centred on mouse position, and precision limit detection.
"""
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QSlider, QHBoxLayout, QGroupBox

from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, 
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QImage, QPixmap

from PIL import Image, ImageDraw, ImageFont


import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, 
    QWidget, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMutex, QWaitCondition
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent

from PIL import Image, ImageDraw, ImageFont

from numba import jit, prange
from numba import config



import math

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMutex, QWaitCondition
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from functools import lru_cache
import hashlib

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

# Numba for JIT compilation - provides near-C performance
from numba import jit, prange, complex128, float64, int32, boolean
from numba import config

# Enable parallel processing in Numba
config.THREADING_LAYER = 'threadsafe'

@dataclass 
class ViewState:
    """Represents the current view into the complex plane."""
    centre_real: float = -0.5
    centre_imag: float = 0.0
    zoom_level: float = 1.0
    
    initial_width: float = 3.5
    initial_height: float = 2.5
    
    @property
    def current_width(self) -> float:
        return self.initial_width / self.zoom_level
    
    @property
    def current_height(self) -> float:
        return self.initial_height / self.zoom_level
    
    @property
    def x_min(self) -> float:
        return self.centre_real - self.current_width / 2
    
    @property
    def x_max(self) -> float:
        return self.centre_real + self.current_width / 2
    
    @property
    def y_min(self) -> float:
        return self.centre_imag - self.current_height / 2
    
    @property
    def y_max(self) -> float:
        return self.centre_imag + self.current_height / 2
    
    def pixel_to_complex(self, pixel_x: int, pixel_y: int,
                         image_width: int, image_height: int) -> tuple[float, float]:
        """Convert pixel coordinates to complex plane coordinates."""
        real = self.x_min + (pixel_x / image_width) * self.current_width
        imag = self.y_max - (pixel_y / image_height) * self.current_height
        return real, imag
    
    def copy(self) -> 'ViewState':
        """Create a copy of this view state."""
        return ViewState(
            centre_real=self.centre_real,
            centre_imag=self.centre_imag,
            zoom_level=self.zoom_level,
            initial_width=self.initial_width,
            initial_height=self.initial_height
        )
    
    def to_metadata_string(self) -> str:
        """Generate a metadata string for embedding in saved images."""
        return (
            f"Mandelbrot Set\n"
            f"Centre: {self.centre_real:.15g} + {self.centre_imag:.15g}i\n"
            f"Zoom: {self.zoom_level:.6e}x\n"
            f"Region: [{self.x_min:.15g}, {self.x_max:.15g}] x "
            f"[{self.y_min:.15g}, {self.y_max:.15g}]"
        )


@dataclass
class RenderConfig:
    """Enhanced configuration with progressive rendering settings."""
    width: int = 800
    height: int = 600
    max_iterations: int = 256
    
    # Progressive rendering passes (each is a divisor of resolution)
    # Renders at 1/8, 1/4, 1/2, then full resolution
    progressive_passes: Tuple[int, ...] = (8, 4, 2, 1)
    
    # Tile cache settings
    tile_size: int = 128
    max_cached_tiles: int = 256
    
    # Precision limit for float64
    min_pixel_spacing: float = field(default_factory=lambda: np.finfo(np.float64).eps * 1000)
    
    # Adaptive iteration scaling with zoom
    base_iterations: int = 256
    iteration_zoom_factor: float = 50.0  # Additional iterations per order of magnitude zoom

@jit(float64[:, :](float64, float64, float64, float64, int32, int32, int32),
     nopython=True, parallel=True, cache=True, fastmath=True)
def compute_mandelbrot_numba(x_min: float, x_max: float, 
                              y_min: float, y_max: float,
                              width: int, height: int,
                              max_iterations: int) -> np.ndarray:
    """
    Numba JIT-compiled Mandelbrot computation with parallel processing.
    
    This function is compiled to machine code and executes across multiple
    CPU cores simultaneously, providing substantial speedup over pure NumPy.
    
    Args:
        x_min, x_max: Real axis bounds.
        y_min, y_max: Imaginary axis bounds.
        width, height: Output image dimensions.
        max_iterations: Maximum iteration count.
        
    Returns:
        2D array of normalised iteration counts.
    """
    # Output array
    result = np.zeros((height, width), dtype=np.float64)
    
    # Pixel dimensions
    pixel_width = (x_max - x_min) / width
    pixel_height = (y_max - y_min) / height
    
    # Parallel loop over rows
    for py in prange(height):
        # Imaginary component for this row
        y0 = y_max - py * pixel_height
        
        for px in range(width):
            # Real component for this column
            x0 = x_min + px * pixel_width
            
            # Iteration variables
            x = 0.0
            y = 0.0
            x_squared = 0.0
            y_squared = 0.0
            
            iteration = 0
            
            # Main iteration loop with early bailout optimisation
            # Using x² + y² <= 4 as escape radius
            # Also using the squared values to avoid redundant multiplication
            while x_squared + y_squared <= 4.0 and iteration < max_iterations:
                # y = 2xy + y0
                y = 2.0 * x * y + y0
                # x = x² - y² + x0
                x = x_squared - y_squared + x0
                
                x_squared = x * x
                y_squared = y * y
                iteration += 1
            
            # Compute smooth iteration count
            if iteration < max_iterations:
                # Normalised iteration count: n + 1 - log2(log2(|z|))
                log_zn = 0.5 * np.log(x_squared + y_squared)  # log(|z|)
                smooth_val = iteration + 1.0 - np.log(log_zn) / np.log(2.0)
                result[py, px] = smooth_val
            else:
                result[py, px] = max_iterations
    
    return result


@jit(float64[:, :](float64, float64, float64, float64, int32, int32, int32, int32),
     nopython=True, parallel=True, cache=True, fastmath=True)
def compute_mandelbrot_subsampled(x_min: float, x_max: float,
                                   y_min: float, y_max: float,
                                   width: int, height: int,
                                   max_iterations: int,
                                   subsample: int) -> np.ndarray:
    """
    Compute Mandelbrot at reduced resolution for progressive rendering.
    
    Computes every nth pixel and returns a smaller array that can be
    upscaled for quick preview display.
    
    Args:
        x_min, x_max: Real axis bounds.
        y_min, y_max: Imaginary axis bounds.
        width, height: Full output image dimensions.
        max_iterations: Maximum iteration count.
        subsample: Subsampling factor (compute every nth pixel).
        
    Returns:
        2D array of size (height/subsample, width/subsample).
    """
    # Output dimensions
    out_height = height // subsample
    out_width = width // subsample
    
    result = np.zeros((out_height, out_width), dtype=np.float64)
    
    # Pixel dimensions at full resolution
    pixel_width = (x_max - x_min) / width
    pixel_height = (y_max - y_min) / height
    
    # Parallel loop
    for out_py in prange(out_height):
        # Map to full resolution pixel
        py = out_py * subsample
        y0 = y_max - py * pixel_height
        
        for out_px in range(out_width):
            px = out_px * subsample
            x0 = x_min + px * pixel_width
            
            x = 0.0
            y = 0.0
            x_squared = 0.0
            y_squared = 0.0
            iteration = 0
            
            while x_squared + y_squared <= 4.0 and iteration < max_iterations:
                y = 2.0 * x * y + y0
                x = x_squared - y_squared + x0
                x_squared = x * x
                y_squared = y * y
                iteration += 1
            
            if iteration < max_iterations:
                log_zn = 0.5 * np.log(x_squared + y_squared)
                smooth_val = iteration + 1.0 - np.log(log_zn) / np.log(2.0)
                result[out_py, out_px] = smooth_val
            else:
                result[out_py, out_px] = max_iterations
    
    return result


@jit(nopython=True, cache=True, fastmath=True)
def upscale_nearest(small: np.ndarray, factor: int) -> np.ndarray:
    """
    Fast nearest-neighbour upscaling using Numba.
    
    Args:
        small: Input array to upscale.
        factor: Upscaling factor.
        
    Returns:
        Upscaled array.
    """
    small_h, small_w = small.shape
    large_h = small_h * factor
    large_w = small_w * factor
    
    result = np.zeros((large_h, large_w), dtype=np.float64)
    
    for y in range(large_h):
        src_y = y // factor
        for x in range(large_w):
            src_x = x // factor
            result[y, x] = small[src_y, src_x]
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def iterations_to_rgb_numba(iterations: np.ndarray, 
                             max_iterations: int) -> np.ndarray:
    """
    Numba-optimised conversion of iteration counts to RGB.
    
    Args:
        iterations: 2D array of iteration counts.
        max_iterations: Maximum iteration value.
        
    Returns:
        3D RGB array (height, width, 3).
    """
    height, width = iterations.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Find max for normalisation (excluding points in set)
    max_external = 0.0
    for y in range(height):
        for x in range(width):
            val = iterations[y, x]
            if val < max_iterations - 1 and val > max_external:
                max_external = val
    
    if max_external <= 0:
        max_external = 1.0
    
    # Convert to RGB in parallel
    for y in prange(height):
        for x in range(width):
            val = iterations[y, x]
            
            if val >= max_iterations - 1:
                # In set - black
                rgb[y, x, 0] = 0
                rgb[y, x, 1] = 0
                rgb[y, x, 2] = 0
            else:
                # Normalise and apply colour mapping
                t = (val / max_external) * 3.0 * np.pi
                
                rgb[y, x, 0] = np.uint8(127.5 * (1.0 + np.sin(t)))
                rgb[y, x, 1] = np.uint8(127.5 * (1.0 + np.sin(t + 2.094)))
                rgb[y, x, 2] = np.uint8(127.5 * (1.0 + np.sin(t + 4.189)))
    
    return rgb


class TileCache:
    """
    Cache for computed Mandelbrot tiles.
    
    Stores computed tiles indexed by their position and zoom level,
    allowing instant display when revisiting previously computed regions.
    """
    
    def __init__(self, max_tiles: int = 256):
        self.max_tiles = max_tiles
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order: list = []  # LRU tracking
        self.mutex = QMutex()
    
    def _make_key(self, x_min: float, x_max: float, 
                  y_min: float, y_max: float,
                  width: int, height: int,
                  max_iter: int) -> str:
        """Generate a unique cache key for tile parameters."""
        # Use a hash of the parameters
        key_data = f"{x_min:.15g},{x_max:.15g},{y_min:.15g},{y_max:.15g},{width},{height},{max_iter}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, x_min: float, x_max: float,
            y_min: float, y_max: float,
            width: int, height: int,
            max_iter: int) -> Optional[np.ndarray]:
        """
        Retrieve a cached tile if available.
        
        Returns None if tile is not in cache.
        """
        key = self._make_key(x_min, x_max, y_min, y_max, width, height, max_iter)
        
        self.mutex.lock()
        try:
            if key in self.cache:
                # Update access order for LRU
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key].copy()
            return None
        finally:
            self.mutex.unlock()
    
    def put(self, x_min: float, x_max: float,
            y_min: float, y_max: float,
            width: int, height: int,
            max_iter: int,
            data: np.ndarray) -> None:
        """
        Store a computed tile in the cache.
        
        Evicts least recently used tiles if cache is full.
        """
        key = self._make_key(x_min, x_max, y_min, y_max, width, height, max_iter)
        
        self.mutex.lock()
        try:
            # Evict if necessary
            while len(self.cache) >= self.max_tiles and self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            self.cache[key] = data.copy()
            self.access_order.append(key)
        finally:
            self.mutex.unlock()
    
    def clear(self) -> None:
        """Clear all cached tiles."""
        self.mutex.lock()
        try:
            self.cache.clear()
            self.access_order.clear()
        finally:
            self.mutex.unlock()


def get_adaptive_iterations(zoom_level: float, config: RenderConfig) -> int:
    """
    Calculate appropriate iteration count based on zoom level.
    
    Deeper zooms require more iterations to resolve detail near the
    set boundary.
    
    Args:
        zoom_level: Current zoom multiplier.
        config: Render configuration.
        
    Returns:
        Recommended maximum iteration count.
    """
    if zoom_level <= 1.0:
        return config.base_iterations
    
    # Add iterations logarithmically with zoom
    zoom_orders = np.log10(zoom_level)
    additional = int(zoom_orders * config.iteration_zoom_factor)
    
    return config.base_iterations + additional


def compute_mandelbrot_vectorised(view: ViewState, config: RenderConfig) -> np.ndarray:
    """
    Vectorised Mandelbrot computation using NumPy.
    
    This replaces the naive per-pixel algorithm with efficient array operations.
    Computes normalised iteration counts for smooth colouring.
    
    Args:
        view: Current view state defining the region to compute.
        config: Rendering configuration.
        
    Returns:
        2D array of normalised iteration counts.
    """
    # Create coordinate arrays
    real_vals = np.linspace(view.x_min, view.x_max, config.width, dtype=np.float64)
    imag_vals = np.linspace(view.y_max, view.y_min, config.height, dtype=np.float64)
    
    # Create meshgrid for all points
    real_grid, imag_grid = np.meshgrid(real_vals, imag_vals)
    
    # Initialise arrays for iteration
    # c is the constant for each point (doesn't change)
    c_real = real_grid.copy()
    c_imag = imag_grid.copy()
    
    # z starts at 0
    z_real = np.zeros_like(c_real)
    z_imag = np.zeros_like(c_imag)
    
    # Track iteration counts and final magnitudes for smooth colouring
    iterations = np.zeros((config.height, config.width), dtype=np.float64)
    
    # Mask of points that haven't escaped yet
    not_escaped = np.ones((config.height, config.width), dtype=bool)
    
    # Main iteration loop
    for i in range(config.max_iterations):
        # Only compute for points that haven't escaped
        # z = z^2 + c using real arithmetic:
        # new_real = real^2 - imag^2 + c_real
        # new_imag = 2 * real * imag + c_imag
        
        z_real_squared = z_real[not_escaped] ** 2
        z_imag_squared = z_imag[not_escaped] ** 2
        
        # Compute new values
        new_real = z_real_squared - z_imag_squared + c_real[not_escaped]
        new_imag = 2.0 * z_real[not_escaped] * z_imag[not_escaped] + c_imag[not_escaped]
        
        z_real[not_escaped] = new_real
        z_imag[not_escaped] = new_imag
        
        # Check escape condition: |z|^2 > 4
        magnitude_squared = z_real ** 2 + z_imag ** 2
        escaped_this_iteration = not_escaped & (magnitude_squared > 4.0)
        
        # Compute smooth iteration count for escaped points
        # Formula: i + 1 - log2(log2(|z|))
        if np.any(escaped_this_iteration):
            escaped_magnitude = np.sqrt(magnitude_squared[escaped_this_iteration])
            
            # Smooth colouring formula
            log_zn = np.log(escaped_magnitude)
            smooth_val = i + 1 - np.log(log_zn) / np.log(2.0)
            
            iterations[escaped_this_iteration] = smooth_val
        
        # Update mask
        not_escaped[escaped_this_iteration] = False
        
        # Early exit if all points have escaped
        if not np.any(not_escaped):
            break
    
    # Points that never escaped get max_iterations
    iterations[not_escaped] = config.max_iterations
    
    return iterations


def iterations_to_rgb(iterations: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Convert iteration counts to RGB image array.
    
    Uses sinusoidal colour mapping for smooth, aesthetically pleasing gradients.
    
    Args:
        iterations: 2D array of normalised iteration counts.
        max_iterations: Maximum iteration value.
        
    Returns:
        3D array of shape (height, width, 3) with RGB values.
    """
    height, width = iterations.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Mask for points in the set
    in_set = iterations >= max_iterations - 1
    
    # Normalise iteration counts for points outside the set
    normalised = np.zeros_like(iterations)
    if np.any(~in_set):
        max_iter_external = np.max(iterations[~in_set])
        if max_iter_external > 0:
            normalised = iterations / max_iter_external
    
    # Apply smooth colouring using sine waves
    # This creates a continuous colour cycle without banding
    t = normalised * 3.0 * np.pi
    
    rgb[:, :, 0] = (127.5 * (1 + np.sin(t))).astype(np.uint8)
    rgb[:, :, 1] = (127.5 * (1 + np.sin(t + 2.094))).astype(np.uint8)  # +2π/3
    rgb[:, :, 2] = (127.5 * (1 + np.sin(t + 4.189))).astype(np.uint8)  # +4π/3
    
    # Points in the set are black
    rgb[in_set] = [0, 0, 0]
    
    return rgb


def colour_selector(iterations: np.ndarray, max_iterations: int,
                    hue_shift: float = 0.0, saturation: float = 1.0,
                    lightness: float = 0.5) -> np.ndarray:
    """
    Convert iteration counts to RGB with adjustable HSL parameters.
    
    Args:
        iterations: 2D array of iteration counts.
        max_iterations: Maximum iteration value.
        hue_shift: Hue rotation in range [0, 1].
        saturation: Saturation multiplier in range [0, 1].
        lightness: Lightness adjustment in range [0, 1].
        
    Returns:
        3D RGB array (height, width, 3).
    """
    height, width = iterations.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Mask for points in the set
    in_set = iterations >= max_iterations - 1
    
    # Normalise iteration counts for points outside the set
    normalised = np.zeros_like(iterations)
    max_external = np.max(iterations[~in_set]) if np.any(~in_set) else 1.0
    if max_external > 0:
        normalised = iterations / max_external
    
    # Calculate hue with shift (wrapping around 0-1)
    hue = (normalised + hue_shift) % 1.0
    
    # Create saturation and lightness arrays
    sat = np.full_like(hue, saturation)
    lit = np.full_like(hue, lightness)
    
    # HSL to RGB conversion
    c = (1 - np.abs(2 * lit - 1)) * sat
    x = c * (1 - np.abs((hue * 6) % 2 - 1))
    m = lit - c / 2
    
    # Determine RGB based on hue sector
    hue_sector = (hue * 6).astype(int) % 6
    
    r = np.zeros_like(hue)
    g = np.zeros_like(hue)
    b = np.zeros_like(hue)
    
    # Sector 0: R=C, G=X, B=0
    mask = hue_sector == 0
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0
    
    # Sector 1: R=X, G=C, B=0
    mask = hue_sector == 1
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0
    
    # Sector 2: R=0, G=C, B=X
    mask = hue_sector == 2
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
    
    # Sector 3: R=0, G=X, B=C
    mask = hue_sector == 3
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
    
    # Sector 4: R=X, G=0, B=C
    mask = hue_sector == 4
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
    
    # Sector 5: R=C, G=0, B=X
    mask = hue_sector == 5
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
    
    # Add m and convert to 0-255
    rgb[:, :, 0] = np.clip((r + m) * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip((g + m) * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip((b + m) * 255, 0, 255).astype(np.uint8)
    
    # Points in the set are black
    rgb[in_set] = [0, 0, 0]
    
    return rgb


class ComputeWorker(QThread):
    """
    Enhanced worker thread with progressive rendering and caching.
    
    Emits multiple signals during computation to provide quick previews
    followed by full resolution results.
    """
    
    # Signal for each progressive pass (rgb_array, view_state, is_final)
    computation_progress = pyqtSignal(np.ndarray, object, bool, object, int)
    
    def __init__(self, config: RenderConfig):
        super().__init__()
        self.config = config
        
        # Thread synchronisation
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # Request state
        self.pending_view: Optional['ViewState'] = None
        self.current_request_id: int = 0
        self.should_stop = False
        self.has_work = False
        
        # Tile cache
        self.tile_cache = TileCache(config.max_cached_tiles)
        
        # HSL colour parameters
        self.hue_shift: float = 0.0
        self.saturation: float = 1.0
        self.lightness: float = 0.5
        
        # Warm up Numba JIT compilation on first run
        self._warmup_jit()
    
    def set_colour_params(self, hue: float, saturation: float, lightness: float) -> None:
        """Update colour parameters and trigger recolour."""
        self.mutex.lock()
        self.hue_shift = hue
        self.saturation = saturation
        self.lightness = lightness
        self.mutex.unlock()

    def _warmup_jit(self) -> None:
        """Pre-compile Numba functions to avoid delay on first zoom."""
        # Small test computation to trigger JIT compilation
        _ = compute_mandelbrot_numba(-2.0, 1.0, -1.0, 1.0, 16, 16, 32)
        _ = compute_mandelbrot_subsampled(-2.0, 1.0, -1.0, 1.0, 16, 16, 32, 4)
        test_iter = np.zeros((4, 4), dtype=np.float64)
        _ = iterations_to_rgb_numba(test_iter, 32)
        _ = upscale_nearest(test_iter, 2)
    
    def request_computation(self, view: 'ViewState') -> None:
        """Request computation for a new view, cancelling any pending work."""
        self.mutex.lock()
        self.pending_view = view.copy()
        self.current_request_id += 1
        self.has_work = True
        self.condition.wakeOne()
        self.mutex.unlock()
    
    def stop(self) -> None:
        """Signal worker to stop."""
        self.mutex.lock()
        self.should_stop = True
        self.has_work = True
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()
    
    def _is_request_stale(self, request_id: int) -> bool:
        """Check if a newer request has superseded this one."""
        self.mutex.lock()
        stale = request_id != self.current_request_id
        self.mutex.unlock()
        return stale
    
    def run(self) -> None:
        """Main worker loop with progressive rendering."""
        while True:
            # Wait for work
            self.mutex.lock()
            while not self.has_work:
                self.condition.wait(self.mutex)
            
            if self.should_stop:
                self.mutex.unlock()
                break
            
            view = self.pending_view
            request_id = self.current_request_id
            self.pending_view = None
            self.has_work = False
            self.mutex.unlock()
            
            if view is None:
                continue
            
            # Get adaptive iteration count
            max_iter = get_adaptive_iterations(view.zoom_level, self.config)
            
            # Check cache for full resolution result
            cached = self.tile_cache.get(
                view.x_min, view.x_max, view.y_min, view.y_max,
                self.config.width, self.config.height, max_iter
            )
            
            if cached is not None:
                # Cache hit - emit immediately
                rgb = iterations_to_rgb_numba(cached, max_iter)
                if not self._is_request_stale(request_id):
                    self.computation_progress.emit(rgb, view, True)
                continue
            
            # Progressive rendering passes
            for i, subsample in enumerate(self.config.progressive_passes):
                if self._is_request_stale(request_id):
                    break
                
                is_final = (subsample == 1)
                
                if subsample > 1:
                    # Compute at reduced resolution
                    iterations = compute_mandelbrot_subsampled(
                        view.x_min, view.x_max,
                        view.y_min, view.y_max,
                        self.config.width, self.config.height,
                        max_iter, subsample
                    )
                    
                    # Upscale to full resolution
                    iterations_full = upscale_nearest(iterations, subsample)
                    # Convert to RGB using colour_selector instead of iterations_to_rgb_numba
                    self.mutex.lock()
                    hue = self.hue_shift
                    sat = self.saturation
                    lit = self.lightness
                    self.mutex.unlock()

                    rgb = colour_selector(iterations_full, max_iter, hue, sat, lit)
                else:
                    # Full resolution computation
                    iterations_full = compute_mandelbrot_numba(
                        view.x_min, view.x_max,
                        view.y_min, view.y_max,
                        self.config.width, self.config.height,
                        max_iter
                    )
                    
                    # Cache the full resolution result
                    self.tile_cache.put(
                        view.x_min, view.x_max, view.y_min, view.y_max,
                        self.config.width, self.config.height, max_iter,
                        iterations_full
                    )
                
                if self._is_request_stale(request_id):
                    break
                
                # Convert to RGB and emit
                self.mutex.lock()
                hue = self.hue_shift
                sat = self.saturation
                lit = self.lightness
                self.mutex.unlock()

                rgb = colour_selector(iterations_full, max_iter, hue, sat, lit)
                
                if not self._is_request_stale(request_id):
                    self.computation_progress.emit(rgb, view, is_final, iterations_full, max_iter)


class MandelbrotWidget(QLabel):
    """
    Widget displaying the Mandelbrot set with scroll-wheel zoom and click-drag pan.
    """
    
    # Signal to notify parent of view changes
    view_changed = pyqtSignal(object, bool)  # (view_state, is_final_render)
    
    def __init__(self, config: 'RenderConfig', parent=None):
        super().__init__(parent)
        
        self.config = config
        self.view = ViewState()
        
        # Mouse tracking state
        self.current_mouse_pos: Optional[QPointF] = None
        self.pan_start_pos: Optional[QPointF] = None
        self.pan_start_view: Optional[ViewState] = None
        self.is_panning = False
        
        # Zoom settings
        self.zoom_factor = 1.5
        
        # Calculate maximum zoom based on float64 precision
        initial_pixel_spacing = self.view.initial_width / self.config.width
        self.max_zoom = initial_pixel_spacing / self.config.min_pixel_spacing
        
        # Set up the widget
        self.setFixedSize(config.width, config.height)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        # Create and start the compute worker
        self.worker = ComputeWorker(config)
        self.worker.computation_progress.connect(self._on_computation_progress)
        self.worker.start()
        
        # Track render state
        self.is_final_render = False
        self.current_rgb_data: Optional[np.ndarray] = None
        # Store current iteration data for recolouring
        self.current_iterations: Optional[np.ndarray] = None
        self.current_max_iter: int = config.max_iterations

        # Display initial loading state
        self._show_loading_state()
        
        # Request initial computation
        self.worker.request_computation(self.view)
    
    def _show_loading_state(self) -> None:
        """Display a placeholder while computing."""
        grey = np.full((self.config.height, self.config.width, 3), 40, dtype=np.uint8)
        self._display_rgb_array(grey)
    
    def update_colours(self, hue: float, saturation: float, lightness: float) -> None:
        """Update colour parameters and recolour current image."""
        self.worker.set_colour_params(hue, saturation, lightness)
        
        # If we have cached iteration data, recolour immediately
        if self.current_iterations is not None:
            rgb = colour_selector(
                self.current_iterations,
                self.current_max_iter,
                hue, saturation, lightness
            )
            self._display_rgb_array(rgb)

    def _display_rgb_array(self, rgb_array: np.ndarray) -> None:
        """Convert numpy RGB array to QPixmap and display."""
        height, width, channels = rgb_array.shape
        bytes_per_line = channels * width
        
        rgb_contiguous = np.ascontiguousarray(rgb_array)
        
        qimage = QImage(
            rgb_contiguous.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Keep reference to prevent garbage collection
        self._current_image_data = rgb_contiguous
        self.current_rgb_data = rgb_contiguous.copy()
        
        self.setPixmap(QPixmap.fromImage(qimage))
    
    def _on_computation_progress(self, rgb_array: np.ndarray,
                                view: ViewState, is_final: bool,
                                iterations: Optional[np.ndarray] = None,
                                max_iter: int = 256) -> None:
        """Handle progressive rendering updates."""
        self._display_rgb_array(rgb_array)
        self.is_final_render = is_final
        if iterations is not None:
            self.current_iterations = iterations.copy()
            self.current_max_iter = max_iter
        self.view_changed.emit(view, is_final)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press to initiate panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.pan_start_pos = event.position()
            self.pan_start_view = self.view.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse movement for tracking and panning."""
        self.current_mouse_pos = event.position()
        
        if self.is_panning and self.pan_start_pos is not None and self.pan_start_view is not None:
            # Calculate pixel displacement
            delta_x = event.position().x() - self.pan_start_pos.x()
            delta_y = event.position().y() - self.pan_start_pos.y()
            
            # Convert pixel displacement to complex plane displacement
            # Note: y-axis is inverted (up in pixels = positive imaginary)
            complex_delta_x = -delta_x * (self.pan_start_view.current_width / self.config.width)
            complex_delta_y = delta_y * (self.pan_start_view.current_height / self.config.height)
            
            # Update view centre
            self.view.centre_real = self.pan_start_view.centre_real + complex_delta_x
            self.view.centre_imag = self.pan_start_view.centre_imag + complex_delta_y
            
            # Request recomputation
            self.worker.request_computation(self.view)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release to end panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.pan_start_pos = None
            self.pan_start_view = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle scroll wheel for zooming."""
        # Don't zoom while panning
        if self.is_panning:
            return
        
        delta = event.angleDelta().y()
        
        if delta == 0:
            return
        
        # Determine zoom direction
        if delta > 0:
            new_zoom = self.view.zoom_level * self.zoom_factor
        else:
            new_zoom = self.view.zoom_level / self.zoom_factor
        
        # Enforce zoom limits
        if new_zoom < 1.0:
            new_zoom = 1.0
        elif new_zoom > self.max_zoom:
            new_zoom = self.max_zoom
            if self.view.zoom_level >= self.max_zoom:
                return
        
        # Get zoom centre point (mouse position)
        pos = event.position()
        zoom_x = pos.x()
        zoom_y = pos.y()
        
        # Convert pixel position to complex coordinates (before zoom)
        complex_x, complex_y = self.view.pixel_to_complex(
            int(zoom_x), int(zoom_y),
            self.config.width, self.config.height
        )
        
        # Calculate relative position of mouse in view (0 to 1)
        rel_x = zoom_x / self.config.width
        rel_y = zoom_y / self.config.height
        
        # Update zoom level
        self.view.zoom_level = new_zoom
        
        # Adjust centre so point under mouse stays fixed
        new_width = self.view.current_width
        new_height = self.view.current_height
        
        self.view.centre_real = complex_x - (rel_x - 0.5) * new_width
        self.view.centre_imag = complex_y + (rel_y - 0.5) * new_height
        
        # Request new computation
        self.worker.request_computation(self.view)
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Return the current RGB image data."""
        return self.current_rgb_data
    
    def get_current_view(self) -> ViewState:
        """Return a copy of the current view state."""
        return self.view.copy()
    
    def cleanup(self) -> None:
        """Stop the worker thread."""
        self.worker.stop()


class ImageSaver:
    """
    Handles saving Mandelbrot images with embedded metadata.
    """
    
    @staticmethod
    def save_image(rgb_data: np.ndarray, view: ViewState, 
                   filepath: Path, embed_text: bool = True) -> bool:
        """
        Save the current view as a PNG image with metadata.
        
        Args:
            rgb_data: RGB image array.
            view: Current view state for metadata.
            filepath: Destination file path.
            embed_text: Whether to embed coordinate text on the image.
            
        Returns:
            True if save was successful, False otherwise.
        """
        try:
            # Create PIL Image from numpy array
            image = Image.fromarray(rgb_data, mode='RGB')
            
            if embed_text:
                image = ImageSaver._add_metadata_overlay(image, view)
            
            # Add PNG metadata
            from PIL import PngImagePlugin
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Mandelbrot_Centre_Real", f"{view.centre_real:.15g}")
            metadata.add_text("Mandelbrot_Centre_Imag", f"{view.centre_imag:.15g}")
            metadata.add_text("Mandelbrot_Zoom", f"{view.zoom_level:.15g}")
            metadata.add_text("Mandelbrot_X_Min", f"{view.x_min:.15g}")
            metadata.add_text("Mandelbrot_X_Max", f"{view.x_max:.15g}")
            metadata.add_text("Mandelbrot_Y_Min", f"{view.y_min:.15g}")
            metadata.add_text("Mandelbrot_Y_Max", f"{view.y_max:.15g}")
            metadata.add_text("Generated", datetime.now().isoformat())
            
            # Save with metadata
            image.save(filepath, "PNG", pnginfo=metadata)
            
            return True
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def _add_metadata_overlay(image: Image.Image, view: ViewState) -> Image.Image:
        """
        Add a semi-transparent overlay with coordinate information.
        
        Args:
            image: PIL Image to modify.
            view: View state containing coordinates.
            
        Returns:
            Modified image with overlay.
        """
        # Create a copy to avoid modifying original
        image = image.copy()
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # Prepare text
        text_lines = [
            f"Centre: {view.centre_real:.10g} + {view.centre_imag:.10g}i",
            f"Zoom: {view.zoom_level:.4e}x"
        ]
        text = "\n".join(text_lines)
        
        # Calculate text size and position
        # Use default font (more reliable across systems)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
        except OSError:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 12)
            except OSError:
                font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position in bottom-left corner with padding
        padding = 10
        x = padding
        y = image.height - text_height - padding - 10
        
        # Draw semi-transparent background
        bg_rect = [
            x - 5,
            y - 5,
            x + text_width + 10,
            y + text_height + 10
        ]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
        
        # Draw text
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        return image
    
    @staticmethod
    def generate_filename(view: ViewState) -> str:
        """
        Generate a descriptive filename based on view parameters.
        
        Args:
            view: Current view state.
            
        Returns:
            Suggested filename string.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zoom_str = f"{view.zoom_level:.2e}".replace("+", "").replace(".", "p")
        return f"mandelbrot_{timestamp}_zoom{zoom_str}.png"


class MandelbrotWindow(QMainWindow):
    """
    Main application window with save functionality.
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Mandelbrot Set Viewer")
        
        # Create render configuration
        self.config = RenderConfig(
            width=800,
            height=600,
            max_iterations=256
        )
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create Mandelbrot display widget
        self.mandelbrot_widget = MandelbrotWidget(self.config)
        layout.addWidget(self.mandelbrot_widget)
        
        # Create status bar for information display
        self.status_label = QLabel()
        self.status_label.setStyleSheet(
            "padding: 8px; "
            "background-color: #2d2d2d; "
            "color: #e0e0e0; "
            "font-family: monospace; "
            "font-size: 11px;"
        )
        self._update_status()
        layout.addWidget(self.status_label)
        
        # Create instructions label
        self.instructions_label = QLabel(
            "Scroll: Zoom  |  Drag: Pan  |  Ctrl+S: Save Image"
        )
        self.instructions_label.setStyleSheet(
            "padding: 5px; "
            "background-color: #1a1a1a; "
            "color: #888888; "
            "font-size: 10px;"
        )
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.instructions_label)
        
        # Connect signals
        self.mandelbrot_widget.view_changed.connect(self._on_view_changed)
        
        # Create colour control sliders
        colour_group = QGroupBox("Colour Controls")
        colour_group.setStyleSheet("""
            QGroupBox {
                color: #e0e0e0;
                border: 1px solid #404040;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        colour_layout = QVBoxLayout(colour_group)

        # Hue slider
        hue_layout = QHBoxLayout()
        hue_label = QLabel("Hue:")
        hue_label.setStyleSheet("color: #e0e0e0; min-width: 70px;")
        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(0, 100)
        self.hue_slider.setValue(0)
        self.hue_slider.setStyleSheet(self._slider_style())
        self.hue_value_label = QLabel("0%")
        self.hue_value_label.setStyleSheet("color: #e0e0e0; min-width: 40px;")
        hue_layout.addWidget(hue_label)
        hue_layout.addWidget(self.hue_slider)
        hue_layout.addWidget(self.hue_value_label)
        colour_layout.addLayout(hue_layout)

        # Saturation slider
        sat_layout = QHBoxLayout()
        sat_label = QLabel("Saturation:")
        sat_label.setStyleSheet("color: #e0e0e0; min-width: 70px;")
        self.sat_slider = QSlider(Qt.Orientation.Horizontal)
        self.sat_slider.setRange(0, 100)
        self.sat_slider.setValue(100)
        self.sat_slider.setStyleSheet(self._slider_style())
        self.sat_value_label = QLabel("100%")
        self.sat_value_label.setStyleSheet("color: #e0e0e0; min-width: 40px;")
        sat_layout.addWidget(sat_label)
        sat_layout.addWidget(self.sat_slider)
        sat_layout.addWidget(self.sat_value_label)
        colour_layout.addLayout(sat_layout)

        # Lightness slider
        lit_layout = QHBoxLayout()
        lit_label = QLabel("Lightness:")
        lit_label.setStyleSheet("color: #e0e0e0; min-width: 70px;")
        self.lit_slider = QSlider(Qt.Orientation.Horizontal)
        self.lit_slider.setRange(0, 100)
        self.lit_slider.setValue(50)
        self.lit_slider.setStyleSheet(self._slider_style())
        self.lit_value_label = QLabel("50%")
        self.lit_value_label.setStyleSheet("color: #e0e0e0; min-width: 40px;")
        lit_layout.addWidget(lit_label)
        lit_layout.addWidget(self.lit_slider)
        lit_layout.addWidget(self.lit_value_label)
        colour_layout.addLayout(lit_layout)

        layout.addWidget(colour_group)

        # Connect slider signals
        self.hue_slider.valueChanged.connect(self._on_colour_changed)
        self.sat_slider.valueChanged.connect(self._on_colour_changed)
        self.lit_slider.valueChanged.connect(self._on_colour_changed)

        # Size window to fit contents
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())
    
    def _slider_style(self) -> str:
        """Return stylesheet for sliders."""
        return """
            QSlider::groove:horizontal {
                border: 1px solid #404040;
                height: 8px;
                background: #2d2d2d;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #606060;
                border: 1px solid #808080;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #707070;
            }
            QSlider::sub-page:horizontal {
                background: #4a90d9;
                border-radius: 4px;
            }
        """

    def _on_colour_changed(self) -> None:
        """Handle colour slider changes."""
        hue = self.hue_slider.value() / 100.0
        saturation = self.sat_slider.value() / 100.0
        lightness = self.lit_slider.value() / 100.0
        
        # Update labels
        self.hue_value_label.setText(f"{self.hue_slider.value()}%")
        self.sat_value_label.setText(f"{self.sat_slider.value()}%")
        self.lit_value_label.setText(f"{self.lit_slider.value()}%")
        
        # Update colours
        self.mandelbrot_widget.update_colours(hue, saturation, lightness)

    def _update_status(self) -> None:
        """Update the status label with current view information."""
        view = self.mandelbrot_widget.view
        
        # Check if at precision limit
        at_limit = view.zoom_level >= self.mandelbrot_widget.max_zoom * 0.99
        limit_warning = "  ⚠ PRECISION LIMIT" if at_limit else ""
        
        # Render quality indicator
        quality = "●" if self.mandelbrot_widget.is_final_render else "○"
        
        self.status_label.setText(
            f"{quality} Centre: ({view.centre_real:.10g}, {view.centre_imag:.10g})  │  "
            f"Zoom: {view.zoom_level:.4e}x{limit_warning}"
        )
    
    def _on_view_changed(self, view: ViewState, is_final: bool) -> None:
        """Called when the view has been updated."""
        self._update_status()
    
    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        # Ctrl+S to save
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_S:
                self._save_image()
                return
        
        super().keyPressEvent(event)
    
    def _save_image(self) -> None:
        """Open save dialog and save current view."""
        # Get current image data
        rgb_data = self.mandelbrot_widget.get_current_image()
        
        if rgb_data is None:
            QMessageBox.warning(
                self,
                "Cannot Save",
                "No image data available to save."
            )
            return
        
        # Generate suggested filename
        view = self.mandelbrot_widget.get_current_view()
        suggested_name = ImageSaver.generate_filename(view)
        
        # Open save dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mandelbrot Image",
            suggested_name,
            "PNG Images (*.png);;All Files (*)"
        )
        
        if not filepath:
            return  # User cancelled
        
        # Ensure .png extension
        if not filepath.lower().endswith('.png'):
            filepath += '.png'
        
        # Save the image
        success = ImageSaver.save_image(
            rgb_data,
            view,
            Path(filepath),
            embed_text=True
        )
        
        if success:
            QMessageBox.information(
                self,
                "Image Saved",
                f"Image saved successfully to:\n{filepath}\n\n"
                f"Coordinates embedded in image metadata."
            )
        else:
            QMessageBox.critical(
                self,
                "Save Failed",
                "Failed to save the image. Please try again."
            )
    
    def closeEvent(self, event) -> None:
        """Clean up worker thread on close."""
        self.mandelbrot_widget.cleanup()
        super().closeEvent(event)


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Apply dark theme
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1a1a1a;
        }
        QMessageBox {
            background-color: #2d2d2d;
            color: #e0e0e0;
        }
        QMessageBox QLabel {
            color: #e0e0e0;
        }
        QMessageBox QPushButton {
            background-color: #404040;
            color: #e0e0e0;
            border: 1px solid #555555;
            padding: 5px 15px;
            min-width: 60px;
        }
        QMessageBox QPushButton:hover {
            background-color: #505050;
        }
        QFileDialog {
            background-color: #2d2d2d;
            color: #e0e0e0;
        }
    """)
    
    window = MandelbrotWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()