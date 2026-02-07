"""
Interactive Mandelbrot Set Viewer

A PyQt6-based interactive viewer with threaded computation, scroll-wheel zooming
centred on mouse position, and precision limit detection.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from datetime import datetime
import os

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMutex, QWaitCondition
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent, QPainter, QPen, QColor

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
    zoom_level: float = 1.0  # 1.0 = initial view showing full set
    
    # Initial view bounds (at zoom_level = 1.0)
    initial_width: float = 3.5   # Real axis span
    initial_height: float = 2.5  # Imaginary axis span
    
    @property
    def current_width(self) -> float:
        """Current width in complex plane units."""
        return self.initial_width / self.zoom_level
    
    @property
    def current_height(self) -> float:
        """Current height in complex plane units."""
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


class ComputeWorker(QThread):
    """
    Enhanced worker thread with progressive rendering and caching.
    
    Emits multiple signals during computation to provide quick previews
    followed by full resolution results.
    """
    
    # Signal for each progressive pass (rgb_array, view_state, is_final)
    computation_progress = pyqtSignal(np.ndarray, object, bool)
    
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
        
        # Warm up Numba JIT compilation on first run
        self._warmup_jit()
    
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
                rgb = iterations_to_rgb_numba(iterations_full, max_iter)
                
                if not self._is_request_stale(request_id):
                    self.computation_progress.emit(rgb, view, is_final)


class MandelbrotWidget(QLabel):
    """
    Enhanced widget with pan support and visual feedback.
    
    Implements click-drag panning and displays pan state visually.
    """
    
    # Signal emitted when view changes (for status updates)
    view_changed = pyqtSignal(object)  # ViewState
    
    def __init__(self, config: RenderConfig, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.view = ViewState()
        self.current_mouse_pos = None
        
        # Zoom settings
        self.zoom_factor = 1.5
        
        # Pan state
        self.is_panning = False
        self.pan_start_pos: Optional[QPointF] = None
        self.pan_start_view: Optional[ViewState] = None
        
        # Calculate maximum zoom
        initial_pixel_spacing = self.view.initial_width / self.config.width
        self.max_zoom = initial_pixel_spacing / self.config.min_pixel_spacing
        
        self.setFixedSize(config.width, config.height)
        self.setMouseTracking(True)
        
        # Create enhanced worker
        self.worker = ComputeWorker(config)
        # FIXED: Connect to the correct signal name
        self.worker.computation_progress.connect(self._on_computation_progress)
        self.worker.start()
        
        self._show_loading_state()
        self.worker.request_computation(self.view)
        
        # Track render quality
        self.is_final_render = False
        
        # Store current display for saving
        self.current_rgb_array: Optional[np.ndarray] = None
    
    def _show_loading_state(self) -> None:
        """Display placeholder while computing."""
        grey = np.full((self.config.height, self.config.width, 3), 40, dtype=np.uint8)
        self._display_rgb_array(grey)
    
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
        
        pixmap = QPixmap.fromImage(qimage)
        
        # If panning, draw overlay to show pan state
        if self.is_panning and self.pan_start_pos and self.current_mouse_pos:
            pixmap = self._draw_pan_overlay(pixmap)
        
        self.setPixmap(pixmap)
    
    def _draw_pan_overlay(self, pixmap: QPixmap) -> QPixmap:
        """Draw visual feedback during panning."""
        # Create a copy to draw on
        overlay_pixmap = QPixmap(pixmap)
        painter = QPainter(overlay_pixmap)
        
        # Semi-transparent overlay
        painter.setOpacity(0.3)
        painter.fillRect(overlay_pixmap.rect(), QColor(100, 150, 200))
        
        # Draw arrow from start to current position
        painter.setOpacity(1.0)
        pen = QPen(QColor(255, 255, 255), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        start_x = int(self.pan_start_pos.x())
        start_y = int(self.pan_start_pos.y())
        curr_x = int(self.current_mouse_pos.x())
        curr_y = int(self.current_mouse_pos.y())
        
        painter.drawLine(start_x, start_y, curr_x, curr_y)
        
        # Draw circles at start and end
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(start_x - 5, start_y - 5, 10, 10)
        painter.setBrush(QColor(200, 200, 255))
        painter.drawEllipse(curr_x - 5, curr_y - 5, 10, 10)
        
        painter.end()
        return overlay_pixmap
    
    def _on_computation_progress(self, rgb_array: np.ndarray, 
                                  view: ViewState, is_final: bool) -> None:
        """Handle progressive rendering updates."""
        self.current_rgb_array = rgb_array.copy()
        self._display_rgb_array(rgb_array)
        self.is_final_render = is_final
        
        # Emit view changed signal for status updates
        self.view_changed.emit(view)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for pan initiation."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Start panning
            self.is_panning = True
            self.pan_start_pos = event.position()
            self.pan_start_view = self.view.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track mouse position and handle panning."""
        self.current_mouse_pos = event.position()
        
        if self.is_panning and self.pan_start_pos and self.pan_start_view:
            # Calculate pan offset in pixels
            dx_pixels = event.position().x() - self.pan_start_pos.x()
            dy_pixels = event.position().y() - self.pan_start_pos.y()
            
            # Convert pixel offset to complex plane offset
            pixel_to_complex_x = self.pan_start_view.current_width / self.config.width
            pixel_to_complex_y = self.pan_start_view.current_height / self.config.height
            
            dx_complex = -dx_pixels * pixel_to_complex_x  # Negative for natural pan direction
            dy_complex = dy_pixels * pixel_to_complex_y   # Positive because y increases downward
            
            # Update view center
            self.view.centre_real = self.pan_start_view.centre_real + dx_complex
            self.view.centre_imag = self.pan_start_view.centre_imag + dy_complex
            
            # Visual feedback - redraw with overlay
            if self.current_rgb_array is not None:
                self._display_rgb_array(self.current_rgb_array)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release to complete pan."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None
            self.pan_start_view = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
            # Trigger recomputation at the new position
            self.worker.request_computation(self.view)
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle scroll wheel zoom."""
        delta = event.angleDelta().y()
        
        if delta == 0:
            return
        
        if delta > 0:
            new_zoom = self.view.zoom_level * self.zoom_factor
        else:
            new_zoom = self.view.zoom_level / self.zoom_factor
        
        if new_zoom < 1.0:
            new_zoom = 1.0
        elif new_zoom > self.max_zoom:
            new_zoom = self.max_zoom
            if self.view.zoom_level >= self.max_zoom:
                return
        
        pos = event.position()
        zoom_x = pos.x()
        zoom_y = pos.y()
        
        complex_x, complex_y = self.view.pixel_to_complex(
            int(zoom_x), int(zoom_y),
            self.config.width, self.config.height
        )
        
        rel_x = zoom_x / self.config.width
        rel_y = zoom_y / self.config.height
        
        self.view.zoom_level = new_zoom
        
        new_width = self.view.current_width
        new_height = self.view.current_height
        
        self.view.centre_real = complex_x - (rel_x - 0.5) * new_width
        self.view.centre_imag = complex_y + (rel_y - 0.5) * new_height
        
        self.worker.request_computation(self.view)
    
    def save_current_view(self, filepath: str) -> bool:
        """
        Save the current view to an image file with metadata.
        
        Args:
            filepath: Path to save the image.
            
        Returns:
            True if save was successful, False otherwise.
        """
        if self.current_rgb_array is None:
            return False
        
        try:
            from PIL import Image
            from PIL.PngImagePlugin import PngInfo
            
            # Create PIL Image from numpy array
            img = Image.fromarray(self.current_rgb_array, mode='RGB')
            
            # Create metadata
            metadata = PngInfo()
            metadata.add_text("Mandelbrot-Centre-Real", f"{self.view.centre_real:.17g}")
            metadata.add_text("Mandelbrot-Centre-Imag", f"{self.view.centre_imag:.17g}")
            metadata.add_text("Mandelbrot-Zoom", f"{self.view.zoom_level:.17g}")
            metadata.add_text("Mandelbrot-Bounds", 
                            f"{self.view.x_min:.17g},{self.view.x_max:.17g},"
                            f"{self.view.y_min:.17g},{self.view.y_max:.17g}")
            metadata.add_text("Mandelbrot-Resolution", 
                            f"{self.config.width}x{self.config.height}")
            metadata.add_text("Mandelbrot-MaxIterations", 
                            f"{get_adaptive_iterations(self.view.zoom_level, self.config)}")
            metadata.add_text("Mandelbrot-Timestamp", 
                            datetime.now().isoformat())
            
            # Save with metadata
            img.save(filepath, "PNG", pnginfo=metadata)
            return True
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def cleanup(self) -> None:
        """Stop worker thread."""
        self.worker.stop()


class MandelbrotWindow(QMainWindow):
    """
    Enhanced main window with save functionality and improved status display.
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Mandelbrot Set Viewer")
        
        # Create render configuration
        self.config = RenderConfig(
            width=800,
            height=600,
            base_iterations=256,
            progressive_passes=(8, 4, 2, 1),
            max_cached_tiles=256
        )
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create Mandelbrot display widget
        self.mandelbrot_widget = MandelbrotWidget(self.config)
        layout.addWidget(self.mandelbrot_widget)
        
        # Create status label with improved styling
        self.status_label = QLabel()
        self.status_label.setStyleSheet(
            "padding: 8px; "
            "background-color: #2b2b2b; "
            "color: #e0e0e0; "
            "font-family: 'Courier New', monospace; "
            "font-size: 10pt;"
        )
        self._update_status()
        layout.addWidget(self.status_label)
        
        # Create help label
        self.help_label = QLabel()
        self.help_label.setStyleSheet(
            "padding: 6px; "
            "background-color: #1a1a1a; "
            "color: #a0a0a0; "
            "font-size: 9pt;"
        )
        self.help_label.setText(
            "🖱️ Scroll: Zoom  |  🖱️ Click+Drag: Pan  |  Ctrl+S: Save Image"
        )
        layout.addWidget(self.help_label)
        
        # Connect to track view changes
        self.mandelbrot_widget.view_changed.connect(self._on_view_changed)
        
        # Set up keyboard shortcuts
        self._setup_shortcuts()
        
        # Size window to fit contents
        self.setFixedSize(self.sizeHint())
    
    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        from PyQt6.QtGui import QKeySequence, QShortcut
        
        # Ctrl+S to save
        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self._save_image)
        
        # Ctrl+Shift+S to quick save
        quick_save_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        quick_save_shortcut.activated.connect(self._quick_save_image)
    
    def _update_status(self) -> None:
        """Update the status label with current view information."""
        view = self.mandelbrot_widget.view
        
        # Check if at precision limit
        at_limit = view.zoom_level >= self.mandelbrot_widget.max_zoom * 0.99
        limit_warning = " ⚠️ PRECISION LIMIT" if at_limit else ""
        
        # Get adaptive iteration count
        max_iter = get_adaptive_iterations(view.zoom_level, self.config)
        
        # Render quality indicator
        quality = "FINAL" if self.mandelbrot_widget.is_final_render else "PREVIEW"
        quality_color = "#00ff00" if self.mandelbrot_widget.is_final_render else "#ffaa00"
        
        self.status_label.setText(
            f"<span style='color: {quality_color};'>●</span> {quality}  |  "
            f"Centre: <b>({view.centre_real:.10g}, {view.centre_imag:.10g})</b>  |  "
            f"Zoom: <b>{view.zoom_level:.2e}x</b>  |  "
            f"Iterations: <b>{max_iter}</b>"
            f"<span style='color: #ff4444;'>{limit_warning}</span>"
        )
    
    def _on_view_changed(self, view: ViewState) -> None:
        """Called when view has been updated."""
        self._update_status()
    
    def _save_image(self) -> None:
        """Show save dialog and save current view."""
        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zoom_str = f"zoom{self.mandelbrot_widget.view.zoom_level:.1e}".replace('+', '')
        default_name = f"mandelbrot_{timestamp}_{zoom_str}.png"
        
        # Show file dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mandelbrot Image",
            default_name,
            "PNG Images (*.png);;All Files (*)"
        )
        
        if filepath:
            # Ensure .png extension
            if not filepath.lower().endswith('.png'):
                filepath += '.png'
            
            # Save the image
            success = self.mandelbrot_widget.save_current_view(filepath)
            
            if success:
                QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Image saved to:\n{filepath}\n\n"
                    f"Metadata includes coordinates and zoom level."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Save Failed",
                    "Failed to save image. Please try again."
                )
    
    def _quick_save_image(self) -> None:
        """Quick save without dialog to default location."""
        # Create saves directory if it doesn't exist
        save_dir = "mandelbrot_saves"
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zoom_str = f"zoom{self.mandelbrot_widget.view.zoom_level:.1e}".replace('+', '')
        filename = f"mandelbrot_{timestamp}_{zoom_str}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Save
        success = self.mandelbrot_widget.save_current_view(filepath)
        
        if success:
            # Show brief notification in status
            original_text = self.status_label.text()
            self.status_label.setText(
                f"<span style='color: #00ff00;'>✓ Saved to {filepath}</span>"
            )
            # Restore original status after 3 seconds
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(3000, lambda: self.status_label.setText(original_text))
    
    def closeEvent(self, event) -> None:
        """Clean up worker thread on close."""
        self.mandelbrot_widget.cleanup()
        super().closeEvent(event)


def main():
    """Application entry point."""
    import sys
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)
    
    window = MandelbrotWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()