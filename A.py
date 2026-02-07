"""
Interactive Mandelbrot Set Viewer

A PyQt6-based interactive viewer with threaded computation, scroll-wheel zooming
centred on mouse position, and precision limit detection.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
import hashlib

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent
import math
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMutex, QWaitCondition
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent


try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Install with 'pip install numba' for better performance.")
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range



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
        real = self.x_min + (pixel_x / image_width) * self.current_width
        imag = self.y_max - (pixel_y / image_height) * self.current_height
        return real, imag
    
    def copy(self) -> 'ViewState':
        return ViewState(
            centre_real=self.centre_real,
            centre_imag=self.centre_imag,
            zoom_level=self.zoom_level,
            initial_width=self.initial_width,
            initial_height=self.initial_height
        )
    
    def get_cache_key(self, width: int, height: int, max_iter: int) -> str:
        """Generate a unique key for caching this view."""
        # Round to avoid floating point precision issues in cache lookup
        key_str = f"{self.centre_real:.15e}_{self.centre_imag:.15e}_{self.zoom_level:.15e}_{width}_{height}_{max_iter}"
        return hashlib.md5(key_str.encode()).hexdigest()


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    width: int = 800
    height: int = 600
    max_iterations: int = 256
    min_pixel_spacing: float = field(default_factory=lambda: np.finfo(np.float64).eps * 1000)
    
    # Progressive rendering settings
    enable_progressive: bool = True
    preview_scale: int = 4  # Render at 1/4 resolution first
    
    # Cache settings
    cache_size: int = 10  # Number of views to cache
    
    # Adaptive iteration depth
    adaptive_iterations: bool = True
    base_iterations: int = 128
    iteration_zoom_factor: float = 1.5  # Increase iterations by this factor per 10x zoom
    max_adaptive_iterations: int = 4096
    
    def get_iterations_for_zoom(self, zoom_level: float) -> int:
        """Calculate appropriate iteration count based on zoom level."""
        if not self.adaptive_iterations:
            return self.max_iterations
        
        # Logarithmic scaling: more detail needed at higher zoom
        zoom_exponent = math.log10(max(1.0, zoom_level))
        iterations = int(self.base_iterations * (self.iteration_zoom_factor ** zoom_exponent))
        
        return min(iterations, self.max_adaptive_iterations)


class ViewCache:
    """LRU cache for computed Mandelbrot views."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached view, moving it to end (most recently used)."""
        if key in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        return None
    
    def put(self, key: str, value: np.ndarray) -> None:
        """Store view in cache, evicting oldest if necessary."""
        if key in self.cache:
            # Update existing entry
            self.cache.move_to_end(key)
            self.cache[key] = value.copy()
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Remove oldest (first) item
                self.cache.popitem(last=False)
            self.cache[key] = value.copy()
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()

# Numba-optimized computation kernel
@jit(nopython=True, parallel=True, cache=True)
def _mandelbrot_kernel_numba(real_vals, imag_vals, max_iterations):
    """
    Highly optimized Mandelbrot computation using Numba JIT compilation.
    
    This function is compiled to native machine code with parallel execution
    across CPU cores.
    """
    height = len(imag_vals)
    width = len(real_vals)
    iterations = np.zeros((height, width), dtype=np.float64)
    
    # Parallel loop over rows
    for py in prange(height):
        c_imag = imag_vals[py]
        
        for px in range(width):
            c_real = real_vals[px]
            
            # Initialize z
            z_real = 0.0
            z_imag = 0.0
            
            iteration = 0
            
            # Iteration loop
            while iteration < max_iterations:
                z_real_sq = z_real * z_real
                z_imag_sq = z_imag * z_imag
                
                # Check escape condition
                if z_real_sq + z_imag_sq > 4.0:
                    # Smooth coloring
                    magnitude = math.sqrt(z_real_sq + z_imag_sq)
                    if magnitude > 1.0:
                        log_zn = math.log(magnitude)
                        if log_zn > 0:
                            smooth_val = iteration + 1 - math.log(log_zn) / math.log(2.0)
                            iterations[py, px] = smooth_val
                            break
                    iterations[py, px] = float(iteration)
                    break
                
                # z = z^2 + c
                new_real = z_real_sq - z_imag_sq + c_real
                new_imag = 2.0 * z_real * z_imag + c_imag
                
                z_real = new_real
                z_imag = new_imag
                iteration += 1
            
            # Didn't escape
            if iteration == max_iterations:
                iterations[py, px] = float(max_iterations)
    
    return iterations

def compute_mandelbrot_optimized(view: ViewState, config: RenderConfig, 
                                  width: int = None, height: int = None) -> np.ndarray:
    """
    Optimized Mandelbrot computation with adaptive parameters.
    
    Uses Numba JIT compilation for native performance with parallel execution.
    Supports variable resolution for progressive rendering.
    
    Args:
        view: Current view state.
        config: Rendering configuration.
        width: Optional width override for progressive rendering.
        height: Optional height override for progressive rendering.
        
    Returns:
        2D array of normalized iteration counts.
    """
    if width is None:
        width = config.width
    if height is None:
        height = config.height
    
    # Get adaptive iteration count based on zoom level
    max_iterations = config.get_iterations_for_zoom(view.zoom_level)
    
    # Create coordinate arrays
    real_vals = np.linspace(view.x_min, view.x_max, width, dtype=np.float64)
    imag_vals = np.linspace(view.y_max, view.y_min, height, dtype=np.float64)
    
    if NUMBA_AVAILABLE:
        # Use highly optimized Numba kernel
        iterations = _mandelbrot_kernel_numba(real_vals, imag_vals, max_iterations)
    else:
        # Fallback to vectorized NumPy (slower but still functional)
        iterations = _compute_mandelbrot_numpy_fallback(real_vals, imag_vals, max_iterations)
    
    return iterations

def _compute_mandelbrot_numpy_fallback(real_vals, imag_vals, max_iterations):
    """Fallback NumPy implementation when Numba is not available."""
    height = len(imag_vals)
    width = len(real_vals)
    
    real_grid, imag_grid = np.meshgrid(real_vals, imag_vals)
    
    c_real = real_grid.copy()
    c_imag = imag_grid.copy()
    
    z_real = np.zeros_like(c_real)
    z_imag = np.zeros_like(c_imag)
    
    iterations = np.zeros((height, width), dtype=np.float64)
    not_escaped = np.ones((height, width), dtype=bool)
    
    for i in range(max_iterations):
        z_real_squared = z_real[not_escaped] ** 2
        z_imag_squared = z_imag[not_escaped] ** 2
        
        new_real = z_real_squared - z_imag_squared + c_real[not_escaped]
        new_imag = 2.0 * z_real[not_escaped] * z_imag[not_escaped] + c_imag[not_escaped]
        
        z_real[not_escaped] = new_real
        z_imag[not_escaped] = new_imag
        
        magnitude_squared = z_real ** 2 + z_imag ** 2
        escaped_this_iteration = not_escaped & (magnitude_squared > 4.0)
        
        if np.any(escaped_this_iteration):
            escaped_magnitude = np.sqrt(magnitude_squared[escaped_this_iteration])
            log_zn = np.log(escaped_magnitude)
            smooth_val = i + 1 - np.log(log_zn) / np.log(2.0)
            iterations[escaped_this_iteration] = smooth_val
        
        not_escaped[escaped_this_iteration] = False
        
        if not np.any(not_escaped):
            break
    
    iterations[not_escaped] = max_iterations
    
    return iterations

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


def resize_iterations(iterations: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize iteration data using bilinear interpolation.
    
    Used to upscale low-resolution previews to full resolution for display.
    """
    from scipy import ndimage
    
    current_height, current_width = iterations.shape
    
    if current_height == target_height and current_width == target_width:
        return iterations
    
    # Calculate zoom factors
    zoom_y = target_height / current_height
    zoom_x = target_width / current_width
    
    # Use scipy's zoom for high-quality interpolation
    # order=1 is bilinear, order=3 is cubic (slower but smoother)
    resized = ndimage.zoom(iterations, (zoom_y, zoom_x), order=1)
    
    return resized


def iterations_to_rgb(iterations: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Convert iteration counts to RGB image array.
    
    Uses sinusoidal colour mapping for smooth gradients.
    """
    height, width = iterations.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    in_set = iterations >= max_iterations - 1
    
    normalised = np.zeros_like(iterations)
    if np.any(~in_set):
        max_iter_external = np.max(iterations[~in_set])
        if max_iter_external > 0:
            normalised = iterations / max_iter_external
    
    t = normalised * 3.0 * np.pi
    
    rgb[:, :, 0] = (127.5 * (1 + np.sin(t))).astype(np.uint8)
    rgb[:, :, 1] = (127.5 * (1 + np.sin(t + 2.094))).astype(np.uint8)
    rgb[:, :, 2] = (127.5 * (1 + np.sin(t + 4.189))).astype(np.uint8)
    
    rgb[in_set] = [0, 0, 0]
    
    return rgb


class ComputeWorker(QThread):
    """
    Optimized worker thread with progressive rendering and caching.
    """
    
    # Signal for preview (low-res) result
    preview_ready = pyqtSignal(np.ndarray, object)  # (rgb_array, view_state)
    
    # Signal for final (full-res) result
    computation_complete = pyqtSignal(np.ndarray, object)  # (rgb_array, view_state)
    
    def __init__(self, config: RenderConfig):
        super().__init__()
        self.config = config
        
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        self.pending_view: Optional[ViewState] = None
        self.should_stop = False
        self.has_work = False
        
        # Initialize cache
        self.cache = ViewCache(max_size=config.cache_size)
        
        # Track current computation to allow cancellation
        self.current_view: Optional[ViewState] = None
    
    def request_computation(self, view: ViewState) -> None:
        """Request computation for a new view."""
        self.mutex.lock()
        self.pending_view = view.copy()
        self.has_work = True
        self.condition.wakeOne()
        self.mutex.unlock()
    
    def stop(self) -> None:
        """Signal the worker to stop."""
        self.mutex.lock()
        self.should_stop = True
        self.has_work = True
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()
    
    def _check_cancelled(self) -> bool:
        """Check if current computation has been superseded."""
        self.mutex.lock()
        cancelled = self.has_work  # New work means current is cancelled
        self.mutex.unlock()
        return cancelled
    
    def run(self) -> None:
        """Main worker loop with progressive rendering and caching."""
        while True:
            self.mutex.lock()
            while not self.has_work:
                self.condition.wait(self.mutex)
            
            if self.should_stop:
                self.mutex.unlock()
                break
            
            view = self.pending_view
            self.pending_view = None
            self.has_work = False
            self.current_view = view
            self.mutex.unlock()
            
            if view is None:
                continue
            
            # Check cache first
            max_iter = self.config.get_iterations_for_zoom(view.zoom_level)
            cache_key = view.get_cache_key(self.config.width, self.config.height, max_iter)
            
            cached_iterations = self.cache.get(cache_key)
            if cached_iterations is not None:
                # Cache hit! Convert and emit immediately
                rgb_array = iterations_to_rgb(cached_iterations, max_iter)
                
                if not self._check_cancelled():
                    self.computation_complete.emit(rgb_array, view)
                continue
            
            # Progressive rendering: compute low-res preview first
            if self.config.enable_progressive:
                preview_width = self.config.width // self.config.preview_scale
                preview_height = self.config.height // self.config.preview_scale
                
                # Compute preview
                preview_iterations = compute_mandelbrot_optimized(
                    view, self.config, preview_width, preview_height
                )
                
                # Check if cancelled
                if self._check_cancelled():
                    continue
                
                # Upscale preview to full resolution for display
                preview_upscaled = resize_iterations(
                    preview_iterations, self.config.width, self.config.height
                )
                
                preview_rgb = iterations_to_rgb(preview_upscaled, max_iter)
                
                # Emit preview
                if not self._check_cancelled():
                    self.preview_ready.emit(preview_rgb, view)
            
            # Compute full resolution
            full_iterations = compute_mandelbrot_optimized(view, self.config)
            
            # Check if cancelled after expensive computation
            if self._check_cancelled():
                continue
            
            # Store in cache
            self.cache.put(cache_key, full_iterations)
            
            # Convert to RGB
            rgb_array = iterations_to_rgb(full_iterations, max_iter)
            
            # Final check before emitting
            if not self._check_cancelled():
                self.computation_complete.emit(rgb_array, view)


class MandelbrotWidget(QLabel):
    """
    Optimized widget with progressive rendering support.
    """
    
    def __init__(self, config: RenderConfig, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.view = ViewState()
        self.current_mouse_pos: Optional[QPointF] = None
        
        self.zoom_factor = 1.5
        
        initial_pixel_spacing = self.view.initial_width / self.config.width
        self.max_zoom = initial_pixel_spacing / self.config.min_pixel_spacing
        
        self.setFixedSize(config.width, config.height)
        self.setMouseTracking(True)
        
        # Create and start the optimized compute worker
        self.worker = ComputeWorker(config)
        self.worker.preview_ready.connect(self._on_preview_ready)
        self.worker.computation_complete.connect(self._on_computation_complete)
        self.worker.start()
        
        self._show_loading_state()
        
        self.worker.request_computation(self.view)
    
    def _show_loading_state(self) -> None:
        """Display a placeholder while computing."""
        grey = np.full((self.config.height, self.config.width, 3), 128, dtype=np.uint8)
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
        
        self._current_image_data = rgb_contiguous
        
        self.setPixmap(QPixmap.fromImage(qimage))
    
    def _on_preview_ready(self, rgb_array: np.ndarray, view: ViewState) -> None:
        """Handle low-res preview from worker thread."""
        # Display preview immediately for responsive feedback
        self._display_rgb_array(rgb_array)
    
    def _on_computation_complete(self, rgb_array: np.ndarray, view: ViewState) -> None:
        """Handle completed full-resolution computation."""
        self._display_rgb_array(rgb_array)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track mouse position for zoom centring."""
        self.current_mouse_pos = event.position()
        super().mouseMoveEvent(event)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle scroll wheel for zooming."""
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
    
    def cleanup(self) -> None:
        """Stop the worker thread."""
        self.worker.stop()


class MandelbrotWindow(QMainWindow):
    """Main application window."""
    
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
        
        # Create Mandelbrot display widget
        self.mandelbrot_widget = MandelbrotWidget(self.config)
        layout.addWidget(self.mandelbrot_widget)
        
        # Create status label for zoom information
        self.status_label = QLabel()
        self.status_label.setStyleSheet("padding: 5px; background-color: #333; color: white;")
        self._update_status()
        layout.addWidget(self.status_label)
        
        # Connect to track zoom changes
        self.mandelbrot_widget.worker.computation_complete.connect(self._on_view_updated)
        
        # Size window to fit contents
        self.setFixedSize(self.sizeHint())
    
    def _update_status(self) -> None:
        """Update the status label with current view information."""
        view = self.mandelbrot_widget.view
        
        # Check if at precision limit
        at_limit = view.zoom_level >= self.mandelbrot_widget.max_zoom * 0.99
        limit_warning = " [PRECISION LIMIT]" if at_limit else ""
        
        self.status_label.setText(
            f"Centre: ({view.centre_real:.10g}, {view.centre_imag:.10g})  |  "
            f"Zoom: {view.zoom_level:.2e}x{limit_warning}"
        )
    
    def _on_view_updated(self, rgb_array: np.ndarray, view: ViewState) -> None:
        """Called when a new view has been computed."""
        self._update_status()
    
    def closeEvent(self, event) -> None:
        """Clean up worker thread on close."""
        self.mandelbrot_widget.cleanup()
        super().closeEvent(event)


def main():
    """Application entry point."""
    import sys
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MandelbrotWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()