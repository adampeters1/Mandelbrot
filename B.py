"""
Interactive Mandelbrot Set Viewer

A PyQt6-based interactive viewer with threaded computation, scroll-wheel zooming
centred on mouse position, and precision limit detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMutex, QWaitCondition
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent


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
    """Configuration for rendering."""
    width: int = 800
    height: int = 600
    max_iterations: int = 256
    
    # Precision limit for float64
    # When pixel spacing drops below this, we've hit precision limits
    min_pixel_spacing: float = field(default_factory=lambda: np.finfo(np.float64).eps * 1000)


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
    Worker thread for Mandelbrot computation.
    
    Runs continuously, waiting for computation requests and emitting results
    without blocking the main GUI thread.
    """
    
    # Signal emitted when computation completes
    computation_complete = pyqtSignal(np.ndarray, object)  # (rgb_array, view_state)
    
    def __init__(self, config: RenderConfig):
        super().__init__()
        self.config = config
        
        # Thread synchronisation
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # Current request
        self.pending_view: Optional[ViewState] = None
        self.should_stop = False
        self.has_work = False
    
    def request_computation(self, view: ViewState) -> None:
        """
        Request computation for a new view.
        
        If a computation is already in progress, this will queue the new
        request to be processed next, effectively cancelling the old one.
        """
        self.mutex.lock()
        self.pending_view = view.copy()
        self.has_work = True
        self.condition.wakeOne()
        self.mutex.unlock()
    
    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self.mutex.lock()
        self.should_stop = True
        self.has_work = True
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()
    
    def run(self) -> None:
        """Main worker loop."""
        while True:
            # Wait for work
            self.mutex.lock()
            while not self.has_work:
                self.condition.wait(self.mutex)
            
            if self.should_stop:
                self.mutex.unlock()
                break
            
            # Grab the pending view and clear the flag
            view = self.pending_view
            self.pending_view = None
            self.has_work = False
            self.mutex.unlock()
            
            if view is None:
                continue
            
            # Perform computation
            iterations = compute_mandelbrot_vectorised(view, self.config)
            rgb_array = iterations_to_rgb(iterations, self.config.max_iterations)
            
            # Check if a new request came in while we were computing
            self.mutex.lock()
            if self.has_work:
                # New request pending, discard this result
                self.mutex.unlock()
                continue
            self.mutex.unlock()
            
            # Emit result
            self.computation_complete.emit(rgb_array, view)


class MandelbrotWidget(QLabel):
    """
    Widget displaying the Mandelbrot set with scroll-wheel zoom.
    
    Handles mouse tracking and wheel events, triggering recomputation
    via the worker thread.
    """
    
    def __init__(self, config: RenderConfig, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.view = ViewState()
        self.current_mouse_pos: Optional[QPointF] = None
        
        # Zoom settings
        self.zoom_factor = 1.5  # How much to zoom per scroll step
        
        # Calculate maximum zoom based on float64 precision
        # When pixel spacing < minimum representable difference, we've hit the limit
        initial_pixel_spacing = self.view.initial_width / self.config.width
        self.max_zoom = initial_pixel_spacing / self.config.min_pixel_spacing
        
        # Set up the widget
        self.setFixedSize(config.width, config.height)
        self.setMouseTracking(True)  # Receive mouse move events without clicking
        
        # Create and start the compute worker
        self.worker = ComputeWorker(config)
        self.worker.computation_complete.connect(self._on_computation_complete)
        self.worker.start()
        
        # Display initial loading state
        self._show_loading_state()
        
        # Request initial computation
        self.worker.request_computation(self.view)
    
    def _show_loading_state(self) -> None:
        """Display a placeholder while computing."""
        # Create a simple grey image
        grey = np.full((self.config.height, self.config.width, 3), 128, dtype=np.uint8)
        self._display_rgb_array(grey)
    
    def _display_rgb_array(self, rgb_array: np.ndarray) -> None:
        """Convert numpy RGB array to QPixmap and display."""
        height, width, channels = rgb_array.shape
        bytes_per_line = channels * width
        
        # Create QImage from numpy array
        # Need to ensure data is contiguous
        rgb_contiguous = np.ascontiguousarray(rgb_array)
        
        qimage = QImage(
            rgb_contiguous.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Keep a reference to the array to prevent garbage collection
        self._current_image_data = rgb_contiguous
        
        self.setPixmap(QPixmap.fromImage(qimage))
    
    def _on_computation_complete(self, rgb_array: np.ndarray, view: ViewState) -> None:
        """Handle completed computation from worker thread."""
        self._display_rgb_array(rgb_array)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track mouse position for zoom centring."""
        self.current_mouse_pos = event.position()
        super().mouseMoveEvent(event)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle scroll wheel for zooming."""
        # Get scroll direction
        delta = event.angleDelta().y()
        
        if delta == 0:
            return
        
        # Determine zoom direction
        if delta > 0:
            # Scroll up = zoom in
            new_zoom = self.view.zoom_level * self.zoom_factor
        else:
            # Scroll down = zoom out
            new_zoom = self.view.zoom_level / self.zoom_factor
        
        # Enforce zoom limits
        if new_zoom < 1.0:
            new_zoom = 1.0  # Don't zoom out beyond initial view
        elif new_zoom > self.max_zoom:
            # Hit precision limit
            new_zoom = self.max_zoom
            if self.view.zoom_level >= self.max_zoom:
                # Already at limit, ignore scroll
                return
        
        # Get zoom centre point (mouse position or widget centre)
        pos = event.position()
        zoom_x = pos.x()
        zoom_y = pos.y()
        
        # Convert pixel position to complex coordinates (before zoom)
        complex_x, complex_y = self.view.pixel_to_complex(
            int(zoom_x), int(zoom_y),
            self.config.width, self.config.height
        )
        
        # Calculate the relative position of mouse in the view (0 to 1)
        rel_x = zoom_x / self.config.width
        rel_y = zoom_y / self.config.height
        
        # Update zoom level
        old_zoom = self.view.zoom_level
        self.view.zoom_level = new_zoom
        
        # Adjust centre so that the point under the mouse stays fixed
        # The complex coordinate under the mouse should remain the same
        # after zooming
        
        # After zoom, what would the new bounds be if we kept the same centre?
        # We need to shift the centre so that (complex_x, complex_y) appears
        # at the same relative position (rel_x, rel_y)
        
        new_width = self.view.current_width
        new_height = self.view.current_height
        
        # New centre such that complex_x is at rel_x, complex_y is at rel_y
        self.view.centre_real = complex_x - (rel_x - 0.5) * new_width
        self.view.centre_imag = complex_y + (rel_y - 0.5) * new_height
        
        # Request new computation
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