"""
Interactive Mandelbrot Set Visualization with PyQt5

This module provides an interactive viewer for the Mandelbrot set with
scroll-based zooming centered on the mouse cursor position, with threaded
computation to prevent UI locking.
"""

import sys
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, QPoint, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PIL import Image


@dataclass
class ViewState:
   """Represents the current view into the complex plane."""
   center_x: float
   center_y: float
   width: float  # Width in complex plane units
   height: float  # Height in complex plane units
   pixel_width: int
   pixel_height: int
   max_iterations: int = 256
   
   def copy(self):
       """Create a copy of this view state."""
       return ViewState(
           self.center_x,
           self.center_y,
           self.width,
           self.height,
           self.pixel_width,
           self.pixel_height,
           self.max_iterations
       )
   
   @property
   def x_min(self) -> float:
       return self.center_x - self.width / 2
   
   @property
   def x_max(self) -> float:
       return self.center_x + self.width / 2
   
   @property
   def y_min(self) -> float:
       return self.center_y - self.height / 2
   
   @property
   def y_max(self) -> float:
       return self.center_y + self.height / 2


def create_complex_matrix(view: ViewState) -> np.ndarray:
   """
   Create a complex matrix representing points in the complex plane.
   
   Args:
       view: ViewState object containing the bounds and resolution.
       
   Returns:
       A 2D numpy array of complex numbers representing the complex plane.
   """
   real_values = np.linspace(view.x_min, view.x_max, view.pixel_width)
   imag_values = np.linspace(view.y_max, view.y_min, view.pixel_height)
   
   real_grid, imag_grid = np.meshgrid(real_values, imag_values)
   complex_matrix = real_grid + 1j * imag_grid
   
   return complex_matrix


def naive_escape_time_algorithm_vectorized(complex_matrix: np.ndarray, 
                                          max_iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   """
   Vectorized naive escape time algorithm using real number arithmetic with NumPy.
   
   This simulates complex number operations using two real number arrays:
       x_{n+1} = x_n^2 - y_n^2 + x_0
       y_{n+1} = 2 * x_n * y_n + y_0
   
   Escape condition: x^2 + y^2 > 4
   
   Args:
       complex_matrix: 2D array of complex numbers to test.
       max_iterations: Maximum iterations before assuming point is in set.
       
   Returns:
       Tuple of (iterations, final_x, final_y) arrays for smooth coloring.
   """
   height, width = complex_matrix.shape
   
   # Extract real and imaginary parts (these are our x0 and y0)
   x0 = np.real(complex_matrix)
   y0 = np.imag(complex_matrix)
   
   # Initialize iteration variables
   x = np.zeros_like(x0, dtype=np.float64)
   y = np.zeros_like(y0, dtype=np.float64)
   
   # Track iteration counts
   iterations = np.zeros((height, width), dtype=np.int32)
   
   # Mask for points that haven't escaped yet
   not_escaped = np.ones((height, width), dtype=bool)
   
   for i in range(max_iterations):
       # Compute x^2 and y^2 (naive approach - computing separately)
       x_squared = x * x
       y_squared = y * y
       
       # Check escape condition: x^2 + y^2 > 4
       magnitude_squared = x_squared + y_squared
       escaped = not_escaped & (magnitude_squared > 4.0)
       
       # Record iteration count for newly escaped points
       iterations[escaped] = i
       
       # Update mask
       not_escaped[escaped] = False
       
       # If all points have escaped, we can exit early
       if not np.any(not_escaped):
           break
       
       # Compute next iteration using real arithmetic
       # new_x = x^2 - y^2 + x0
       # new_y = 2xy + y0
       new_x = x_squared - y_squared + x0
       new_y = 2.0 * x * y + y0
       
       # Update only points that haven't escaped
       x = new_x
       y = new_y
   
   # Points that never escaped remain at max_iterations
   iterations[not_escaped] = max_iterations
   
   return iterations, x, y


def compute_normalized_iteration_count(iterations: np.ndarray, 
                                       final_x: np.ndarray, 
                                       final_y: np.ndarray,
                                       max_iterations: int) -> np.ndarray:
   """
   Compute normalized iteration counts for smooth coloring (vectorized).
   
   The normalized iteration count formula: nu = n + 1 - log2(log2(|z|))
   
   Args:
       iterations: Array of discrete iteration counts.
       final_x: Final x coordinates (real part of z).
       final_y: Final y coordinates (imaginary part of z).
       max_iterations: Maximum iterations.
       
   Returns:
       Array of normalized iteration counts.
   """
   normalized = iterations.astype(np.float64)
   
   # Only compute for points that escaped (not in the set)
   escaped_mask = iterations < max_iterations
   
   if not np.any(escaped_mask):
       return normalized
   
   # Calculate magnitude squared for escaped points
   magnitude_squared = final_x[escaped_mask]**2 + final_y[escaped_mask]**2
   
   # Safety check: magnitude should be > 4 for escaped points
   valid_mask = magnitude_squared > 1.0
   
   if not np.any(valid_mask):
       return normalized
   
   # Calculate magnitude
   magnitude = np.sqrt(magnitude_squared[valid_mask])
   
   # Compute log(log(magnitude)) / log(2) / log(2)
   log_magnitude = np.log(magnitude)
   
   # Filter out any invalid values
   valid_log_mask = log_magnitude > 0
   
   if not np.any(valid_log_mask):
       return normalized
   
   log2_log_magnitude = np.log(log_magnitude[valid_log_mask]) / np.log(2.0)
   
   # Apply the normalized iteration count formula
   # Create a temporary array for the escaped points
   escaped_normalized = iterations[escaped_mask].astype(np.float64)
   escaped_normalized_valid = escaped_normalized[valid_mask]
   escaped_normalized_valid[valid_log_mask] = (
       escaped_normalized_valid[valid_log_mask] + 1.0 - log2_log_magnitude
   )
   
   # Update the normalized array
   temp_escaped = escaped_normalized.copy()
   temp_escaped[valid_mask] = escaped_normalized_valid
   normalized[escaped_mask] = temp_escaped
   
   return np.maximum(0.0, normalized)


def smooth_iterations_to_rgb(smooth_iterations: np.ndarray, 
                            max_iterations: int) -> np.ndarray:
   """
   Convert smooth iteration counts to RGB values.
   
   Args:
       smooth_iterations: 2D array of normalized iteration counts.
       max_iterations: Maximum iterations (for identifying set membership).
       
   Returns:
       3D numpy array of shape (height, width, 3) with RGB values (0-255).
   """
   height, width = smooth_iterations.shape
   rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
   
   # Mask for points in the set (render as black)
   in_set_mask = smooth_iterations >= max_iterations - 1
   
   # Mask for points outside the set
   outside_mask = ~in_set_mask
   
   if not np.any(outside_mask):
       return rgb_array  # All black if everything is in the set
   
   # Normalize the values for points outside the set
   outside_values = smooth_iterations[outside_mask]
   max_val = np.max(outside_values) if np.any(outside_values) else 1.0
   
   if max_val > 0:
       normalized = outside_values / max_val
   else:
       normalized = outside_values
   
   # Apply smooth coloring using sine functions for RGB
   # This creates a pleasing gradient
   r = (255 * (0.5 + 0.5 * np.sin(3.0 * np.pi * normalized))).astype(np.uint8)
   g = (255 * (0.5 + 0.5 * np.sin(3.0 * np.pi * normalized + 2.094))).astype(np.uint8)
   b = (255 * (0.5 + 0.5 * np.sin(3.0 * np.pi * normalized + 4.189))).astype(np.uint8)
   
   # Assign RGB values
   rgb_array[outside_mask, 0] = r
   rgb_array[outside_mask, 1] = g
   rgb_array[outside_mask, 2] = b
   
   # Points in the set remain black (already initialized to 0)
   
   return rgb_array


class MandelbrotComputeThread(QThread):
   """
   Thread for computing the Mandelbrot set without blocking the UI.
   """
   # Signal emitted when computation is complete: (rgb_array, view_state)
   computation_complete = pyqtSignal(np.ndarray, ViewState)
   
   def __init__(self):
       super().__init__()
       self.view_state: Optional[ViewState] = None
       self.should_stop = False
   
   def set_view(self, view: ViewState):
       """Set the view state to compute."""
       self.view_state = view.copy()
       self.should_stop = False
   
   def stop(self):
       """Signal the thread to stop computation."""
       self.should_stop = True
   
   def run(self):
       """Execute the computation in a separate thread."""
       if self.view_state is None:
           return
       
       # Create complex matrix
       complex_matrix = create_complex_matrix(self.view_state)
       
       # Check if we should stop before heavy computation
       if self.should_stop:
           return
       
       # Compute using vectorized naive escape time algorithm
       iterations, final_x, final_y = naive_escape_time_algorithm_vectorized(
           complex_matrix, 
           self.view_state.max_iterations
       )
       
       # Check again before continuing
       if self.should_stop:
           return
       
       # Compute smooth iteration counts
       smooth_iterations = compute_normalized_iteration_count(
           iterations,
           final_x,
           final_y,
           self.view_state.max_iterations
       )
       
       # Check before final step
       if self.should_stop:
           return
       
       # Convert to RGB
       rgb_array = smooth_iterations_to_rgb(
           smooth_iterations,
           self.view_state.max_iterations
       )
       
       # Emit the result if we weren't stopped
       if not self.should_stop:
           self.computation_complete.emit(rgb_array, self.view_state)


class MandelbrotViewer(QMainWindow):
   """
   Interactive Mandelbrot set viewer with scroll-based zooming.
   """
   
   # Precision limit for float64: approximately 10^-15 relative precision
   # With a standard view width of ~3.5, the practical zoom limit is around 10^14
   MIN_VIEW_WIDTH = 1e-13  # Absolute minimum width in complex plane units
   
   # Zoom factor per scroll step
   ZOOM_FACTOR = 1.2
   
   def __init__(self):
       super().__init__()
       self.setWindowTitle("Interactive Mandelbrot Set Viewer")
       
       # Initialize view state to show the classic full Mandelbrot view
       self.current_view = ViewState(
           center_x=-0.5,
           center_y=0.0,
           width=3.5,
           height=2.5,
           pixel_width=800,
           pixel_height=600,
           max_iterations=256
       )
       
       # Current displayed image
       self.current_image: Optional[np.ndarray] = None
       self.current_pixmap: Optional[QPixmap] = None
       
       # Computation thread
       self.compute_thread = MandelbrotComputeThread()
       self.compute_thread.computation_complete.connect(self.on_computation_complete)
       
       # Track if computation is in progress
       self.computing = False
       
       # Track if a new computation is needed
       self.computation_pending = False
       self.pending_view: Optional[ViewState] = None
       
       # Setup UI
       self.setup_ui()
       
       # Start initial computation
       self.request_computation(self.current_view)
   
   def setup_ui(self):
       """Initialize the user interface."""
       # Central widget
       central_widget = QWidget()
       self.setCentralWidget(central_widget)
       
       # Layout
       layout = QVBoxLayout()
       central_widget.setLayout(layout)
       
       # Image label
       self.image_label = QLabel()
       self.image_label.setFixedSize(
           self.current_view.pixel_width,
           self.current_view.pixel_height
       )
       self.image_label.setStyleSheet("background-color: black;")
       self.image_label.setMouseTracking(True)
       
       # Enable mouse tracking for the entire widget
       self.setMouseTracking(True)
       central_widget.setMouseTracking(True)
       
       layout.addWidget(self.image_label)
       
       # Status label
       self.status_label = QLabel("Initializing...")
       layout.addWidget(self.status_label)
       
       # Adjust window size
       self.adjustSize()
       self.setFixedSize(self.size())
   
   def request_computation(self, view: ViewState):
       """
       Request a computation for the given view state.
       If a computation is in progress, queue this request.
       """
       if self.computing:
           # Queue this computation
           self.computation_pending = True
           self.pending_view = view.copy()
           self.status_label.setText("Computing... (new request queued)")
       else:
           # Start computation immediately
           self.computing = True
           self.computation_pending = False
           self.pending_view = None
           self.compute_thread.set_view(view)
           self.compute_thread.start()
           self.status_label.setText("Computing...")
   
   def on_computation_complete(self, rgb_array: np.ndarray, view: ViewState):
       """
       Handle completion of a computation thread.
       """
       self.computing = False
       
       # Update current image and view
       self.current_image = rgb_array
       self.current_view = view
       
       # Convert to QPixmap and display
       height, width, _ = rgb_array.shape
       bytes_per_line = 3 * width
       
       q_image = QImage(
           rgb_array.data,
           width,
           height,
           bytes_per_line,
           QImage.Format_RGB888
       )
       
       self.current_pixmap = QPixmap.fromImage(q_image)
       self.image_label.setPixmap(self.current_pixmap)
       
       # Update status
       self.update_status_label()
       
       # If there's a pending computation, start it
       if self.computation_pending and self.pending_view is not None:
           self.request_computation(self.pending_view)
   
   def update_status_label(self):
       """Update the status label with current view information."""
       # Calculate current zoom level relative to initial view
       initial_width = 3.5
       zoom_level = initial_width / self.current_view.width
       
       # Check if we're near the precision limit
       precision_percent = (self.current_view.width / self.MIN_VIEW_WIDTH) * 100
       
       if precision_percent < 200:  # Within 2x of the limit
           status_text = (
               f"Center: ({self.current_view.center_x:.10e}, {self.current_view.center_y:.10e}) | "
               f"Zoom: {zoom_level:.2e}x | "
               f"⚠ Approaching precision limit ({precision_percent:.0f}%)"
           )
       else:
           status_text = (
               f"Center: ({self.current_view.center_x:.6f}, {self.current_view.center_y:.6f}) | "
               f"Zoom: {zoom_level:.2e}x"
           )
       
       self.status_label.setText(status_text)
   
   def wheelEvent(self, event):
       """
       Handle mouse wheel events for zooming.
       Zoom is centered on the current mouse position.
       """
       # Get mouse position relative to the image label
       mouse_pos = self.image_label.mapFromGlobal(event.globalPos())
       
       # Check if mouse is within the image bounds
       if not (0 <= mouse_pos.x() < self.current_view.pixel_width and
               0 <= mouse_pos.y() < self.current_view.pixel_height):
           return
       
       # Determine zoom direction
       angle_delta = event.angleDelta().y()
       
       if angle_delta > 0:
           # Zoom in
           zoom_factor = 1.0 / self.ZOOM_FACTOR
       else:
           # Zoom out
           zoom_factor = self.ZOOM_FACTOR
       
       # Calculate new view dimensions
       new_width = self.current_view.width * zoom_factor
       new_height = self.current_view.height * zoom_factor
       
       # Check precision limit (only for zoom in)
       if new_width < self.MIN_VIEW_WIDTH:
           self.status_label.setText(
               "⚠ Zoom limit reached - float64 precision exhausted"
           )
           return
       
       # Convert mouse pixel coordinates to complex plane coordinates
       pixel_x = mouse_pos.x()
       pixel_y = mouse_pos.y()
       
       # Current position of mouse in complex plane
       mouse_complex_x = (self.current_view.x_min + 
                         (pixel_x / self.current_view.pixel_width) * self.current_view.width)
       mouse_complex_y = (self.current_view.y_max - 
                         (pixel_y / self.current_view.pixel_height) * self.current_view.height)
       
       # Calculate offset from current center to mouse position
       offset_x = mouse_complex_x - self.current_view.center_x
       offset_y = mouse_complex_y - self.current_view.center_y
       
       # New center: zoom toward the mouse position
       # When zooming in (zoom_factor < 1), we move the center toward the mouse
       # When zooming out (zoom_factor > 1), we move the center away from the mouse
       new_center_x = self.current_view.center_x + offset_x * (1 - zoom_factor)
       new_center_y = self.current_view.center_y + offset_y * (1 - zoom_factor)
       
       # Create new view state
       new_view = ViewState(
           center_x=new_center_x,
           center_y=new_center_y,
           width=new_width,
           height=new_height,
           pixel_width=self.current_view.pixel_width,
           pixel_height=self.current_view.pixel_height,
           max_iterations=self.current_view.max_iterations
       )
       
       # Request computation for new view
       self.request_computation(new_view)
   
   def closeEvent(self, event):
       """Handle window close event."""
       # Stop any ongoing computation
       if self.computing:
           self.compute_thread.stop()
           self.compute_thread.wait()
       
       event.accept()


def main():
   """Main entry point for the interactive Mandelbrot viewer."""
   app = QApplication(sys.argv)
   
   viewer = MandelbrotViewer()
   viewer.show()
   
   sys.exit(app.exec_())


if __name__ == "__main__":
   main()