"""
Interactive Mandelbrot Set Viewer

A PyQt6-based interactive viewer with threaded computation, scroll-wheel zooming
centred on mouse position, and precision limit detection.
"""

import sys
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

from PyQt6.QtWidgets import (
   QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
   QWidget, QFileDialog, QMessageBox, QSlider, QGroupBox, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMutex, QWaitCondition
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent

from PIL import Image, ImageDraw, ImageFont

from numba import jit, prange
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
   progressive_passes: Tuple[int, ...] = (8, 4, 2, 1)
   
   # Tile cache settings
   max_cached_tiles: int = 256
   
   # Precision limit for float64
   min_pixel_spacing: float = field(default_factory=lambda: np.finfo(np.float64).eps * 1000)
   
   # Adaptive iteration scaling with zoom
   base_iterations: int = 256
   iteration_zoom_factor: float = 50.0


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_mandelbrot_numba(x_min: float, x_max: float, 
                             y_min: float, y_max: float,
                             width: int, height: int,
                             max_iterations: int) -> np.ndarray:
   """
   Numba JIT-compiled Mandelbrot computation with parallel processing.
   """
   result = np.zeros((height, width), dtype=np.float64)
   pixel_width = (x_max - x_min) / width
   pixel_height = (y_max - y_min) / height
   
   for py in prange(height):
       y0 = y_max - py * pixel_height
       
       for px in range(width):
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
               result[py, px] = smooth_val
           else:
               result[py, px] = max_iterations
   
   return result


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_mandelbrot_subsampled(x_min: float, x_max: float,
                                  y_min: float, y_max: float,
                                  width: int, height: int,
                                  max_iterations: int,
                                  subsample: int) -> np.ndarray:
   """
   Compute Mandelbrot at reduced resolution for progressive rendering.
   """
   out_height = height // subsample
   out_width = width // subsample
   result = np.zeros((out_height, out_width), dtype=np.float64)
   
   pixel_width = (x_max - x_min) / width
   pixel_height = (y_max - y_min) / height
   
   for out_py in prange(out_height):
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
   """Fast nearest-neighbour upscaling using Numba."""
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


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def colour_selector_numba(iterations: np.ndarray, max_iterations: int,
                          hue_shift: float, saturation: float,
                          lightness: float, max_external: float) -> np.ndarray:
    """
    Numba JIT-compiled HSL colour conversion with parallel processing.
    
    Args:
        iterations: 2D array of iteration counts.
        max_iterations: Maximum iteration value.
        hue_shift: Hue rotation in range [0, 1].
        saturation: Saturation multiplier in range [0, 1].
        lightness: Lightness adjustment in range [0, 1].
        max_external: Pre-computed maximum iteration value for normalisation.
        
    Returns:
        3D RGB array (height, width, 3).
    """
    height, width = iterations.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Precompute constants
    threshold = max_iterations - 1.0
    inv_max_external = 1.0 / max_external if max_external > 0 else 1.0
    
    for y in prange(height):
        for x in range(width):
            val = iterations[y, x]
            
            # Points in the set are black
            if val >= threshold:
                rgb[y, x, 0] = 0
                rgb[y, x, 1] = 0
                rgb[y, x, 2] = 0
                continue
            
            # Normalise and apply hue shift
            normalised = val * inv_max_external
            hue = (normalised + hue_shift) % 1.0
            
            # HSL to RGB conversion (inline)
            c = (1.0 - abs(2.0 * lightness - 1.0)) * saturation
            h_sector = hue * 6.0
            x_val = c * (1.0 - abs(h_sector % 2.0 - 1.0))
            m = lightness - c * 0.5
            
            # Determine RGB based on hue sector
            sector = int(h_sector) % 6
            
            if sector == 0:
                r, g, b = c, x_val, 0.0
            elif sector == 1:
                r, g, b = x_val, c, 0.0
            elif sector == 2:
                r, g, b = 0.0, c, x_val
            elif sector == 3:
                r, g, b = 0.0, x_val, c
            elif sector == 4:
                r, g, b = x_val, 0.0, c
            else:
                r, g, b = c, 0.0, x_val
            
            # Add m and convert to 0-255, clamping values
            r_val = (r + m) * 255.0
            g_val = (g + m) * 255.0
            b_val = (b + m) * 255.0
            
            # Clamp to valid range
            if r_val < 0.0:
                r_val = 0.0
            elif r_val > 255.0:
                r_val = 255.0
            if g_val < 0.0:
                g_val = 0.0
            elif g_val > 255.0:
                g_val = 255.0
            if b_val < 0.0:
                b_val = 0.0
            elif b_val > 255.0:
                b_val = 255.0
            
            rgb[y, x, 0] = np.uint8(r_val)
            rgb[y, x, 1] = np.uint8(g_val)
            rgb[y, x, 2] = np.uint8(b_val)
    
    return rgb


def colour_selector(iterations: np.ndarray, max_iterations: int,
                    hue_shift: float = 0.0, saturation: float = 1.0,
                    lightness: float = 0.5) -> np.ndarray:
    """
    Wrapper function that calls the Numba-optimised colour conversion.
    
    Maintains the same interface as the original function.
    """
    # Mask for points in the set
    in_set = iterations >= max_iterations - 1
    
    # Calculate max_external for normalisation
    if np.any(~in_set):
        max_external = np.max(iterations[~in_set])
    else:
        max_external = 1.0
    
    if max_external <= 0:
        max_external = 1.0
    
    return colour_selector_numba(
        iterations, max_iterations,
        hue_shift, saturation, lightness,
        max_external
    )

def keyboard_shortcuts(key: int) -> Optional[Tuple[str, float]]:
    """
    Map keyboard keys to navigation actions.
    
    Args:
        key: Qt key code from the key event.
        
    Returns:
        Tuple of (action_type, value) or None if key not mapped.
        Action types: 'pan_x', 'pan_y', 'zoom', 'reset'
        Values: direction/factor for pan/zoom, 0 for reset.
    """
    key_mappings = {
        # WASD panning (value is direction multiplier)
        Qt.Key.Key_W: ('pan_y', 1.0),   # Pan up (positive imaginary)
        Qt.Key.Key_S: ('pan_y', -1.0),  # Pan down (negative imaginary)
        Qt.Key.Key_A: ('pan_x', -1.0),  # Pan left (negative real)
        Qt.Key.Key_D: ('pan_x', 1.0),   # Pan right (positive real)
        
        # QE zooming (value is zoom multiplier)
        Qt.Key.Key_Q: ('zoom', 0.5),    # Zoom out (decrease zoom level)
        Qt.Key.Key_E: ('zoom', 2.0),    # Zoom in (increase zoom level)
        
        # Reset
        Qt.Key.Key_R: ('reset', 0.0),   # Reset to initial view
    }
    
    return key_mappings.get(key)

class TileCache:
   """Cache for computed Mandelbrot tiles."""
   
   def __init__(self, max_tiles: int = 256):
       self.max_tiles = max_tiles
       self.cache: Dict[str, np.ndarray] = {}
       self.access_order: list = []
       self.mutex = QMutex()
   
   def _make_key(self, x_min: float, x_max: float, 
                 y_min: float, y_max: float,
                 width: int, height: int,
                 max_iter: int) -> str:
       """Generate a unique cache key for tile parameters."""
       key_data = f"{x_min:.15g},{x_max:.15g},{y_min:.15g},{y_max:.15g},{width},{height},{max_iter}"
       return hashlib.md5(key_data.encode()).hexdigest()
   
   def get(self, x_min: float, x_max: float,
           y_min: float, y_max: float,
           width: int, height: int,
           max_iter: int) -> Optional[np.ndarray]:
       """Retrieve a cached tile if available."""
       key = self._make_key(x_min, x_max, y_min, y_max, width, height, max_iter)
       
       self.mutex.lock()
       try:
           if key in self.cache:
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
       """Store a computed tile in the cache."""
       key = self._make_key(x_min, x_max, y_min, y_max, width, height, max_iter)
       
       self.mutex.lock()
       try:
           while len(self.cache) >= self.max_tiles and self.access_order:
               oldest_key = self.access_order.pop(0)
               if oldest_key in self.cache:
                   del self.cache[oldest_key]
           
           self.cache[key] = data.copy()
           self.access_order.append(key)
       finally:
           self.mutex.unlock()


def get_adaptive_iterations(zoom_level: float, config: RenderConfig) -> int:
   """Calculate appropriate iteration count based on zoom level."""
   if zoom_level <= 1.0:
       return config.base_iterations
   
   zoom_orders = np.log10(zoom_level)
   additional = int(zoom_orders * config.iteration_zoom_factor)
   
   return config.base_iterations + additional


class ComputeWorker(QThread):
   """Worker thread with progressive rendering and caching."""
   
   # Signal for each progressive pass (rgb_array, view_state, is_final, iterations, max_iter)
   computation_progress = pyqtSignal(np.ndarray, object, bool, object, int)
   
   def __init__(self, config: RenderConfig):
       super().__init__()
       self.config = config
       
       # Thread synchronisation
       self.mutex = QMutex()
       self.condition = QWaitCondition()
       
       # Request state
       self.pending_view: Optional[ViewState] = None
       self.current_request_id: int = 0
       self.should_stop = False
       self.has_work = False
       
       # Tile cache
       self.tile_cache = TileCache(config.max_cached_tiles)
       
       # HSL colour parameters
       self.hue_shift: float = 0.0
       self.saturation: float = 1.0
       self.lightness: float = 0.5
       
       # Warm up Numba JIT compilation
       self._warmup_jit()
   
   def _warmup_jit(self) -> None:
       """Pre-compile Numba functions to avoid delay on first zoom."""
       _ = compute_mandelbrot_numba(-2.0, 1.0, -1.0, 1.0, 16, 16, 32)
       _ = compute_mandelbrot_subsampled(-2.0, 1.0, -1.0, 1.0, 16, 16, 32, 4)
       test_iter = np.zeros((4, 4), dtype=np.float64)
       _ = upscale_nearest(test_iter, 2)
       _ = colour_selector_numba(test_iter, 32, 0.0, 1.0, 0.5, 1.0)
   
   def set_colour_params(self, hue: float, saturation: float, lightness: float) -> None:
       """Update colour parameters."""
       self.mutex.lock()
       self.hue_shift = hue
       self.saturation = saturation
       self.lightness = lightness
       self.mutex.unlock()
   
   def request_computation(self, view: ViewState) -> None:
       """Request computation for a new view."""
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
               self.mutex.lock()
               hue = self.hue_shift
               sat = self.saturation
               lit = self.lightness
               self.mutex.unlock()
               
               rgb = colour_selector(cached, max_iter, hue, sat, lit)
               if not self._is_request_stale(request_id):
                   self.computation_progress.emit(rgb, view, True, cached, max_iter)
               continue
           
           # Progressive rendering passes
           for subsample in self.config.progressive_passes:
               if self._is_request_stale(request_id):
                   break
               
               is_final = (subsample == 1)
               
               if subsample > 1:
                   iterations = compute_mandelbrot_subsampled(
                       view.x_min, view.x_max,
                       view.y_min, view.y_max,
                       self.config.width, self.config.height,
                       max_iter, subsample
                   )
                   iterations_full = upscale_nearest(iterations, subsample)
               else:
                   iterations_full = compute_mandelbrot_numba(
                       view.x_min, view.x_max,
                       view.y_min, view.y_max,
                       self.config.width, self.config.height,
                       max_iter
                   )
                   
                   self.tile_cache.put(
                       view.x_min, view.x_max, view.y_min, view.y_max,
                       self.config.width, self.config.height, max_iter,
                       iterations_full
                   )
               
               if self._is_request_stale(request_id):
                   break
               
               # Convert to RGB using colour_selector
               self.mutex.lock()
               hue = self.hue_shift
               sat = self.saturation
               lit = self.lightness
               self.mutex.unlock()
               
               rgb = colour_selector(iterations_full, max_iter, hue, sat, lit)
               
               if not self._is_request_stale(request_id):
                   self.computation_progress.emit(rgb, view, is_final, iterations_full, max_iter)


class MandelbrotWidget(QLabel):
    """Widget displaying the Mandelbrot set with scroll-wheel zoom and click-drag pan."""
    
    view_changed = pyqtSignal(object, bool)
    
    def __init__(self, config: RenderConfig, parent=None):
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
    
    def handle_keyboard_navigation(self, key: int) -> bool:
        """
        Process keyboard navigation input.
        
        Args:
            key: Qt key code from the key event.
            
        Returns:
            True if the key was handled, False otherwise.
        """
        action = keyboard_shortcuts(key)
        
        if action is None:
            return False
        
        action_type, value = action
        
        # Pan distance as fraction of current view
        pan_fraction = 0.2
        
        if action_type == 'pan_x':
            # Pan horizontally
            pan_amount = self.view.current_width * pan_fraction * value
            self.view.centre_real += pan_amount
            self.worker.request_computation(self.view)
            
        elif action_type == 'pan_y':
            # Pan vertically
            pan_amount = self.view.current_height * pan_fraction * value
            self.view.centre_imag += pan_amount
            self.worker.request_computation(self.view)
            
        elif action_type == 'zoom':
            # Zoom in/out from centre
            new_zoom = self.view.zoom_level * value
            
            # Enforce zoom limits
            if new_zoom < 1.0:
                new_zoom = 1.0
            elif new_zoom > self.max_zoom:
                new_zoom = self.max_zoom
                if self.view.zoom_level >= self.max_zoom:
                    return True
            
            self.view.zoom_level = new_zoom
            self.worker.request_computation(self.view)
            
        elif action_type == 'reset':
            # Reset to initial view
            self.view = ViewState()
            self.worker.request_computation(self.view)
        
        return True

    def load_bookmark(self, bookmark: Dict) -> None:
        """
        Load a view state from a bookmark.
        
        Args:
            bookmark: Dict containing 'centre_real', 'centre_imag', 'zoom_level'.
        """
        self.view.centre_real = bookmark['centre_real']
        self.view.centre_imag = bookmark['centre_imag']
        self.view.zoom_level = bookmark['zoom_level']
        self.worker.request_computation(self.view)

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
            delta_x = event.position().x() - self.pan_start_pos.x()
            delta_y = event.position().y() - self.pan_start_pos.y()
            
            complex_delta_x = -delta_x * (self.pan_start_view.current_width / self.config.width)
            complex_delta_y = delta_y * (self.pan_start_view.current_height / self.config.height)
            
            self.view.centre_real = self.pan_start_view.centre_real + complex_delta_x
            self.view.centre_imag = self.pan_start_view.centre_imag + complex_delta_y
            
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
        if self.is_panning:
            return
        
        delta = event.angleDelta().y()
        
        if delta == 0:
            return
        
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
   """Handles saving Mandelbrot images with embedded metadata."""
   
   @staticmethod
   def save_image(rgb_data: np.ndarray, view: ViewState, 
                  filepath: Path, embed_text: bool = True) -> bool:
       """Save the current view as a PNG image with metadata."""
       try:
           image = Image.fromarray(rgb_data, mode='RGB')
           
           if embed_text:
               image = ImageSaver._add_metadata_overlay(image, view)
           
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
           
           image.save(filepath, "PNG", pnginfo=metadata)
           
           return True
           
       except Exception as e:
           print(f"Error saving image: {e}")
           return False
   
   @staticmethod
   def _add_metadata_overlay(image: Image.Image, view: ViewState) -> Image.Image:
       """Add a semi-transparent overlay with coordinate information."""
       image = image.copy()
       draw = ImageDraw.Draw(image, 'RGBA')
       
       text_lines = [
           f"Centre: {view.centre_real:.10g} + {view.centre_imag:.10g}i",
           f"Zoom: {view.zoom_level:.4e}x"
       ]
       text = "\n".join(text_lines)
       
       try:
           font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
       except OSError:
           try:
               font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 12)
           except OSError:
               font = ImageFont.load_default()
       
       bbox = draw.textbbox((0, 0), text, font=font)
       text_width = bbox[2] - bbox[0]
       text_height = bbox[3] - bbox[1]
       
       padding = 10
       x = padding
       y = image.height - text_height - padding - 10
       
       bg_rect = [
           x - 5,
           y - 5,
           x + text_width + 10,
           y + text_height + 10
       ]
       draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
       draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
       
       return image
   
   @staticmethod
   def generate_filename(view: ViewState) -> str:
       """Generate a descriptive filename based on view parameters."""
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       zoom_str = f"{view.zoom_level:.2e}".replace("+", "").replace(".", "p")
       return f"mandelbrot_{timestamp}_zoom{zoom_str}.png"


class BookmarkManager:
    """
    Manages persistent bookmarks for Mandelbrot set positions.
    
    Stores up to 5 bookmarks in a JSON file, each containing
    view state and a user-provided name.
    """
    
    MAX_BOOKMARKS = 5
    FILENAME = "mandelbrot_bookmarks.json"
    
    def __init__(self):
        self.filepath = Path.home() / ".config" / self.FILENAME
        self.bookmarks: Dict[int, Dict] = {}
        self._load()
    
    def _load(self) -> None:
        """Load bookmarks from disk."""
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to integers
                    self.bookmarks = {int(k): v for k, v in data.items()}
        except (json.JSONDecodeError, IOError):
            self.bookmarks = {}
    
    def _save(self) -> None:
        """Save bookmarks to disk."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self.bookmarks, f, indent=2)
        except IOError as e:
            print(f"Failed to save bookmarks: {e}")
    
    def get(self, slot: int) -> Optional[Dict]:
        """
        Retrieve a bookmark by slot number.
        
        Args:
            slot: Slot number (1-5).
            
        Returns:
            Bookmark dict with 'name', 'centre_real', 'centre_imag', 'zoom_level'
            or None if slot is empty.
        """
        return self.bookmarks.get(slot)
    
    def save(self, slot: int, name: str, view: ViewState) -> None:
        """
        Save a bookmark to a slot.
        
        Args:
            slot: Slot number (1-5).
            name: User-provided name for the bookmark.
            view: ViewState to save.
        """
        self.bookmarks[slot] = {
            'name': name,
            'centre_real': view.centre_real,
            'centre_imag': view.centre_imag,
            'zoom_level': view.zoom_level
        }
        self._save()
    
    def get_empty_slot(self) -> Optional[int]:
        """
        Find the first empty slot.
        
        Returns:
            Slot number (1-5) or None if all slots are full.
        """
        for slot in range(1, self.MAX_BOOKMARKS + 1):
            if slot not in self.bookmarks:
                return slot
        return None
    
    def get_all(self) -> Dict[int, Dict]:
        """Return all bookmarks."""
        return self.bookmarks.copy()
    
    def delete(self, slot: int) -> bool:
        """
        Delete a bookmark from a slot.
        
        Args:
            slot: Slot number (1-5).
            
        Returns:
            True if bookmark was deleted, False if slot was empty.
        """
        if slot in self.bookmarks:
            del self.bookmarks[slot]
            self._save()
            return True
        return False


class MandelbrotWindow(QMainWindow):
    """Main application window with save functionality."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Mandelbrot Set Viewer")
        
        # Create render configuration
        self.config = RenderConfig(
            width=800,
            height=600,
            max_iterations=256
        )

        self.bookmark_manager = BookmarkManager()

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
        
        # Create status bar
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
        
        # Create instructions label
        self.instructions_label = QLabel(
            "Scroll/Q/E: Zoom  |  Drag/WASD: Pan  |  R: Reset  |  "
            "1-5: Load/Save Bookmark  |  Shift+1-5: Save  |  Ctrl+S: Save Image"
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
        self.hue_slider.valueChanged.connect(self._on_colour_changed)
        self.sat_slider.valueChanged.connect(self._on_colour_changed)
        self.lit_slider.valueChanged.connect(self._on_colour_changed)
        
        # Size window to fit contents
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())
    
    def _handle_bookmark_key(self, slot: int) -> None:
        """
        Handle bookmark key press (1-5).
        
        Shift+number saves, number alone loads.
        
        Args:
            slot: Bookmark slot number (1-5).
        """
        bookmark = self.bookmark_manager.get(slot)
    
        if bookmark:
            # Load existing bookmark
            self.mandelbrot_widget.load_bookmark(bookmark)
            self._show_status_message(f"Loaded: {bookmark['name']}")
        else:
            # Empty slot - prompt to save
            self._save_bookmark_to_slot(slot)

    def _save_bookmark(self) -> None:
        """Initiate saving a bookmark, finding a slot or prompting for overwrite."""
        empty_slot = self.bookmark_manager.get_empty_slot()
        
        if empty_slot:
            self._save_bookmark_to_slot(empty_slot)
        else:
            self._prompt_overwrite_bookmark()

    def _save_bookmark_to_slot(self, slot: int) -> None:
        """
        Save current view to a specific slot.
        
        Args:
            slot: Slot number (1-5).
        """
        
        view = self.mandelbrot_widget.get_current_view()
        default_name = f"Zoom {view.zoom_level:.2e}x"
        
        name, ok = QInputDialog.getText(
            self,
            f"Save Bookmark (Slot {slot})",
            "Enter bookmark name:",
            text=default_name
        )
        
        if ok and name.strip():
            self.bookmark_manager.save(slot, name.strip(), view)
            self._show_status_message(f"Saved to slot {slot}: {name.strip()}")

    def _prompt_overwrite_bookmark(self) -> None:
        """Show dialog to select which bookmark slot to overwrite."""
        from PyQt6.QtWidgets import QInputDialog
        
        bookmarks = self.bookmark_manager.get_all()
        
        # Build list of current bookmarks
        choices = []
        for slot in range(1, BookmarkManager.MAX_BOOKMARKS + 1):
            if slot in bookmarks:
                name = bookmarks[slot]['name']
                zoom = bookmarks[slot]['zoom_level']
                choices.append(f"{slot}: {name} (Zoom: {zoom:.2e}x)")
        
        choice, ok = QInputDialog.getItem(
            self,
            "All Slots Full",
            "Select a bookmark to overwrite:",
            choices,
            0,
            False
        )
        
        if ok and choice:
            # Extract slot number from choice string
            slot = int(choice.split(':')[0])
            self._save_bookmark_to_slot(slot)

    def _show_status_message(self, message: str) -> None:
        """
        Temporarily show a message in the status bar.
        
        Args:
            message: Message to display.
        """
        from PyQt6.QtCore import QTimer
        
        original_text = self.status_label.text()
        self.status_label.setText(f"  {message}")
        
        # Restore original status after 2 seconds
        QTimer.singleShot(2000, lambda: self._update_status())

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
        # Ctrl+S to save image
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_S:
                self._save_image()
                return
        
        # Shift+number to save bookmark
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            slot = self._key_to_slot(event.key())
            if slot:
                self._save_bookmark_to_slot(slot)
                return
        
        # Handle navigation and bookmark keys (no modifiers)
        if event.modifiers() == Qt.KeyboardModifier.NoModifier:
            # Check for bookmark keys (1-5)
            slot = self._key_to_slot(event.key())
            if slot:
                self._handle_bookmark_key(slot)
                return
            
            # Check for navigation keys
            if self.mandelbrot_widget.handle_keyboard_navigation(event.key()):
                return
        
        super().keyPressEvent(event)

    def _key_to_slot(self, key: int) -> Optional[int]:
        """
        Convert a key code to a bookmark slot number.
        
        Args:
            key: Qt key code.
            
        Returns:
            Slot number (1-5) or None if not a bookmark key.
        """
        key_to_slot_map = {
            Qt.Key.Key_1: 1,
            Qt.Key.Key_2: 2,
            Qt.Key.Key_3: 3,
            Qt.Key.Key_4: 4,
            Qt.Key.Key_5: 5,
        }
        return key_to_slot_map.get(key)
        
    
    def _save_image(self) -> None:
        """Open save dialog and save current view."""
        rgb_data = self.mandelbrot_widget.get_current_image()
        
        if rgb_data is None:
            QMessageBox.warning(
                self,
                "Cannot Save",
                "No image data available to save."
            )
            return
        
        view = self.mandelbrot_widget.get_current_view()
        suggested_name = ImageSaver.generate_filename(view)
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mandelbrot Image",
            suggested_name,
            "PNG Images (*.png);;All Files (*)"
        )
        
        if not filepath:
            return
        
        if not filepath.lower().endswith('.png'):
            filepath += '.png'
        
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
   
   app.setStyle('Fusion')
   
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