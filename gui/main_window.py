"""
Main Window for the Intruder Detection System

This module implements the main dashboard with 5 specialized modules
and real-time system status display.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow:
    """
    Main application window with modular interface.
    
    Features:
    - 5-module navigation system
    - Real-time system status
    - Performance metrics display
    - Centralized control interface
    """
    
    def __init__(self, title: str = "Intruder Detection System 2025"):
        """
        Initialize the main window.
        
        Args:
            title: Window title
        """
        self.title = title
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1280x720")
        self.root.minsize(1024, 600)
        
        # Module instances
        self.modules = {}
        self.current_module = None
        
        # System status
        self.system_status = {
            'detection_active': False,
            'camera_connected': False,
            'telegram_connected': False,
            'database_connected': False,
            'performance_metrics': {}
        }
        
        # Status update thread
        self.status_update_thread = None
        self.status_update_running = False
        
        # Callbacks for system integration
        self.callbacks = {
            'start_detection': None,
            'stop_detection': None,
            'get_system_status': None,
            'get_performance_metrics': None
        }
        
        self._setup_ui()
        self._setup_status_updates()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for title and status
        self._setup_header()
        
        # Middle frame for navigation and content
        self._setup_content_area()
        
        # Bottom frame for status bar
        self._setup_status_bar()
    
    def _setup_header(self):
        """Set up the header with title and quick status."""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="🚀 Intruder Detection System 2025",
            font=("Arial", 20, "bold")
        )
        title_label.pack(side=tk.LEFT)
        
        # Quick status indicators
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side=tk.RIGHT)
        
        self.status_indicators = {}
        indicators = [
            ('detection', '🔍 Detection'),
            ('camera', '📹 Camera'),
            ('telegram', '📱 Telegram'),
            ('database', '💾 Database')
        ]
        
        for key, label in indicators:
            indicator_frame = ttk.Frame(status_frame)
            indicator_frame.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(indicator_frame, text=label, font=("Arial", 9)).pack()
            status_label = ttk.Label(
                indicator_frame, 
                text="●", 
                foreground="red",
                font=("Arial", 12)
            )
            status_label.pack()
            self.status_indicators[key] = status_label
    
    def _setup_content_area(self):
        """Set up the main content area with navigation and module display."""
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Navigation panel (left side)
        nav_frame = ttk.LabelFrame(content_frame, text="Modules", padding=10)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Module buttons
        modules = [
            ('detection', '🔍 Detection View', 'Real-time detection and monitoring'),
            ('cameras', '📹 IP Camera Manager', 'Network camera configuration'),
            ('entities', '👥 Entity Management', 'Human and animal registration'),
            ('notifications', '📱 Notification Center', 'Telegram user management'),
            ('performance', '📊 Performance Monitor', 'System metrics and analysis')
        ]
        
        self.nav_buttons = {}
        for module_id, title, description in modules:
            btn_frame = ttk.Frame(nav_frame)
            btn_frame.pack(fill=tk.X, pady=2)
            
            btn = ttk.Button(
                btn_frame,
                text=title,
                command=lambda mid=module_id: self.show_module(mid),
                width=25
            )
            btn.pack(fill=tk.X)
            
            # Description label
            desc_label = ttk.Label(
                btn_frame,
                text=description,
                font=("Arial", 8),
                foreground="gray"
            )
            desc_label.pack(fill=tk.X)
            
            self.nav_buttons[module_id] = btn
        
        # Quick action buttons
        ttk.Separator(nav_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(
            nav_frame,
            text="🚀 Start Detection",
            command=self.start_detection,
            style="Accent.TButton"
        )
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(
            nav_frame,
            text="⏹️ Stop Detection",
            command=self.stop_detection,
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Module content area (right side)
        self.content_frame = ttk.LabelFrame(content_frame, text="Module Content", padding=10)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Welcome message
        self._show_welcome()
    
    def _setup_status_bar(self):
        """Set up the bottom status bar."""
        self.status_bar = ttk.Frame(self.main_frame)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Status text
        self.status_text = ttk.Label(
            self.status_bar,
            text="Ready - Select a module to begin",
            font=("Arial", 9)
        )
        self.status_text.pack(side=tk.LEFT)
        
        # Performance metrics
        self.metrics_text = ttk.Label(
            self.status_bar,
            text="",
            font=("Arial", 9),
            foreground="blue"
        )
        self.metrics_text.pack(side=tk.RIGHT)
    
    def _show_welcome(self):
        """Show welcome message in content area."""
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        welcome_frame = ttk.Frame(self.content_frame)
        welcome_frame.pack(expand=True, fill=tk.BOTH)
        
        # Welcome message
        welcome_label = ttk.Label(
            welcome_frame,
            text="Welcome to the Intruder Detection System",
            font=("Arial", 16, "bold")
        )
        welcome_label.pack(pady=20)
        
        # Feature overview
        features_text = """
🔍 Advanced Detection: YOLO11n with person tracking
🎭 Face Recognition: Multi-face detection with configurable thresholds
🐕 Pet Identification: Individual pet recognition (e.g., 'Jacky')
📹 IP Camera Support: HTTP/HTTPS with automatic fallback
📱 Telegram Integration: Bidirectional bot with command listening
📊 Performance Monitoring: Real-time metrics and optimization

Select a module from the left panel to get started.
        """
        
        features_label = ttk.Label(
            welcome_frame,
            text=features_text,
            font=("Arial", 11),
            justify=tk.CENTER
        )
        features_label.pack(pady=20)
        
        # Quick start button
        quick_start_btn = ttk.Button(
            welcome_frame,
            text="🚀 Quick Start - Detection View",
            command=lambda: self.show_module('detection'),
            style="Accent.TButton"
        )
        quick_start_btn.pack(pady=10)
    
    def show_module(self, module_id: str):
        """
        Show a specific module in the content area.
        
        Args:
            module_id: ID of the module to show
        """
        try:
            # Clear content frame
            for widget in self.content_frame.winfo_children():
                widget.destroy()
            
            # Update navigation button states
            for btn_id, btn in self.nav_buttons.items():
                if btn_id == module_id:
                    btn.configure(style="Accent.TButton")
                else:
                    btn.configure(style="TButton")
            
            # Load and show module
            if module_id not in self.modules:
                self._load_module(module_id)
            
            if module_id in self.modules:
                module = self.modules[module_id]
                module.show_in_frame(self.content_frame)
                self.current_module = module_id
                self.content_frame.configure(text=f"Module: {module.get_title()}")
                self.status_text.configure(text=f"Active module: {module.get_title()}")
            else:
                self._show_module_placeholder(module_id)
                
        except Exception as e:
            logger.error(f"Error showing module {module_id}: {e}")
            messagebox.showerror("Error", f"Failed to load module: {e}")
    
    def _load_module(self, module_id: str):
        """Load a module dynamically."""
        try:
            if module_id == 'detection':
                from .detection_view import DetectionView
                module = DetectionView()
                if hasattr(self, 'main_system') and self.main_system:
                    module.set_main_system(self.main_system)
                    # Set up detection callback
                    if hasattr(module, 'set_detection_callback') and hasattr(self.main_system, 'update_detection_settings'):
                        module.set_detection_callback(self.main_system.update_detection_settings)
                        logger.info("Detection settings callback configured for detection module")
                self.modules[module_id] = module
            elif module_id == 'cameras':
                from .ip_camera_manager import IPCameraManager
                module = IPCameraManager()
                if hasattr(self, 'main_system') and self.main_system:
                    module.set_main_system(self.main_system)
                    logger.info("Main system reference configured for camera module")
                self.modules[module_id] = module
            elif module_id == 'entities':
                from .entity_management import EntityManagement
                self.modules[module_id] = EntityManagement()
            elif module_id == 'notifications':
                from .notification_center import NotificationCenter
                module = NotificationCenter()
                if hasattr(self, 'main_system') and self.main_system:
                    # Pass system components to notification center
                    module.set_database_manager(self.main_system.db_manager)
                    module.set_notification_system(self.main_system.notification_system)
                    module.set_config_manager(self.main_system.config_manager)
                self.modules[module_id] = module
            elif module_id == 'performance':
                from .performance_monitor import PerformanceMonitor
                module = PerformanceMonitor()
                if hasattr(self, 'main_system') and self.main_system:
                    module.set_main_system(self.main_system)
                self.modules[module_id] = module
            else:
                logger.warning(f"Unknown module: {module_id}")
                
        except ImportError as e:
            logger.error(f"Failed to import module {module_id}: {e}")
        except Exception as e:
            logger.error(f"Error loading module {module_id}: {e}")
    
    def _show_module_placeholder(self, module_id: str):
        """Show placeholder for modules that couldn't be loaded."""
        placeholder_frame = ttk.Frame(self.content_frame)
        placeholder_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(
            placeholder_frame,
            text=f"Module '{module_id}' is being developed",
            font=("Arial", 14)
        ).pack(expand=True)
        
        ttk.Label(
            placeholder_frame,
            text="This module will be available in the next update.",
            font=("Arial", 10),
            foreground="gray"
        ).pack()
    
    def start_detection(self):
        """Start the detection system."""
        try:
            if self.callbacks['start_detection']:
                success = self.callbacks['start_detection']()
                if success:
                    self.system_status['detection_active'] = True
                    self.start_btn.configure(state=tk.DISABLED)
                    self.stop_btn.configure(state=tk.NORMAL)
                    self.status_text.configure(text="Detection system started")
                    self._update_status_indicators()
                else:
                    messagebox.showerror("Error", "Failed to start detection system")
            else:
                messagebox.showwarning("Warning", "Detection system not initialized")
                
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            messagebox.showerror("Error", f"Failed to start detection: {e}")
    
    def stop_detection(self):
        """Stop the detection system."""
        try:
            if self.callbacks['stop_detection']:
                success = self.callbacks['stop_detection']()
                if success:
                    self.system_status['detection_active'] = False
                    self.start_btn.configure(state=tk.NORMAL)
                    self.stop_btn.configure(state=tk.DISABLED)
                    self.status_text.configure(text="Detection system stopped")
                    self._update_status_indicators()
                else:
                    messagebox.showerror("Error", "Failed to stop detection system")
            else:
                messagebox.showwarning("Warning", "Detection system not initialized")
                
        except Exception as e:
            logger.error(f"Error stopping detection: {e}")
            messagebox.showerror("Error", f"Failed to stop detection: {e}")
    
    def _setup_status_updates(self):
        """Set up automatic status updates."""
        self.status_update_running = True
        self.status_update_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.status_update_thread.start()
    
    def _status_update_loop(self):
        """Background thread for status updates."""
        while self.status_update_running:
            try:
                # Update system status
                if self.callbacks['get_system_status']:
                    status = self.callbacks['get_system_status']()
                    self.system_status.update(status)
                
                # Update performance metrics
                if self.callbacks['get_performance_metrics']:
                    metrics = self.callbacks['get_performance_metrics']()
                    self.system_status['performance_metrics'] = metrics
                
                # Update UI in main thread
                self.root.after(0, self._update_status_indicators)
                self.root.after(0, self._update_performance_display)
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in status update loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _update_status_indicators(self):
        """Update status indicator colors."""
        indicators = {
            'detection': self.system_status.get('detection_active', False),
            'camera': self.system_status.get('camera_connected', False),
            'telegram': self.system_status.get('telegram_connected', False),
            'database': self.system_status.get('database_connected', False)
        }
        
        for key, status in indicators.items():
            if key in self.status_indicators:
                color = "green" if status else "red"
                self.status_indicators[key].configure(foreground=color)
    
    def _update_performance_display(self):
        """Update performance metrics display."""
        metrics = self.system_status.get('performance_metrics', {})
        
        if metrics:
            fps = metrics.get('current_fps', 0)
            cpu = metrics.get('cpu_usage', 0)
            memory = metrics.get('memory_usage', 0)
            
            metrics_text = f"FPS: {fps:.1f} | CPU: {cpu:.1f}% | Memory: {memory:.1f}%"
            self.metrics_text.configure(text=metrics_text)
        else:
            self.metrics_text.configure(text="")
    
    def set_callback(self, callback_name: str, callback_func: Callable):
        """Set a callback function for system integration."""
        if callback_name in self.callbacks:
            self.callbacks[callback_name] = callback_func
            logger.info(f"Set callback: {callback_name}")
        else:
            logger.warning(f"Unknown callback: {callback_name}")

    def set_main_system(self, main_system):
        """Set reference to main system for database access."""
        self.main_system = main_system

        # Update existing modules
        for module_id, module in self.modules.items():
            if hasattr(module, 'set_main_system'):
                module.set_main_system(main_system)
            elif module_id == 'notifications':
                # Special handling for notification center
                if hasattr(module, 'set_database_manager'):
                    module.set_database_manager(main_system.db_manager)
                if hasattr(module, 'set_notification_system'):
                    module.set_notification_system(main_system.notification_system)
                if hasattr(module, 'set_config_manager'):
                    module.set_config_manager(main_system.config_manager)

    def run(self):
        """Start the main application loop."""
        try:
            logger.info("Starting main application")
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources before closing."""
        self.status_update_running = False
        if self.status_update_thread:
            self.status_update_thread.join(timeout=2)
        logger.info("Main window cleanup completed")
