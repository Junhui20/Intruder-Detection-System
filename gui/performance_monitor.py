"""
Performance Monitor Module

Real-time system metrics and performance analysis interface with live data integration.
"""

import tkinter as tk
from tkinter import ttk
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque

# Try to import matplotlib for performance graphs
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Performance graphs will show placeholder.")

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring and metrics display interface with real-time data."""

    def __init__(self, main_system=None):
        self.title = "📊 Performance Monitor"
        self.main_system = main_system
        self.update_interval = 2  # seconds
        self.is_updating = False
        self.update_thread = None

        # GUI components storage
        self.metric_labels = {}
        self.detection_tree = None
        self.accuracy_bars = {}
        self.status_labels = {}
        self.resource_bars = {}

        # Performance graph components
        self.graph_canvas = None
        self.graph_figure = None
        self.graph_axes = {}

        # Data cache
        self.cached_metrics = {}
        self.cached_detections = []
        self.cached_stats = {}

        # Graph data storage (keep last 50 data points for better performance)
        self.graph_data = {
            'timestamps': deque(maxlen=50),
            'fps': deque(maxlen=50),
            'cpu_usage': deque(maxlen=50),
            'memory_usage': deque(maxlen=50),
            'gpu_usage': deque(maxlen=50)
        }

        # Graph update optimization
        self.last_graph_update = 0
        self.graph_update_interval = 5  # Update graphs every 5 seconds instead of every 2
        
    def get_title(self):
        return self.title

    def set_main_system(self, main_system):
        """Set the main system reference for data access."""
        self.main_system = main_system
        logger.info("Main system reference set for Performance Monitor")

    def show_in_frame(self, parent_frame):
        """Display the performance monitor interface with real-time data."""
        # Store reference to root widget for thread-safe updates
        self.root_widget = parent_frame.winfo_toplevel()

        # Main container with notebook
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Store notebook reference and bind tab change events
        self.notebook = notebook

        # Bind to tab selection events to pause/resume updates
        try:
            # Get the parent window to detect when we're not visible
            parent_window = parent_frame.winfo_toplevel()
            parent_window.bind('<FocusOut>', self._on_window_focus_out, add='+')
            parent_window.bind('<FocusIn>', self._on_window_focus_in, add='+')
        except Exception as e:
            logger.debug(f"Could not bind focus events: {e}")

        # Real-time metrics tab
        self._create_realtime_tab(notebook)

        # Detection statistics tab
        self._create_detection_stats_tab(notebook)

        # System health tab
        self._create_system_health_tab(notebook)

        # Start real-time updates
        self.start_updates()

    def start_updates(self):
        """Start real-time data updates."""
        if not self.is_updating and self.main_system:
            self.is_updating = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            logger.info("Performance monitor real-time updates started")

    def stop_updates(self):
        """Stop real-time data updates and cleanup."""
        self.is_updating = False
        if self.update_thread:
            self.update_thread.join(timeout=2)

        # Clear widget references to prevent invalid updates
        self.metric_labels.clear()
        self.accuracy_bars.clear()
        self.status_labels.clear()
        self.resource_bars.clear()
        self.detection_tree = None
        self.graph_canvas = None

        logger.info("Performance monitor real-time updates stopped")

    def _update_loop(self):
        """Background thread for updating GUI with real data."""
        while self.is_updating:
            try:
                if self.main_system:
                    # Update cached data
                    self._update_cached_data()

                    # Update GUI components
                    self._update_gui_components()

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in performance monitor update loop: {e}")
                time.sleep(5)  # Wait longer on error

    def _update_cached_data(self):
        """Update cached data from main system."""
        try:
            # Get current performance metrics
            self.cached_metrics = self.main_system.get_current_performance_metrics()

            # Update graph data
            self._update_graph_data()

            # Get recent detections
            self.cached_detections = self.main_system.get_recent_detections(10)

            # Get detection statistics
            detection_stats = self.main_system.get_detection_stats(7)
            uptime_stats = self.main_system.get_uptime_stats()
            pet_stats = self.main_system.get_pet_identification_stats(7)

            self.cached_stats = {
                **detection_stats,
                **uptime_stats,
                **pet_stats
            }

        except Exception as e:
            logger.error(f"Error updating cached data: {e}")

    def _update_graph_data(self):
        """Update graph data with current metrics."""
        try:
            current_time = datetime.now()
            metrics = self.cached_metrics

            # Add new data points
            self.graph_data['timestamps'].append(current_time)
            self.graph_data['fps'].append(metrics.get('detection_fps', 0))
            self.graph_data['cpu_usage'].append(metrics.get('cpu_usage', 0))
            self.graph_data['memory_usage'].append(metrics.get('memory_usage', 0))
            self.graph_data['gpu_usage'].append(metrics.get('gpu_usage', 0))

        except Exception as e:
            logger.error(f"Error updating graph data: {e}")

    def _update_gui_components(self):
        """Update GUI components with cached data using thread-safe approach."""
        try:
            # Check if we're still updating and components exist
            if not self.is_updating:
                return

            # Schedule GUI updates on the main thread
            if hasattr(self, 'root_widget') and self.root_widget:
                self.root_widget.after_idle(self._safe_update_gui)

        except Exception as e:
            logger.error(f"Error scheduling GUI update: {e}")

    def _safe_update_gui(self):
        """Safely update GUI components on the main thread."""
        try:
            # Check if components still exist before updating
            if not self.is_updating:
                return

            # Update metric labels
            self._safe_update_metric_labels()

            # Update detection tree
            self._safe_update_detection_tree()

            # Update accuracy bars
            self._safe_update_accuracy_bars()

            # Update status labels
            self._safe_update_status_labels()

            # Update resource bars
            self._safe_update_resource_bars()

            # Update performance graph (less frequently to reduce lag)
            current_time = time.time()
            if current_time - self.last_graph_update >= self.graph_update_interval:
                self._update_performance_graph()
                self.last_graph_update = current_time

        except Exception as e:
            logger.debug(f"GUI update skipped (component destroyed): {e}")

    def _create_performance_graph(self, parent_frame):
        """Create matplotlib performance graph."""
        try:
            # Create figure and subplots
            self.graph_figure = Figure(figsize=(12, 6), dpi=80)
            self.graph_figure.patch.set_facecolor('white')

            # Create subplots for different metrics
            gs = self.graph_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # FPS subplot
            self.graph_axes['fps'] = self.graph_figure.add_subplot(gs[0, 0])
            self.graph_axes['fps'].set_title('Detection FPS', fontsize=10, fontweight='bold')
            self.graph_axes['fps'].set_ylabel('FPS')
            self.graph_axes['fps'].grid(True, alpha=0.3)
            self.graph_axes['fps'].set_ylim(0, 60)

            # CPU Usage subplot
            self.graph_axes['cpu'] = self.graph_figure.add_subplot(gs[0, 1])
            self.graph_axes['cpu'].set_title('CPU Usage', fontsize=10, fontweight='bold')
            self.graph_axes['cpu'].set_ylabel('Usage (%)')
            self.graph_axes['cpu'].grid(True, alpha=0.3)
            self.graph_axes['cpu'].set_ylim(0, 100)

            # Memory Usage subplot
            self.graph_axes['memory'] = self.graph_figure.add_subplot(gs[1, 0])
            self.graph_axes['memory'].set_title('Memory Usage', fontsize=10, fontweight='bold')
            self.graph_axes['memory'].set_ylabel('Usage (%)')
            self.graph_axes['memory'].grid(True, alpha=0.3)
            self.graph_axes['memory'].set_ylim(0, 100)

            # GPU Usage subplot
            self.graph_axes['gpu'] = self.graph_figure.add_subplot(gs[1, 1])
            self.graph_axes['gpu'].set_title('GPU Usage', fontsize=10, fontweight='bold')
            self.graph_axes['gpu'].set_ylabel('Usage (%)')
            self.graph_axes['gpu'].grid(True, alpha=0.3)
            self.graph_axes['gpu'].set_ylim(0, 100)

            # Create canvas
            self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, parent_frame)
            self.graph_canvas.draw()
            self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Initialize empty plots
            self._initialize_empty_plots()

        except Exception as e:
            logger.error(f"Error creating performance graph: {e}")
            # Fallback to placeholder
            placeholder = ttk.Label(parent_frame, text=f"Graph creation failed: {e}")
            placeholder.pack(expand=True, fill=tk.BOTH)

    def _initialize_empty_plots(self):
        """Initialize empty plot lines."""
        try:
            # Initialize plot lines
            self.plot_lines = {}

            # FPS line
            self.plot_lines['fps'], = self.graph_axes['fps'].plot([], [], 'b-', linewidth=2, label='FPS')

            # CPU line
            self.plot_lines['cpu'], = self.graph_axes['cpu'].plot([], [], 'r-', linewidth=2, label='CPU %')

            # Memory line
            self.plot_lines['memory'], = self.graph_axes['memory'].plot([], [], 'g-', linewidth=2, label='Memory %')

            # GPU line
            self.plot_lines['gpu'], = self.graph_axes['gpu'].plot([], [], 'm-', linewidth=2, label='GPU %')

            # Set up time formatting for x-axis with proper tick control
            for ax in self.graph_axes.values():
                # Use MaxNLocator for better control over number of ticks
                from matplotlib.ticker import MaxNLocator

                # Set up time formatting
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

                # Use MaxNLocator instead of SecondLocator to control tick count
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))

                # Set y-axis locator to prevent too many ticks
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

            self.graph_canvas.draw()

        except Exception as e:
            logger.error(f"Error initializing empty plots: {e}")

    def _create_simple_graph_placeholder(self, parent_frame):
        """Create a simple text-based graph placeholder."""
        try:
            # Create a frame for the simple graph display
            simple_frame = ttk.Frame(parent_frame)
            simple_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            title_label = ttk.Label(simple_frame, text="📊 Performance Metrics", font=("Arial", 14, "bold"))
            title_label.pack(pady=10)

            # Create text display for metrics
            self.simple_graph_text = tk.Text(simple_frame, height=15, font=("Courier", 10))
            scrollbar = ttk.Scrollbar(simple_frame, orient=tk.VERTICAL, command=self.simple_graph_text.yview)
            self.simple_graph_text.configure(yscrollcommand=scrollbar.set)

            self.simple_graph_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Initial text
            self.simple_graph_text.insert(tk.END, "📈 Real-time Performance Data\n")
            self.simple_graph_text.insert(tk.END, "=" * 40 + "\n\n")
            self.simple_graph_text.insert(tk.END, "Waiting for data...\n")
            self.simple_graph_text.configure(state=tk.DISABLED)

            # Store reference for updates
            self.graph_canvas = None  # No matplotlib canvas

        except Exception as e:
            logger.error(f"Error creating simple graph placeholder: {e}")
            # Ultimate fallback
            fallback_label = ttk.Label(parent_frame, text="📊 Performance monitoring active\n(Graph display unavailable)")
            fallback_label.pack(expand=True, fill=tk.BOTH)
    
    def _create_realtime_tab(self, notebook):
        """Create real-time metrics tab with live data."""
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="⚡ Real-time")

        # Metrics grid
        metrics_frame = ttk.LabelFrame(realtime_frame, text="Current Performance", padding=20)
        metrics_frame.pack(fill=tk.X, padx=20, pady=20)

        # Define metric configurations
        metric_configs = [
            ("detection_fps", "🎯 Detection FPS", "fps"),
            ("face_recognition_fps", "🎭 Face Recognition", "fps"),
            ("animal_id_fps", "🐕 Animal ID", "fps"),
            ("processing_time", "⏱️ Processing Time", "ms"),
            ("cpu_usage", "🧠 CPU Usage", "%"),
            ("memory_usage_mb", "💾 Memory Usage", "MB"),
            ("gpu_usage", "🎮 GPU Usage", "%"),
            ("gpu_temperature", "🌡️ GPU Temp", "°C")
        ]

        # Create metric displays in grid
        for i, (metric_key, label, unit) in enumerate(metric_configs):
            row = i // 4
            col = i % 4

            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=row, column=col, padx=10, pady=10, sticky="ew")

            ttk.Label(metric_frame, text=label, font=("Arial", 10)).pack()

            # Create value label and store reference
            value_label = ttk.Label(metric_frame, text="--", font=("Arial", 16, "bold"), foreground="blue")
            value_label.pack()

            ttk.Label(metric_frame, text=unit, font=("Arial", 9), foreground="gray").pack()

            # Store reference for updates
            self.metric_labels[metric_key] = value_label

        # Configure grid weights
        for i in range(4):
            metrics_frame.columnconfigure(i, weight=1)
        
        # Performance graph
        graph_frame = ttk.LabelFrame(realtime_frame, text="Performance Graph", padding=20)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Create performance graph with fallback options
        if MATPLOTLIB_AVAILABLE:
            try:
                self._create_performance_graph(graph_frame)
            except Exception as e:
                logger.warning(f"Failed to create matplotlib graph: {e}")
                self._create_simple_graph_placeholder(graph_frame)
        else:
            self._create_simple_graph_placeholder(graph_frame)
        
        # Controls
        controls_frame = ttk.Frame(graph_frame)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(controls_frame, text="⏸️ Pause").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="🔄 Reset").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="💾 Export").pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="📊 DB Metrics",
                  command=self.show_database_metrics).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(controls_frame, text="Update Interval:").pack(side=tk.RIGHT, padx=(10, 5))
        interval_combo = ttk.Combobox(controls_frame, values=["1s", "2s", "5s"], width=8)
        interval_combo.set("2s")
        interval_combo.pack(side=tk.RIGHT)
    
    def _create_detection_stats_tab(self, notebook):
        """Create detection statistics tab."""
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📊 Detection Stats")
        
        # Summary statistics
        summary_frame = ttk.LabelFrame(stats_frame, text="Detection Summary", padding=20)
        summary_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Statistics in columns
        left_stats = ttk.Frame(summary_frame)
        left_stats.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        right_stats = ttk.Frame(summary_frame)
        right_stats.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Define stat configurations for left column
        left_stat_configs = [
            ("total_detections", "Total Detections"),
            ("human_detections", "Human Detections"),
            ("animal_detections", "Animal Detections"),
            ("known_humans", "Known Humans"),
            ("unknown_humans", "Unknown Humans")
        ]

        # Create left column stats with stored references
        for stat_key, label in left_stat_configs:
            stat_frame = ttk.Frame(left_stats)
            stat_frame.pack(fill=tk.X, pady=2)
            ttk.Label(stat_frame, text=f"{label}:", width=20).pack(side=tk.LEFT)

            value_label = ttk.Label(stat_frame, text="--", font=("Arial", 10, "bold"))
            value_label.pack(side=tk.LEFT)
            self.metric_labels[stat_key] = value_label

        # Define stat configurations for right column
        right_stat_configs = [
            ("pet_identifications", "Pet Identifications"),
            ("unknown_animals", "Unknown Animals"),
            ("notifications_sent", "Notifications Sent"),
            ("avg_confidence", "Average Confidence"),
            ("uptime", "Uptime")
        ]

        # Create right column stats with stored references
        for stat_key, label in right_stat_configs:
            stat_frame = ttk.Frame(right_stats)
            stat_frame.pack(fill=tk.X, pady=2)
            ttk.Label(stat_frame, text=f"{label}:", width=20).pack(side=tk.LEFT)

            value_label = ttk.Label(stat_frame, text="--", font=("Arial", 10, "bold"))
            value_label.pack(side=tk.LEFT)
            self.metric_labels[stat_key] = value_label
        
        # Accuracy metrics
        accuracy_frame = ttk.LabelFrame(stats_frame, text="Accuracy Metrics", padding=20)
        accuracy_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        # Progress bars for accuracy with stored references
        accuracy_configs = [
            ("face_recognition_accuracy", "Face Recognition Accuracy"),
            ("pet_identification_accuracy", "Pet Identification Accuracy"),
            ("animal_detection_accuracy", "Animal Detection Accuracy"),
            ("overall_system_accuracy", "Overall System Accuracy")
        ]

        for accuracy_key, label in accuracy_configs:
            label_widget = ttk.Label(accuracy_frame, text=f"{label}: --%")
            label_widget.pack(anchor=tk.W)

            progress = ttk.Progressbar(accuracy_frame, length=300, mode='determinate')
            progress['value'] = 0
            progress.pack(fill=tk.X, pady=(0, 10))

            # Store references for updates
            self.accuracy_bars[accuracy_key] = {
                'label': label_widget,
                'progress': progress,
                'base_text': label
            }
        
        # Recent detections log
        log_frame = ttk.LabelFrame(stats_frame, text="Recent Detections", padding=20)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Treeview for detection log
        columns = ("Time", "Type", "Entity", "Confidence", "Action")
        tree = ttk.Treeview(log_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Store tree reference for updates
        self.detection_tree = tree

        # Scrollbar for tree
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_system_health_tab(self, notebook):
        """Create system health tab."""
        health_frame = ttk.Frame(notebook)
        notebook.add(health_frame, text="🏥 System Health")
        
        # System status indicators
        status_frame = ttk.LabelFrame(health_frame, text="System Status", padding=20)
        status_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Status indicators in grid with real data
        status_configs = [
            ("detection_engine", "🔍 Detection Engine"),
            ("camera_connection", "📹 Camera Connection"),
            ("telegram_bot", "📱 Telegram Bot"),
            ("database", "💾 Database"),
            ("gpu_acceleration", "🎮 GPU Acceleration"),
            ("performance_monitoring", "📊 Performance Monitor"),
            ("storage_space", "💾 Storage Space"),
            ("temperature", "🌡️ Temperature")
        ]

        for i, (status_key, component) in enumerate(status_configs):
            row = i // 4
            col = i % 4

            status_item = ttk.Frame(status_frame)
            status_item.grid(row=row, column=col, padx=10, pady=5, sticky="w")

            ttk.Label(status_item, text=component, font=("Arial", 9)).pack(anchor=tk.W)

            status_label = ttk.Label(status_item, text="--", font=("Arial", 9, "bold"))
            status_label.pack(anchor=tk.W)

            # Store reference for updates
            self.status_labels[status_key] = status_label
        
        # Configure grid weights
        for i in range(4):
            status_frame.columnconfigure(i, weight=1)
        
        # Resource usage
        resources_frame = ttk.LabelFrame(health_frame, text="Resource Usage", padding=20)
        resources_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Resource bars with real data
        resource_configs = [
            ("cpu_usage", "CPU Usage", 80),
            ("memory_usage", "Memory Usage", 90),
            ("gpu_usage", "GPU Usage", 85),
            ("storage_usage", "Storage Usage", 90)
        ]

        for resource_key, name, warning_threshold in resource_configs:
            resource_frame = ttk.Frame(resources_frame)
            resource_frame.pack(fill=tk.X, pady=5)

            label = ttk.Label(resource_frame, text=f"{name}: --%", width=20)
            label.pack(side=tk.LEFT)

            progress = ttk.Progressbar(resource_frame, length=200, mode='determinate')
            progress['value'] = 0
            progress.pack(side=tk.LEFT, padx=(10, 0))

            warning_label = ttk.Label(resource_frame, text="--")
            warning_label.pack(side=tk.LEFT, padx=(5, 0))

            # Store references for updates
            self.resource_bars[resource_key] = {
                'label': label,
                'progress': progress,
                'warning': warning_label,
                'threshold': warning_threshold,
                'name': name
            }
        
        # Recommendations
        recommendations_frame = ttk.LabelFrame(health_frame, text="Optimization Recommendations", padding=20)
        recommendations_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        recommendations_text = tk.Text(recommendations_frame, height=8, font=("Arial", 10))
        recommendations_scrollbar = ttk.Scrollbar(recommendations_frame, orient=tk.VERTICAL, command=recommendations_text.yview)
        recommendations_text.configure(yscrollcommand=recommendations_scrollbar.set)
        
        recommendations_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        recommendations_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sample recommendations
        sample_recommendations = """✅ System Performance: Excellent
• Detection FPS is above target (45.2 > 30 FPS)
• Face recognition performance is optimal
• GPU utilization is within normal range

⚠️ Storage Warning:
• Storage is 78% full - consider cleaning old detection logs
• Recommend enabling automatic log rotation

💡 Optimization Suggestions:
• Consider increasing confidence thresholds to reduce false positives
• GPU temperature is normal but monitor during extended use
• Network latency to IP cameras is acceptable

🔧 Maintenance:
• Last database optimization: 2 days ago
• Face encodings cache: Healthy
• Model files: Up to date
"""
        recommendations_text.insert(tk.END, sample_recommendations)
        recommendations_text.configure(state=tk.DISABLED)

    def set_main_system(self, main_system):
        """Set reference to main system for database access."""
        self.main_system = main_system

    def show_database_metrics(self):
        """Show database performance metrics in a new window."""
        try:
            # Create new window
            metrics_window = tk.Toplevel()
            metrics_window.title("📊 Database Performance Metrics")
            metrics_window.geometry("700x500")

            # Create notebook for different metric categories
            notebook = ttk.Notebook(metrics_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Performance metrics tab
            perf_frame = ttk.Frame(notebook)
            notebook.add(perf_frame, text="📈 Performance")

            # Create text widget for performance metrics
            perf_text = tk.Text(perf_frame, font=("Courier", 10))
            perf_scrollbar = ttk.Scrollbar(perf_frame, orient=tk.VERTICAL, command=perf_text.yview)
            perf_text.configure(yscrollcommand=perf_scrollbar.set)

            perf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Database statistics tab
            db_frame = ttk.Frame(notebook)
            notebook.add(db_frame, text="🗄️ Database")

            # Create text widget for database stats
            db_text = tk.Text(db_frame, font=("Courier", 10))
            db_scrollbar = ttk.Scrollbar(db_frame, orient=tk.VERTICAL, command=db_text.yview)
            db_text.configure(yscrollcommand=db_scrollbar.set)

            db_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            db_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Get data from main system
            if hasattr(self, 'main_system') and self.main_system:
                # Performance metrics
                metric_averages = self.main_system.get_metric_averages(24)
                recent_metrics = self.main_system.get_system_metrics('fps', 20)

                perf_text.insert(tk.END, "📈 PERFORMANCE METRICS (24h Average)\n")
                perf_text.insert(tk.END, "=" * 50 + "\n\n")

                if metric_averages:
                    for metric, value in metric_averages.items():
                        unit = "fps" if metric == "fps" else "%" if "usage" in metric else "ms" if "time" in metric else ""
                        perf_text.insert(tk.END, f"{metric.replace('_', ' ').title()}: {value}{unit}\n")
                else:
                    perf_text.insert(tk.END, "No performance metrics available\n")

                if recent_metrics:
                    perf_text.insert(tk.END, f"\n📊 RECENT FPS METRICS ({len(recent_metrics)} samples)\n")
                    perf_text.insert(tk.END, "-" * 30 + "\n")

                    for metric in recent_metrics[-10:]:  # Show last 10
                        timestamp = metric.recorded_at.strftime("%H:%M:%S") if metric.recorded_at else "Unknown"
                        perf_text.insert(tk.END, f"{timestamp}: {metric.metric_value:.1f} fps\n")

                # Database statistics
                db_stats = self.main_system.db_manager.get_database_stats() if self.main_system.db_manager else {}
                detection_stats = self.main_system.get_detection_stats(7)

                db_text.insert(tk.END, "🗄️ DATABASE STATISTICS\n")
                db_text.insert(tk.END, "=" * 40 + "\n\n")

                db_text.insert(tk.END, "📊 Table Record Counts:\n")
                for table, count in db_stats.items():
                    table_name = table.replace('_count', '').replace('_', ' ').title()
                    db_text.insert(tk.END, f"  • {table_name}: {count}\n")

                db_text.insert(tk.END, "\n📅 Detection Summary (7 days):\n")
                if detection_stats:
                    for detection_type, count in detection_stats.items():
                        db_text.insert(tk.END, f"  • {detection_type.title()}: {count}\n")
                else:
                    db_text.insert(tk.END, "  No detections in last 7 days\n")

            else:
                perf_text.insert(tk.END, "Main system not available\n")
                db_text.insert(tk.END, "Database connection not available\n")

            perf_text.configure(state=tk.DISABLED)
            db_text.configure(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Error showing database metrics: {e}")

    def _update_metric_labels(self):
        """Update metric labels with current data."""
        try:
            # Update real-time performance metrics
            metrics = self.cached_metrics

            # Map cached metrics to GUI labels
            metric_mappings = {
                'detection_fps': metrics.get('detection_fps', 0),
                'face_recognition_fps': metrics.get('face_recognition_fps', 0),
                'animal_id_fps': metrics.get('animal_id_fps', 0),
                'processing_time': metrics.get('processing_time', 0),
                'cpu_usage': metrics.get('cpu_usage', 0),
                'memory_usage_mb': metrics.get('memory_usage_mb', 0),
                'gpu_usage': metrics.get('gpu_usage', 0),
                'gpu_temperature': metrics.get('gpu_temperature', 0)
            }

            # Update metric labels
            for metric_key, value in metric_mappings.items():
                if metric_key in self.metric_labels:
                    formatted_value = f"{value:.1f}" if isinstance(value, (int, float)) else str(value)
                    self.metric_labels[metric_key].config(text=formatted_value)

            # Update detection statistics
            stats = self.cached_stats

            # Calculate totals
            human_count = stats.get('human', 0)
            animal_count = stats.get('animal', 0)
            total_detections = human_count + animal_count

            stat_mappings = {
                'total_detections': f"{total_detections:,}",
                'human_detections': f"{human_count:,}",
                'animal_detections': f"{animal_count:,}",
                'pet_identifications': f"{stats.get('pet_identifications', 0):,}",
                'unknown_animals': f"{stats.get('unknown_animals', 0):,}",
                'notifications_sent': f"{stats.get('notifications_sent', 0):,}",
                'avg_confidence': f"{stats.get('avg_confidence', 0):.1f}%"
            }

            # Calculate uptime
            if 'first_detection' in stats and stats['first_detection']:
                try:
                    first_time = datetime.fromisoformat(stats['first_detection'].replace('Z', '+00:00'))
                    uptime_delta = datetime.now() - first_time.replace(tzinfo=None)
                    days = uptime_delta.days
                    hours, remainder = divmod(uptime_delta.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    stat_mappings['uptime'] = f"{days}d {hours}h {minutes}m"
                except:
                    stat_mappings['uptime'] = "Unknown"
            else:
                stat_mappings['uptime'] = "No data"

            # Update stat labels
            for stat_key, value in stat_mappings.items():
                if stat_key in self.metric_labels:
                    self.metric_labels[stat_key].config(text=value)

        except Exception as e:
            logger.error(f"Error updating metric labels: {e}")

    def _safe_update_metric_labels(self):
        """Safely update metric labels with existence checks."""
        try:
            # Update real-time performance metrics
            metrics = self.cached_metrics

            # Map cached metrics to GUI labels
            metric_mappings = {
                'detection_fps': metrics.get('detection_fps', 0),
                'face_recognition_fps': metrics.get('face_recognition_fps', 0),
                'animal_id_fps': metrics.get('animal_id_fps', 0),
                'processing_time': metrics.get('processing_time', 0),
                'cpu_usage': metrics.get('cpu_usage', 0),
                'memory_usage_mb': metrics.get('memory_usage_mb', 0),
                'gpu_usage': metrics.get('gpu_usage', 0),
                'gpu_temperature': metrics.get('gpu_temperature', 0)
            }

            # Update metric labels with existence check
            for metric_key, value in metric_mappings.items():
                if metric_key in self.metric_labels:
                    try:
                        label = self.metric_labels[metric_key]
                        if label.winfo_exists():
                            formatted_value = f"{value:.1f}" if isinstance(value, (int, float)) else str(value)
                            label.config(text=formatted_value)
                    except tk.TclError:
                        # Widget was destroyed, remove from our tracking
                        del self.metric_labels[metric_key]

            # Update detection statistics
            stats = self.cached_stats

            # Calculate totals
            human_count = stats.get('human', 0)
            animal_count = stats.get('animal', 0)
            total_detections = human_count + animal_count

            stat_mappings = {
                'total_detections': f"{total_detections:,}",
                'human_detections': f"{human_count:,}",
                'animal_detections': f"{animal_count:,}",
                'pet_identifications': f"{stats.get('pet_identifications', 0):,}",
                'unknown_animals': f"{stats.get('unknown_animals', 0):,}",
                'notifications_sent': f"{stats.get('notifications_sent', 0):,}",
                'avg_confidence': f"{stats.get('avg_confidence', 0):.1f}%"
            }

            # Calculate uptime
            if 'first_detection' in stats and stats['first_detection']:
                try:
                    first_time = datetime.fromisoformat(stats['first_detection'].replace('Z', '+00:00'))
                    uptime_delta = datetime.now() - first_time.replace(tzinfo=None)
                    days = uptime_delta.days
                    hours, remainder = divmod(uptime_delta.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    stat_mappings['uptime'] = f"{days}d {hours}h {minutes}m"
                except:
                    stat_mappings['uptime'] = "Unknown"
            else:
                stat_mappings['uptime'] = "No data"

            # Update stat labels with existence check
            for stat_key, value in stat_mappings.items():
                if stat_key in self.metric_labels:
                    try:
                        label = self.metric_labels[stat_key]
                        if label.winfo_exists():
                            label.config(text=value)
                    except tk.TclError:
                        # Widget was destroyed, remove from our tracking
                        del self.metric_labels[stat_key]

        except Exception as e:
            logger.debug(f"Safe metric update skipped: {e}")

    def _update_detection_tree(self):
        """Update detection tree with recent detections."""
        try:
            if not self.detection_tree:
                return

            # Clear existing items
            for item in self.detection_tree.get_children():
                self.detection_tree.delete(item)

            # Add recent detections
            for detection in self.cached_detections:
                try:
                    # Format detection time
                    if detection.detected_at:
                        if isinstance(detection.detected_at, str):
                            time_str = detection.detected_at.split(' ')[1][:8]  # Extract time part
                        else:
                            time_str = detection.detected_at.strftime("%H:%M:%S")
                    else:
                        time_str = "Unknown"

                    # Format confidence
                    confidence_str = f"{detection.confidence:.1f}%" if detection.confidence else "N/A"

                    # Determine action
                    if detection.notification_sent:
                        action = "Notified"
                    elif detection.entity_name and detection.entity_name != "Unknown":
                        action = "Recognized" if detection.detection_type == "human" else "Pet ID"
                    else:
                        action = "Alert Sent"

                    # Insert into tree
                    self.detection_tree.insert("", tk.END, values=(
                        time_str,
                        detection.detection_type.title(),
                        detection.entity_name or "Unknown",
                        confidence_str,
                        action
                    ))

                except Exception as e:
                    logger.error(f"Error formatting detection: {e}")

        except Exception as e:
            logger.error(f"Error updating detection tree: {e}")

    def _safe_update_detection_tree(self):
        """Safely update detection tree with existence checks."""
        try:
            if not self.detection_tree:
                return

            # Check if tree still exists
            if not self.detection_tree.winfo_exists():
                self.detection_tree = None
                return

            # Clear existing items
            for item in self.detection_tree.get_children():
                self.detection_tree.delete(item)

            # Add recent detections
            for detection in self.cached_detections:
                try:
                    # Format detection time
                    if detection.detected_at:
                        if isinstance(detection.detected_at, str):
                            time_str = detection.detected_at.split(' ')[1][:8]  # Extract time part
                        else:
                            time_str = detection.detected_at.strftime("%H:%M:%S")
                    else:
                        time_str = "Unknown"

                    # Format confidence
                    confidence_str = f"{detection.confidence:.1f}%" if detection.confidence else "N/A"

                    # Determine action
                    if detection.notification_sent:
                        action = "Notified"
                    elif detection.entity_name and detection.entity_name != "Unknown":
                        action = "Recognized" if detection.detection_type == "human" else "Pet ID"
                    else:
                        action = "Alert Sent"

                    # Insert into tree
                    self.detection_tree.insert("", tk.END, values=(
                        time_str,
                        detection.detection_type.title(),
                        detection.entity_name or "Unknown",
                        confidence_str,
                        action
                    ))

                except Exception as e:
                    logger.debug(f"Error formatting detection: {e}")

        except tk.TclError:
            # Tree was destroyed
            self.detection_tree = None
        except Exception as e:
            logger.debug(f"Safe detection tree update skipped: {e}")

    def _update_accuracy_bars(self):
        """Update accuracy progress bars."""
        try:
            # Mock accuracy data - in real implementation, calculate from detection results
            accuracy_data = {
                'face_recognition_accuracy': 95.2,
                'pet_identification_accuracy': 87.8,
                'animal_detection_accuracy': 92.1,
                'overall_system_accuracy': 91.7
            }

            for accuracy_key, percentage in accuracy_data.items():
                if accuracy_key in self.accuracy_bars:
                    bar_info = self.accuracy_bars[accuracy_key]
                    bar_info['label'].config(text=f"{bar_info['base_text']}: {percentage:.1f}%")
                    bar_info['progress']['value'] = percentage

        except Exception as e:
            logger.error(f"Error updating accuracy bars: {e}")

    def _safe_update_accuracy_bars(self):
        """Safely update accuracy progress bars with existence checks."""
        try:
            # Mock accuracy data - in real implementation, calculate from detection results
            accuracy_data = {
                'face_recognition_accuracy': 95.2,
                'pet_identification_accuracy': 87.8,
                'animal_detection_accuracy': 92.1,
                'overall_system_accuracy': 91.7
            }

            for accuracy_key, percentage in accuracy_data.items():
                if accuracy_key in self.accuracy_bars:
                    try:
                        bar_info = self.accuracy_bars[accuracy_key]
                        if bar_info['label'].winfo_exists() and bar_info['progress'].winfo_exists():
                            bar_info['label'].config(text=f"{bar_info['base_text']}: {percentage:.1f}%")
                            bar_info['progress']['value'] = percentage
                    except tk.TclError:
                        # Widget was destroyed, remove from tracking
                        del self.accuracy_bars[accuracy_key]

        except Exception as e:
            logger.debug(f"Safe accuracy bars update skipped: {e}")

    def _safe_update_status_labels(self):
        """Safely update system status labels with existence checks."""
        try:
            if not self.main_system:
                return

            status = self.main_system.get_system_health_status()

            for status_key, value in status.items():
                if status_key in self.status_labels:
                    try:
                        label = self.status_labels[status_key]
                        if label.winfo_exists():
                            # Determine color based on status
                            color = self._get_status_color(value)

                            # Add status icon
                            if 'Running' in value or 'Connected' in value or 'Active' in value or 'Enabled' in value or 'Normal' in value:
                                display_text = f"✅ {value}"
                            elif 'Warning' in value or 'Low' in value or 'High' in value:
                                display_text = f"⚠️ {value}"
                            elif 'Stopped' in value or 'Disconnected' in value or 'Inactive' in value or 'Disabled' in value:
                                display_text = f"❌ {value}"
                            else:
                                display_text = f"ℹ️ {value}"

                            label.config(text=display_text, foreground=color)
                    except tk.TclError:
                        # Widget was destroyed, remove from tracking
                        del self.status_labels[status_key]

        except Exception as e:
            logger.debug(f"Safe status labels update skipped: {e}")

    def _safe_update_resource_bars(self):
        """Safely update resource usage bars with existence checks."""
        try:
            metrics = self.cached_metrics

            resource_data = {
                'cpu_usage': metrics.get('cpu_usage', 0),
                'memory_usage': metrics.get('memory_usage', 0),
                'gpu_usage': metrics.get('gpu_usage', 0),
                'storage_usage': metrics.get('storage_usage', 0)
            }

            for resource_key, percentage in resource_data.items():
                if resource_key in self.resource_bars:
                    try:
                        bar_info = self.resource_bars[resource_key]

                        # Check if widgets still exist
                        if (bar_info['label'].winfo_exists() and
                            bar_info['progress'].winfo_exists() and
                            bar_info['warning'].winfo_exists()):

                            # Update label
                            bar_info['label'].config(text=f"{bar_info['name']}: {percentage:.1f}%")

                            # Update progress bar
                            bar_info['progress']['value'] = percentage

                            # Update warning indicator
                            if percentage > bar_info['threshold']:
                                bar_info['warning'].config(text="⚠️", foreground="orange")
                            else:
                                bar_info['warning'].config(text="✅", foreground="green")
                    except tk.TclError:
                        # Widget was destroyed, remove from tracking
                        del self.resource_bars[resource_key]

        except Exception as e:
            logger.debug(f"Safe resource bars update skipped: {e}")

    def _update_status_labels(self):
        """Update system status labels with color coding."""
        try:
            if not self.main_system:
                return

            status = self.main_system.get_system_health_status()

            for status_key, value in status.items():
                if status_key in self.status_labels:
                    # Determine color based on status
                    color = self._get_status_color(value)

                    # Add status icon
                    if 'Running' in value or 'Connected' in value or 'Active' in value or 'Enabled' in value or 'Normal' in value:
                        display_text = f"[OK] {value}"
                    elif 'Warning' in value or 'Low' in value or 'High' in value:
                        display_text = f"[WARN] {value}"
                    elif 'Stopped' in value or 'Disconnected' in value or 'Inactive' in value or 'Disabled' in value:
                        display_text = f"[ERROR] {value}"
                    else:
                        display_text = f"[INFO] {value}"

                    self.status_labels[status_key].config(text=display_text, foreground=color)

        except Exception as e:
            logger.error(f"Error updating status labels: {e}")

    def _get_status_color(self, status_value: str) -> str:
        """Get color for status value."""
        if any(word in status_value for word in ['Running', 'Connected', 'Active', 'Enabled', 'Normal']):
            return 'green'
        elif any(word in status_value for word in ['Warning', 'Low', 'High']):
            return 'orange'
        elif any(word in status_value for word in ['Stopped', 'Disconnected', 'Inactive', 'Disabled']):
            return 'red'
        else:
            return 'black'

    def _update_resource_bars(self):
        """Update resource usage bars."""
        try:
            metrics = self.cached_metrics

            resource_data = {
                'cpu_usage': metrics.get('cpu_usage', 0),
                'memory_usage': metrics.get('memory_usage', 0),
                'gpu_usage': metrics.get('gpu_usage', 0),
                'storage_usage': metrics.get('storage_usage', 0)
            }

            for resource_key, percentage in resource_data.items():
                if resource_key in self.resource_bars:
                    bar_info = self.resource_bars[resource_key]

                    # Update label
                    bar_info['label'].config(text=f"{bar_info['name']}: {percentage:.1f}%")

                    # Update progress bar
                    bar_info['progress']['value'] = percentage

                    # Update warning indicator
                    if percentage > bar_info['threshold']:
                        bar_info['warning'].config(text="⚠️", foreground="orange")
                    else:
                        bar_info['warning'].config(text="✅", foreground="green")

        except Exception as e:
            logger.error(f"Error updating resource bars: {e}")

    def _update_performance_graph(self):
        """Update the performance graph with new data."""
        try:
            # Check if we have matplotlib graph or simple text display
            if hasattr(self, 'simple_graph_text'):
                self._update_simple_graph_display()
                return

            if not MATPLOTLIB_AVAILABLE or not self.graph_canvas or not self.plot_lines:
                return

            # Get data for plotting
            timestamps = list(self.graph_data['timestamps'])
            fps_data = list(self.graph_data['fps'])
            cpu_data = list(self.graph_data['cpu_usage'])
            memory_data = list(self.graph_data['memory_usage'])
            gpu_data = list(self.graph_data['gpu_usage'])

            if not timestamps:
                return

            # Limit data points to prevent performance issues
            # Only show last 50 points to keep graphs responsive
            max_points = 50
            if len(timestamps) > max_points:
                timestamps = timestamps[-max_points:]
                fps_data = fps_data[-max_points:]
                cpu_data = cpu_data[-max_points:]
                memory_data = memory_data[-max_points:]
                gpu_data = gpu_data[-max_points:]

            # Update plot lines
            self.plot_lines['fps'].set_data(timestamps, fps_data)
            self.plot_lines['cpu'].set_data(timestamps, cpu_data)
            self.plot_lines['memory'].set_data(timestamps, memory_data)
            self.plot_lines['gpu'].set_data(timestamps, gpu_data)

            # Update axis limits with better control
            if len(timestamps) > 1:
                time_range = timestamps[-1] - timestamps[0]
                if time_range.total_seconds() > 0:
                    # Set x-axis to show last 5 minutes of data
                    end_time = timestamps[-1]
                    start_time = end_time - timedelta(minutes=5)

                    for ax_name, ax in self.graph_axes.items():
                        ax.set_xlim(start_time, end_time)

                        # Set appropriate y-axis limits based on data
                        if ax_name == 'fps':
                            ax.set_ylim(0, max(60, max(fps_data) * 1.1) if fps_data else 60)
                        else:  # CPU, Memory, GPU usage
                            ax.set_ylim(0, 100)

                        # Refresh the axis with controlled ticks
                        ax.relim()
                        ax.autoscale_view(scalex=False, scaley=False)

                        # The tick locators are already set in _initialize_empty_plots()
                        # No need to reset them here

            # Redraw canvas efficiently
            try:
                self.graph_canvas.draw_idle()
            except Exception as draw_error:
                # If drawing fails, skip this update to prevent lag
                logger.debug(f"Graph draw skipped due to error: {draw_error}")

        except Exception as e:
            logger.error(f"Error updating performance graph: {e}")

    def _update_simple_graph_display(self):
        """Update simple text-based graph display."""
        try:
            if not hasattr(self, 'simple_graph_text'):
                return

            # Get current metrics
            metrics = self.cached_metrics

            # Create text display
            display_text = "📈 Real-time Performance Data\n"
            display_text += "=" * 40 + "\n"
            display_text += f"⏰ Updated: {datetime.now().strftime('%H:%M:%S')}\n\n"

            # Current metrics
            display_text += "🎯 Current Metrics:\n"
            display_text += f"  Detection FPS:     {metrics.get('detection_fps', 0):.1f}\n"
            display_text += f"  CPU Usage:         {metrics.get('cpu_usage', 0):.1f}%\n"
            display_text += f"  Memory Usage:      {metrics.get('memory_usage_mb', 0):.0f} MB\n"
            display_text += f"  GPU Usage:         {metrics.get('gpu_usage', 0):.1f}%\n"
            display_text += f"  Processing Time:   {metrics.get('processing_time', 0):.1f} ms\n\n"

            # Recent trends (last 10 data points)
            display_text += "📊 Recent Trends (last 10 readings):\n"

            fps_history = list(self.graph_data['fps'])[-10:]
            cpu_history = list(self.graph_data['cpu_usage'])[-10:]

            if fps_history:
                display_text += f"  FPS Trend:    {self._create_simple_sparkline(fps_history)}\n"
            if cpu_history:
                display_text += f"  CPU Trend:    {self._create_simple_sparkline(cpu_history)}\n"

            display_text += "\n📈 Graph display using text mode (matplotlib issues detected)\n"
            display_text += "   For full graphs, restart the application\n"

            # Update text widget
            self.simple_graph_text.configure(state=tk.NORMAL)
            self.simple_graph_text.delete(1.0, tk.END)
            self.simple_graph_text.insert(tk.END, display_text)
            self.simple_graph_text.configure(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Error updating simple graph display: {e}")

    def _create_simple_sparkline(self, data):
        """Create a simple ASCII sparkline from data."""
        if not data or len(data) < 2:
            return "No data"

        # Normalize data to 0-7 range for ASCII characters
        min_val = min(data)
        max_val = max(data)

        if max_val == min_val:
            return "-" * len(data)  # Flat line

        # ASCII characters for different heights
        chars = " .:-=+*#%@"

        sparkline = ""
        for value in data:
            normalized = (value - min_val) / (max_val - min_val)
            char_index = int(normalized * (len(chars) - 1))
            sparkline += chars[char_index]

        return f"{sparkline} ({min_val:.1f}-{max_val:.1f})"

    def _on_window_focus_out(self, event=None):
        """Handle window losing focus - reduce update frequency."""
        try:
            # Reduce update frequency when window is not focused
            self.update_interval = 10  # Update every 10 seconds when not focused
        except Exception as e:
            logger.debug(f"Focus out handler error: {e}")

    def _on_window_focus_in(self, event=None):
        """Handle window gaining focus - restore normal update frequency."""
        try:
            # Restore normal update frequency when window is focused
            self.update_interval = 2  # Back to 2 seconds when focused
        except Exception as e:
            logger.debug(f"Focus in handler error: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_updates()
