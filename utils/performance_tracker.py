"""
Performance Tracking and Metrics Collection

This module provides comprehensive performance monitoring and analysis.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime
import logging

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Comprehensive performance tracking system.
    
    Features:
    - Real-time metrics collection
    - FPS tracking
    - Resource utilization monitoring
    - Performance analysis and recommendations
    """
    
    def __init__(self, max_history: int = 1000, db_manager=None):
        """
        Initialize performance tracker.

        Args:
            max_history: Maximum number of metrics to keep in history
            db_manager: Database manager for metrics logging
        """
        self.max_history = max_history
        self.db_manager = db_manager
        self.metrics_history = {}
        self.timers = {}
        self.counters = {}

        # Performance metrics
        self.fps_tracker = FPSTracker()
        self.resource_monitor = ResourceMonitor()

        # Metrics logging interval (seconds)
        self.metrics_log_interval = 30
        self.last_metrics_log = 0
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info("Performance tracker initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start background monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.resource_monitor.update()
                
                # Record metrics
                self.record_metric("cpu_usage", self.resource_monitor.get_cpu_usage())
                self.record_metric("memory_usage", self.resource_monitor.get_memory_usage())
                self.record_metric("memory_usage_mb", self.resource_monitor.get_memory_usage_mb())
                
                if GPU_AVAILABLE:
                    gpu_usage = self.resource_monitor.get_gpu_usage()
                    if gpu_usage is not None:
                        self.record_metric("gpu_usage", gpu_usage)
                        
                    gpu_memory = self.resource_monitor.get_gpu_memory_usage()
                    if gpu_memory is not None:
                        self.record_metric("gpu_memory", gpu_memory)
                
                # Log metrics to database periodically
                self.log_metrics_to_database()

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval * 2)  # Wait longer on error

    def log_metrics_to_database(self):
        """Log current metrics to database if available."""
        if not self.db_manager:
            return

        current_time = time.time()
        if current_time - self.last_metrics_log < self.metrics_log_interval:
            return

        try:
            # Get current metrics
            current_metrics = self.get_current_metrics()

            # Log to database
            if current_metrics and self.db_manager.log_performance_metrics(current_metrics):
                self.last_metrics_log = current_time
                logger.debug("Performance metrics logged to database")

        except Exception as e:
            logger.error(f"Failed to log metrics to database: {e}")

    def start_timer(self, name: str):
        """Start a performance timer."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        End a performance timer and record the duration.
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds
        """
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.record_metric(f"{name}_time", duration * 1000)  # Convert to ms
            del self.timers[name]
            return duration
        return 0.0
    
    def record_metric(self, name: str, value: float):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics_history:
            self.metrics_history[name] = deque(maxlen=self.max_history)
        
        self.metrics_history[name].append({
            'timestamp': time.time(),
            'value': value
        })
    
    def increment_counter(self, name: str, amount: int = 1):
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += amount
    
    def get_metric_history(self, name: str, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get metric history.
        
        Args:
            name: Metric name
            last_n: Number of recent entries to return
            
        Returns:
            List of metric entries
        """
        if name not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[name])
        if last_n:
            return history[-last_n:]
        return history
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {}

        # Get latest values for each metric
        for name, history in self.metrics_history.items():
            if history:
                metrics[name] = history[-1]['value']

        # Add counters
        metrics.update(self.counters)

        # Add FPS metrics
        fps_metrics = self.fps_tracker.get_metrics()
        metrics.update(fps_metrics)

        return metrics

    def get_recent_values(self, metric_name: str, count: int = 10) -> List[float]:
        """Get recent values for a specific metric."""
        if metric_name not in self.metrics_history:
            return []

        history = self.metrics_history[metric_name]
        recent_entries = list(history)[-count:]
        return [entry['value'] for entry in recent_entries]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'current_metrics': self.get_current_metrics(),
            'averages': {},
            'peaks': {},
            'recommendations': []
        }
        
        # Calculate averages and peaks
        for name, history in self.metrics_history.items():
            if history:
                values = [entry['value'] for entry in history]
                summary['averages'][name] = sum(values) / len(values)
                summary['peaks'][name] = max(values)
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        current = summary['current_metrics']
        
        # CPU usage recommendations
        cpu_usage = current.get('cpu_usage', 0)
        if cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider reducing detection frequency or lowering video resolution.")
        
        # Memory usage recommendations
        memory_usage = current.get('memory_usage', 0)
        if memory_usage > 85:
            recommendations.append("High memory usage detected. Consider clearing face encoding cache or reducing max faces per frame.")
        
        # GPU recommendations
        if GPU_AVAILABLE:
            gpu_usage = current.get('gpu_usage', 0)
            if gpu_usage > 90:
                recommendations.append("High GPU usage detected. Consider enabling model quantization or reducing batch size.")
        
        # FPS recommendations
        current_fps = current.get('current_fps', 0)
        if current_fps < 15:
            recommendations.append("Low FPS detected. Consider optimizing detection settings or upgrading hardware.")
        elif current_fps > 60:
            recommendations.append("High FPS detected. System is performing well - consider enabling additional features.")
        
        return recommendations
    
    def export_metrics(self, filepath: str) -> bool:
        """
        Export metrics to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if successful
        """
        try:
            import json
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics_history': {
                    name: list(history) for name, history in self.metrics_history.items()
                },
                'counters': self.counters,
                'summary': self.get_performance_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False


class FPSTracker:
    """FPS tracking utility."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS tracker.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
    
    def update(self):
        """Update FPS calculation with new frame."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get FPS metrics."""
        return {
            'current_fps': self.get_fps(),
            'frame_count': len(self.frame_times)
        }


class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.process = psutil.Process()
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.memory_usage_mb = 0.0
        self.gpu_usage = None
        self.gpu_memory = None
    
    def update(self):
        """Update resource usage metrics."""
        try:
            # CPU usage
            self.cpu_usage = self.process.cpu_percent()
            
            # Memory usage
            memory_info = self.process.memory_info()
            self.memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # System memory percentage
            system_memory = psutil.virtual_memory()
            self.memory_usage = system_memory.percent
            
            # GPU usage (if available)
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        self.gpu_usage = gpu.load * 100
                        self.gpu_memory = gpu.memoryUtil * 100
                except Exception:
                    pass  # GPU monitoring failed
                    
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return self.cpu_usage
    
    def get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        return self.memory_usage
    
    def get_memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.memory_usage_mb
    
    def get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        return self.gpu_usage
    
    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage percentage."""
        return self.gpu_memory
