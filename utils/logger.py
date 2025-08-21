"""
Comprehensive Logging System

This module provides centralized logging configuration for the entire system.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file_path: str = "logs/detection_system.log",
    max_file_size: int = 10,  # MB
    max_files: int = 5,
    console_output: bool = True
) -> None:
    """
    Set up comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file_path: Path to log file
        max_file_size: Maximum log file size in MB
        max_files: Maximum number of log files to keep
        console_output: Whether to output to console
    """
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation and UTF-8 encoding
    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size * 1024 * 1024,  # Convert MB to bytes
            backupCount=max_files,
            encoding='utf-8'  # Use UTF-8 encoding for file output
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Console output: {console_output}")
    logger.info(f"File logging: {log_to_file}")
    if log_to_file:
        logger.info(f"Log file: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """
    Specialized logger for performance metrics.
    """
    
    def __init__(self, name: str = "performance"):
        """Initialize performance logger."""
        self.logger = logging.getLogger(name)
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and log the duration.
        
        Args:
            operation: Operation name
            
        Returns:
            Duration in seconds
        """
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.debug(f"{operation} completed in {duration:.3f}s")
            del self.start_times[operation]
            return duration
        else:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
    
    def log_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log a performance metric."""
        self.logger.info(f"METRIC: {metric_name} = {value} {unit}")
    
    def log_fps(self, fps: float) -> None:
        """Log FPS metric."""
        self.log_metric("FPS", fps, "fps")
    
    def log_processing_time(self, operation: str, time_ms: float) -> None:
        """Log processing time metric."""
        self.log_metric(f"{operation}_time", time_ms, "ms")


class DetectionLogger:
    """
    Specialized logger for detection events.
    """
    
    def __init__(self, name: str = "detection"):
        """Initialize detection logger."""
        self.logger = logging.getLogger(name)
    
    def log_human_detection(self, name: str, confidence: float, bbox: tuple) -> None:
        """Log human detection event."""
        self.logger.info(f"HUMAN_DETECTED: {name} (confidence: {confidence:.2f}, bbox: {bbox})")
    
    def log_animal_detection(self, animal_type: str, confidence: float, bbox: tuple) -> None:
        """Log animal detection event."""
        self.logger.info(f"ANIMAL_DETECTED: {animal_type} (confidence: {confidence:.2f}, bbox: {bbox})")
    
    def log_pet_identification(self, pet_name: str, confidence: float, method: str) -> None:
        """Log pet identification event."""
        self.logger.info(f"PET_IDENTIFIED: {pet_name} (confidence: {confidence:.2f}, method: {method})")
    
    def log_unknown_detection(self, detection_type: str, confidence: float) -> None:
        """Log unknown detection event."""
        self.logger.warning(f"UNKNOWN_{detection_type.upper()}: confidence: {confidence:.2f}")
    
    def log_notification_sent(self, recipient_count: int, message_type: str) -> None:
        """Log notification event."""
        self.logger.info(f"NOTIFICATION_SENT: {message_type} to {recipient_count} recipients")


class SystemLogger:
    """
    Specialized logger for system events.
    """
    
    def __init__(self, name: str = "system"):
        """Initialize system logger."""
        self.logger = logging.getLogger(name)
    
    def log_startup(self, component: str) -> None:
        """Log component startup."""
        self.logger.info(f"STARTUP: {component} initialized")
    
    def log_shutdown(self, component: str) -> None:
        """Log component shutdown."""
        self.logger.info(f"SHUTDOWN: {component} stopped")
    
    def log_error(self, component: str, error: str) -> None:
        """Log system error."""
        self.logger.error(f"ERROR: {component} - {error}")
    
    def log_camera_connection(self, camera_id: str, status: str) -> None:
        """Log camera connection event."""
        self.logger.info(f"CAMERA: {camera_id} - {status}")
    
    def log_database_operation(self, operation: str, table: str, success: bool) -> None:
        """Log database operation."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"DATABASE: {operation} on {table} - {status}")
    
    def log_telegram_event(self, event: str, details: str = "") -> None:
        """Log Telegram bot event."""
        self.logger.info(f"TELEGRAM: {event} - {details}")


# Global logger instances for easy access
performance_logger = PerformanceLogger()
detection_logger = DetectionLogger()
system_logger = SystemLogger()


def log_exception(logger: logging.Logger, exception: Exception, context: str = "") -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context information
    """
    import traceback
    
    context_str = f" in {context}" if context else ""
    logger.error(f"Exception{context_str}: {str(exception)}")
    logger.error(f"Traceback: {traceback.format_exc()}")


def configure_module_loggers() -> None:
    """Configure loggers for specific modules with appropriate levels."""
    
    # Set specific log levels for noisy modules
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Keep our modules at INFO level
    logging.getLogger("core").setLevel(logging.INFO)
    logging.getLogger("gui").setLevel(logging.INFO)
    logging.getLogger("database").setLevel(logging.INFO)
    logging.getLogger("config").setLevel(logging.INFO)
    logging.getLogger("utils").setLevel(logging.INFO)


# Initialize module loggers when imported
configure_module_loggers()
