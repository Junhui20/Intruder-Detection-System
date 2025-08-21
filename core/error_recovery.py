#!/usr/bin/env python3
"""
Comprehensive Error Handling and Recovery System

This module provides automatic error recovery, system restart capabilities,
database corruption detection/repair, and system health monitoring.
"""

import os
import sys
import time
import sqlite3
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SystemComponent(Enum):
    """System components that can fail."""
    DATABASE = "database"
    CAMERA = "camera"
    DETECTION_ENGINE = "detection_engine"
    NOTIFICATION_SYSTEM = "notification_system"
    GUI = "gui"
    PERFORMANCE_TRACKER = "performance_tracker"

class ErrorRecoveryManager:
    """Comprehensive error recovery and system health manager."""
    
    def __init__(self, main_system=None):
        """
        Initialize error recovery manager.
        
        Args:
            main_system: Reference to main detection system
        """
        self.main_system = main_system
        self.error_history = []
        self.recovery_strategies = {}
        self.component_health = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Recovery settings
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 60  # seconds
        self.health_check_interval = 30  # seconds
        
        # Component failure thresholds
        self.failure_thresholds = {
            SystemComponent.DATABASE: 2,
            SystemComponent.CAMERA: 3,
            SystemComponent.DETECTION_ENGINE: 2,
            SystemComponent.NOTIFICATION_SYSTEM: 3,
            SystemComponent.GUI: 2,
            SystemComponent.PERFORMANCE_TRACKER: 5
        }
        
        self._setup_recovery_strategies()
        self._initialize_component_health()
        
        logger.info("Error recovery manager initialized")
    
    def _setup_recovery_strategies(self):
        """Set up recovery strategies for different components."""
        self.recovery_strategies = {
            SystemComponent.DATABASE: self._recover_database,
            SystemComponent.CAMERA: self._recover_camera,
            SystemComponent.DETECTION_ENGINE: self._recover_detection_engine,
            SystemComponent.NOTIFICATION_SYSTEM: self._recover_notification_system,
            SystemComponent.GUI: self._recover_gui,
            SystemComponent.PERFORMANCE_TRACKER: self._recover_performance_tracker
        }
    
    def _initialize_component_health(self):
        """Initialize component health tracking."""
        for component in SystemComponent:
            self.component_health[component] = {
                'status': 'healthy',
                'last_check': time.time(),
                'failure_count': 0,
                'last_failure': None,
                'recovery_attempts': 0,
                'last_recovery': None
            }
    
    def start_monitoring(self):
        """Start system health monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_system_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(30)
    
    def _check_system_health(self):
        """Check health of all system components."""
        if not self.main_system:
            return
        
        # Check database health
        self._check_database_health()
        
        # Check camera health
        self._check_camera_health()
        
        # Check detection engine health
        self._check_detection_engine_health()
        
        # Check notification system health
        self._check_notification_system_health()
        
        # Check GUI health
        self._check_gui_health()
        
        # Check performance tracker health
        self._check_performance_tracker_health()
    
    def _check_database_health(self):
        """Check database health."""
        try:
            if self.main_system.db_manager:
                # Try a simple query
                stats = self.main_system.db_manager.get_database_stats()
                if stats:
                    self._mark_component_healthy(SystemComponent.DATABASE)
                else:
                    self._mark_component_unhealthy(SystemComponent.DATABASE, "Database query failed")
            else:
                self._mark_component_unhealthy(SystemComponent.DATABASE, "Database manager not initialized")
        except Exception as e:
            self._mark_component_unhealthy(SystemComponent.DATABASE, f"Database error: {e}")
    
    def _check_camera_health(self):
        """Check camera health."""
        try:
            if self.main_system.camera_manager:
                if self.main_system.camera_manager.is_connected():
                    self._mark_component_healthy(SystemComponent.CAMERA)
                else:
                    self._mark_component_unhealthy(SystemComponent.CAMERA, "Camera not connected")
            else:
                self._mark_component_unhealthy(SystemComponent.CAMERA, "Camera manager not initialized")
        except Exception as e:
            self._mark_component_unhealthy(SystemComponent.CAMERA, f"Camera error: {e}")
    
    def _check_detection_engine_health(self):
        """Check detection engine health."""
        try:
            if self.main_system.detection_engine:
                # Detection engine is considered healthy if it exists and was initialized
                self._mark_component_healthy(SystemComponent.DETECTION_ENGINE)
            else:
                self._mark_component_unhealthy(SystemComponent.DETECTION_ENGINE, "Detection engine not initialized")
        except Exception as e:
            self._mark_component_unhealthy(SystemComponent.DETECTION_ENGINE, f"Detection engine error: {e}")
    
    def _check_notification_system_health(self):
        """Check notification system health."""
        try:
            if self.main_system.notification_system:
                # Check if notification system is responsive
                self._mark_component_healthy(SystemComponent.NOTIFICATION_SYSTEM)
            else:
                self._mark_component_unhealthy(SystemComponent.NOTIFICATION_SYSTEM, "Notification system not initialized")
        except Exception as e:
            self._mark_component_unhealthy(SystemComponent.NOTIFICATION_SYSTEM, f"Notification system error: {e}")
    
    def _check_gui_health(self):
        """Check GUI health."""
        try:
            if self.main_system.gui:
                # GUI is considered healthy if it exists
                self._mark_component_healthy(SystemComponent.GUI)
            else:
                # GUI might not be initialized in headless mode
                self._mark_component_healthy(SystemComponent.GUI)
        except Exception as e:
            self._mark_component_unhealthy(SystemComponent.GUI, f"GUI error: {e}")
    
    def _check_performance_tracker_health(self):
        """Check performance tracker health."""
        try:
            if self.main_system.performance_tracker:
                self._mark_component_healthy(SystemComponent.PERFORMANCE_TRACKER)
            else:
                self._mark_component_unhealthy(SystemComponent.PERFORMANCE_TRACKER, "Performance tracker not initialized")
        except Exception as e:
            self._mark_component_unhealthy(SystemComponent.PERFORMANCE_TRACKER, f"Performance tracker error: {e}")
    
    def _mark_component_healthy(self, component: SystemComponent):
        """Mark a component as healthy."""
        health = self.component_health[component]
        if health['status'] != 'healthy':
            logger.info(f"Component {component.value} recovered")
            health['status'] = 'healthy'
            health['failure_count'] = 0
        health['last_check'] = time.time()
    
    def _mark_component_unhealthy(self, component: SystemComponent, error_message: str):
        """Mark a component as unhealthy."""
        health = self.component_health[component]
        health['status'] = 'unhealthy'
        health['failure_count'] += 1
        health['last_failure'] = time.time()
        health['last_check'] = time.time()
        
        logger.warning(f"Component {component.value} unhealthy: {error_message}")
        
        # Attempt recovery if threshold reached
        if health['failure_count'] >= self.failure_thresholds[component]:
            self._attempt_component_recovery(component, error_message)
    
    def _attempt_component_recovery(self, component: SystemComponent, error_message: str):
        """Attempt to recover a failed component."""
        health = self.component_health[component]
        
        # Check recovery cooldown
        if (health['last_recovery'] and 
            time.time() - health['last_recovery'] < self.recovery_cooldown):
            return
        
        # Check max recovery attempts
        if health['recovery_attempts'] >= self.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for {component.value}")
            return
        
        logger.info(f"Attempting recovery for {component.value}")
        
        try:
            recovery_func = self.recovery_strategies.get(component)
            if recovery_func:
                success = recovery_func(error_message)
                
                health['recovery_attempts'] += 1
                health['last_recovery'] = time.time()
                
                if success:
                    logger.info(f"Recovery successful for {component.value}")
                    health['failure_count'] = 0
                    health['status'] = 'recovering'
                else:
                    logger.error(f"Recovery failed for {component.value}")
            else:
                logger.error(f"No recovery strategy for {component.value}")
                
        except Exception as e:
            logger.error(f"Error during recovery of {component.value}: {e}")
    
    def _recover_database(self, error_message: str) -> bool:
        """Recover database component."""
        try:
            logger.info("Attempting database recovery...")
            
            # Check for database corruption
            if self._check_database_corruption():
                logger.warning("Database corruption detected, attempting repair...")
                if self._repair_database():
                    logger.info("Database repair successful")
                else:
                    logger.error("Database repair failed")
                    return False
            
            # Reinitialize database manager
            if self.main_system:
                from database.database_manager import DatabaseManager
                self.main_system.db_manager = DatabaseManager()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False
    
    def _recover_camera(self, error_message: str) -> bool:
        """Recover camera component."""
        try:
            logger.info("Attempting camera recovery...")
            
            if self.main_system and self.main_system.camera_manager:
                # Try to reconnect camera
                self.main_system.camera_manager.disconnect()
                time.sleep(2)
                return self.main_system.camera_manager.connect()
            
            return False
            
        except Exception as e:
            logger.error(f"Camera recovery failed: {e}")
            return False
    
    def _recover_detection_engine(self, error_message: str) -> bool:
        """Recover detection engine component."""
        try:
            logger.info("Attempting detection engine recovery...")
            
            if self.main_system:
                # Reinitialize detection engine
                from core.detection_engine import DetectionEngine
                from config.detection_config import DetectionConfig
                
                config = DetectionConfig()
                self.main_system.detection_engine = DetectionEngine(config)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Detection engine recovery failed: {e}")
            return False
    
    def _recover_notification_system(self, error_message: str) -> bool:
        """Recover notification system component."""
        try:
            logger.info("Attempting notification system recovery...")
            
            if self.main_system and hasattr(self.main_system, 'settings'):
                # Reinitialize notification system
                from core.notification_system import NotificationSystem
                
                if self.main_system.settings.bot_token:
                    self.main_system.notification_system = NotificationSystem(
                        self.main_system.settings.bot_token
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Notification system recovery failed: {e}")
            return False
    
    def _recover_gui(self, error_message: str) -> bool:
        """Recover GUI component."""
        try:
            logger.info("Attempting GUI recovery...")
            # GUI recovery is complex and might require restart
            return True  # Placeholder
            
        except Exception as e:
            logger.error(f"GUI recovery failed: {e}")
            return False
    
    def _recover_performance_tracker(self, error_message: str) -> bool:
        """Recover performance tracker component."""
        try:
            logger.info("Attempting performance tracker recovery...")
            
            if self.main_system:
                # Reinitialize performance tracker
                from utils.performance_tracker import PerformanceTracker
                self.main_system.performance_tracker = PerformanceTracker()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Performance tracker recovery failed: {e}")
            return False
    
    def _check_database_corruption(self) -> bool:
        """Check if database is corrupted."""
        try:
            if self.main_system and self.main_system.db_manager:
                db_path = self.main_system.db_manager.db_path
                
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute("PRAGMA integrity_check")
                    result = cursor.fetchone()
                    
                    return result[0] != "ok"
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking database corruption: {e}")
            return True  # Assume corrupted if we can't check
    
    def _repair_database(self) -> bool:
        """Attempt to repair corrupted database."""
        try:
            if self.main_system and self.main_system.db_manager:
                db_path = self.main_system.db_manager.db_path
                backup_path = f"{db_path}.backup_{int(time.time())}"
                
                # Create backup
                import shutil
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backup created: {backup_path}")
                
                # Try to repair using SQLite dump/restore
                dump_path = f"{db_path}.dump"
                
                # Dump database
                with open(dump_path, 'w') as f:
                    subprocess.run(['sqlite3', db_path, '.dump'], stdout=f, check=True)
                
                # Remove corrupted database
                os.remove(db_path)
                
                # Restore from dump
                with open(dump_path, 'r') as f:
                    subprocess.run(['sqlite3', db_path], stdin=f, check=True)
                
                # Clean up dump file
                os.remove(dump_path)
                
                logger.info("Database repair completed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Database repair failed: {e}")
            return False
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        report = {
            'overall_status': 'healthy',
            'components': {},
            'error_count': len(self.error_history),
            'monitoring_active': self.monitoring_active,
            'last_check': datetime.now().isoformat()
        }
        
        unhealthy_count = 0
        
        for component, health in self.component_health.items():
            component_report = {
                'status': health['status'],
                'failure_count': health['failure_count'],
                'recovery_attempts': health['recovery_attempts'],
                'last_check': datetime.fromtimestamp(health['last_check']).isoformat() if health['last_check'] else None,
                'last_failure': datetime.fromtimestamp(health['last_failure']).isoformat() if health['last_failure'] else None
            }
            
            report['components'][component.value] = component_report
            
            if health['status'] != 'healthy':
                unhealthy_count += 1
        
        # Determine overall status
        if unhealthy_count == 0:
            report['overall_status'] = 'healthy'
        elif unhealthy_count <= 2:
            report['overall_status'] = 'degraded'
        else:
            report['overall_status'] = 'critical'
        
        return report
    
    def force_system_restart(self):
        """Force system restart as last resort."""
        logger.critical("Forcing system restart...")
        
        try:
            if self.main_system:
                self.main_system.shutdown()
            
            # Restart the application
            python = sys.executable
            os.execl(python, python, *sys.argv)
            
        except Exception as e:
            logger.error(f"Error during forced restart: {e}")
            sys.exit(1)
