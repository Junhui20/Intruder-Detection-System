"""
Detection View Module

Real-time detection interface with live video feed and controls.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import logging
import numpy as np
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class DetectionView:
    """Real-time detection view with video feed and controls."""

    def __init__(self):
        self.title = "🔍 Detection View"
        self.video_label = None
        self.detection_active = False
        self.current_frame = None
        self.video_width = 640
        self.video_height = 480
        self.log_text = None  # Reference to the detection log text widget
        self.main_system = None  # Reference to main system for camera access

        # Confidence threshold controls
        self.human_conf_var = tk.DoubleVar(value=0.6)
        self.animal_conf_var = tk.DoubleVar(value=0.6)
        self.human_conf_label = None
        self.animal_conf_label = None

        # Detection toggle controls
        self.human_detect_var = tk.BooleanVar(value=True)
        self.animal_detect_var = tk.BooleanVar(value=True)
        self.face_recog_var = tk.BooleanVar(value=True)
        self.pet_id_var = tk.BooleanVar(value=True)

        # Callbacks
        self.detection_callback = None
        
    def get_title(self):
        return self.title
    
    def show_in_frame(self, parent_frame):
        """Display the detection view in the given frame."""
        # Main container
        main_frame = ttk.Frame(parent_frame)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display area
        video_frame = ttk.LabelFrame(main_frame, text="Live Video Feed", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Video feed will appear here", anchor=tk.CENTER)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Controls panel
        controls_frame = ttk.LabelFrame(main_frame, text="Detection Controls", padding=10)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Detection settings
        ttk.Label(controls_frame, text="Detection Settings", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # Confidence thresholds
        self.human_conf_label = ttk.Label(controls_frame, text=f"Human Confidence: {self.human_conf_var.get():.0%}")
        self.human_conf_label.pack(anchor=tk.W)

        human_conf = ttk.Scale(
            controls_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.human_conf_var,
            command=self._update_human_conf_label
        )
        human_conf.pack(fill=tk.X, pady=(0, 10))

        self.animal_conf_label = ttk.Label(controls_frame, text=f"Animal Confidence: {self.animal_conf_var.get():.0%}")
        self.animal_conf_label.pack(anchor=tk.W)

        animal_conf = ttk.Scale(
            controls_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.animal_conf_var,
            command=self._update_animal_conf_label
        )
        animal_conf.pack(fill=tk.X, pady=(0, 10))
        
        # Detection toggles
        ttk.Separator(controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Checkbutton(
            controls_frame,
            text="Human Detection",
            variable=self.human_detect_var,
            command=self._update_detection_settings
        ).pack(anchor=tk.W)

        ttk.Checkbutton(
            controls_frame,
            text="Animal Detection",
            variable=self.animal_detect_var,
            command=self._update_detection_settings
        ).pack(anchor=tk.W)

        ttk.Checkbutton(
            controls_frame,
            text="Face Recognition",
            variable=self.face_recog_var,
            command=self._update_detection_settings
        ).pack(anchor=tk.W)

        ttk.Checkbutton(
            controls_frame,
            text="Pet Identification",
            variable=self.pet_id_var,
            command=self._update_detection_settings
        ).pack(anchor=tk.W)
        
        # Manual actions
        ttk.Separator(controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(controls_frame, text="Manual Actions", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        ttk.Button(controls_frame, text="📸 Capture Photo", width=20, command=self._capture_photo).pack(pady=2)
        ttk.Button(controls_frame, text="🔄 Reset Tracking", width=20, command=self._reset_tracking).pack(pady=2)
        ttk.Button(controls_frame, text="📊 View Statistics", width=20, command=self._view_statistics).pack(pady=2)
        
        # Detection log
        log_frame = ttk.LabelFrame(controls_frame, text="Recent Detections", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Scrollable text widget for detection log
        self.log_text = tk.Text(log_frame, height=8, width=25, font=("Courier", 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial message
        initial_message = "[System] Detection log ready...\n"
        self.log_text.insert(tk.END, initial_message)
        self.log_text.configure(state=tk.DISABLED)

        # Add database log viewer button
        db_log_frame = ttk.Frame(controls_frame)
        db_log_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(db_log_frame, text="📊 View Database Log",
                  command=self.show_database_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(db_log_frame, text="📈 Detection Stats",
                  command=self.show_detection_stats).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(db_log_frame, text="🔄 Refresh",
                  command=self.refresh_detection_log).pack(side=tk.LEFT)

    def update_video_feed(self, frame):
        """Update the video feed with a new frame."""
        if self.video_label is None or frame is None:
            return

        try:
            # Check if widget still exists and is valid
            if not self.video_label.winfo_exists():
                return

            # Resize frame to fit display
            height, width = frame.shape[:2]
            aspect_ratio = width / height

            if aspect_ratio > self.video_width / self.video_height:
                # Width is the limiting factor
                new_width = self.video_width
                new_height = int(self.video_width / aspect_ratio)
            else:
                # Height is the limiting factor
                new_height = self.video_height
                new_width = int(self.video_height * aspect_ratio)

            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # Update label - check again before updating
            if self.video_label.winfo_exists():
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo  # Keep a reference

        except tk.TclError as e:
            # Widget has been destroyed, ignore silently
            logger.debug(f"Widget destroyed during video update: {e}")
        except Exception as e:
            logger.error(f"Error updating video feed: {e}")

    def clear_video_feed(self):
        """Clear the video feed display."""
        if self.video_label:
            self.video_label.configure(image="", text="Video feed will appear here")

    def add_detection_log(self, detection_type, name, confidence=None, additional_info=None):
        """Add a detection entry to the log."""
        if not self.log_text:
            return

        try:
            # Check if widget still exists and is valid
            if not self.log_text.winfo_exists():
                return

            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Format the log entry based on detection type
            if detection_type == "human":
                if confidence is not None and confidence > 0:
                    log_entry = f"[{timestamp}] Human detected: {name} ({confidence:.1f}%)\n"
                else:
                    log_entry = f"[{timestamp}] Unknown person detected\n"
            elif detection_type == "animal":
                if name != "Unknown":
                    log_entry = f"[{timestamp}] Pet identified: {name}"
                    if confidence is not None:
                        log_entry += f" ({confidence:.1f}%)"
                    if additional_info:
                        log_entry += f" ({additional_info})"
                    log_entry += "\n"
                else:
                    color_info = f" ({additional_info})" if additional_info else ""
                    log_entry = f"[{timestamp}] Animal detected: Unknown{color_info}\n"
            elif detection_type == "notification":
                log_entry = f"[{timestamp}] Notification sent to {name} users\n"
            else:
                log_entry = f"[{timestamp}] {detection_type}: {name}\n"

            # Update the text widget with additional safety checks
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, log_entry)

            # Keep only the last 50 lines to prevent memory issues
            lines = self.log_text.get("1.0", tk.END).split('\n')
            if len(lines) > 50:
                self.log_text.delete("1.0", f"{len(lines)-50}.0")

            # Auto-scroll to bottom
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        except tk.TclError as e:
            # Widget has been destroyed or is invalid
            logger.debug(f"Widget destroyed during log update: {e}")
        except Exception as e:
            logger.error(f"Error adding detection log: {e}")

    def clear_detection_log(self):
        """Clear the detection log."""
        if self.log_text:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert(tk.END, "[System] Detection log cleared...\n")
            self.log_text.configure(state=tk.DISABLED)

    def _update_human_conf_label(self, value):
        """Update the human confidence label with current value."""
        if self.human_conf_label:
            percentage = float(value) * 100
            self.human_conf_label.configure(text=f"Human Confidence: {percentage:.0f}%")
        self._update_detection_settings()

    def _update_animal_conf_label(self, value):
        """Update the animal confidence label with current value."""
        if self.animal_conf_label:
            percentage = float(value) * 100
            self.animal_conf_label.configure(text=f"Animal Confidence: {percentage:.0f}%")
        self._update_detection_settings()

    def _update_detection_settings(self):
        """Update detection settings based on GUI controls"""
        if hasattr(self, 'detection_callback') and self.detection_callback:
            settings = {
                'human_detection': self.human_detect_var.get(),
                'animal_detection': self.animal_detect_var.get(),
                'face_recognition': self.face_recog_var.get(),
                'pet_identification': self.pet_id_var.get(),
                'human_confidence': self.human_conf_var.get(),
                'animal_confidence': self.animal_conf_var.get()
            }
            print(f"[DEBUG] Detection settings updated: {settings}")  # Debug log
            self.detection_callback(settings)
        else:
            print(f"[DEBUG] No detection callback available: hasattr={hasattr(self, 'detection_callback')}, callback={getattr(self, 'detection_callback', None)}")  # Debug log

    def set_detection_callback(self, callback):
        """Set the callback for detection settings updates"""
        self.detection_callback = callback

    def set_main_system(self, main_system):
        """Set reference to main system for database access."""
        self.main_system = main_system

    def _capture_photo(self):
        """Capture a photo from the camera and save it."""
        try:
            if not self.main_system or not self.main_system.camera_manager:
                messagebox.showerror("Error", "Camera system not available")
                return

            # Capture frame from camera
            frame = self.main_system.camera_manager.capture_frame()
            if frame is None:
                messagebox.showerror("Capture Error", "Failed to capture frame from camera")
                return

            # Create photos directory if it doesn't exist
            photos_dir = "data/photos"
            os.makedirs(photos_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_capture_{timestamp}.jpg"
            filepath = os.path.join(photos_dir, filename)

            # Save image
            success = cv2.imwrite(filepath, frame)
            if success:
                messagebox.showinfo("Success", f"Photo captured and saved to:\n{filepath}")

                # Send notification if notification system is available
                if self.main_system.notification_system:
                    try:
                        self.main_system.notification_system.send_notification(
                            'manual_capture',
                            f"📸 Manual photo captured at {datetime.now().strftime('%H:%M:%S')}",
                            photo_path=filepath
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send notification: {e}")

                logger.info(f"Manual photo captured: {filepath}")
            else:
                messagebox.showerror("Save Error", "Failed to save captured photo")

        except Exception as e:
            logger.error(f"Photo capture error: {e}")
            messagebox.showerror("Capture Error", f"Failed to capture photo: {e}")

    def _reset_tracking(self):
        """Reset person tracking."""
        try:
            if not self.main_system or not self.main_system.detection_engine:
                messagebox.showwarning("Warning", "Detection system not available")
                return

            # Reset person tracker if available
            if hasattr(self.main_system.detection_engine, 'person_tracker'):
                self.main_system.detection_engine.person_tracker.tracks.clear()
                self.main_system.detection_engine.person_tracker.disappeared.clear()
                self.main_system.detection_engine.person_tracker.next_id = 0

            # Reset face recognition tracking if available
            if self.main_system.face_recognition and hasattr(self.main_system.face_recognition, 'track_identities'):
                self.main_system.face_recognition.track_identities.clear()

            messagebox.showinfo("Success", "Person tracking has been reset")
            logger.info("Person tracking reset manually")

        except Exception as e:
            logger.error(f"Reset tracking error: {e}")
            messagebox.showerror("Reset Error", f"Failed to reset tracking: {e}")

    def _view_statistics(self):
        """Show detection statistics."""
        try:
            if not self.main_system:
                messagebox.showwarning("Warning", "System not available")
                return

            # Collect statistics from various components
            stats_text = "=== Detection System Statistics ===\n\n"

            # Detection engine stats
            if self.main_system.detection_engine:
                detection_stats = self.main_system.detection_engine.get_performance_stats()
                stats_text += f"Detection Engine:\n"
                stats_text += f"  Total detections: {detection_stats.get('total_detections', 0)}\n"
                stats_text += f"  Average processing time: {detection_stats.get('avg_processing_time', 0):.3f}s\n\n"

            # Face recognition stats
            if self.main_system.face_recognition:
                face_stats = self.main_system.face_recognition.get_performance_stats()
                stats_text += f"Face Recognition:\n"
                stats_text += f"  Total faces processed: {face_stats.get('total_faces_processed', 0)}\n"
                stats_text += f"  Successful recognitions: {face_stats.get('successful_recognitions', 0)}\n"
                stats_text += f"  Recognition rate: {face_stats.get('recognition_accuracy', 0):.1f}%\n"
                stats_text += f"  Active tracks: {face_stats.get('active_tracks', 0)}\n"
                stats_text += f"  Identity changes: {face_stats.get('identity_changes', 0)}\n"
                stats_text += f"  Temporal smoothing applied: {face_stats.get('temporal_smoothing_applied', 0)}\n\n"

            # Camera stats
            if self.main_system.camera_manager:
                camera_stats = self.main_system.camera_manager.get_performance_stats()
                stats_text += f"Camera System:\n"
                stats_text += f"  Active camera: {self.main_system.camera_manager.active_camera}\n"
                stats_text += f"  Total cameras: {len(self.main_system.camera_manager.cameras)}\n"
                stats_text += f"  Average capture time: {camera_stats.get('avg_capture_time', 0):.3f}s\n"

            # Show statistics in a new window
            stats_window = tk.Toplevel()
            stats_window.title("Detection Statistics")
            stats_window.geometry("500x400")

            text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert(tk.END, stats_text)
            text_widget.config(state=tk.DISABLED)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(stats_window, orient=tk.VERTICAL, command=text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)

            # Close button
            ttk.Button(stats_window, text="Close", command=stats_window.destroy).pack(pady=10)

        except Exception as e:
            logger.error(f"View statistics error: {e}")
            messagebox.showerror("Statistics Error", f"Failed to show statistics: {e}")

    def show_database_log(self):
        """Show recent detections from database in a new window."""
        try:
            # Create new window
            log_window = tk.Toplevel()
            log_window.title("📊 Database Detection Log")
            log_window.geometry("800x600")

            # Create text widget with scrollbar
            text_frame = ttk.Frame(log_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            text_widget = tk.Text(text_frame, font=("Courier", 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Get recent detections from main system
            if hasattr(self, 'main_system') and self.main_system:
                recent_detections = self.main_system.get_recent_detections(50)

                if recent_detections:
                    text_widget.insert(tk.END, "📊 RECENT DETECTIONS FROM DATABASE\n")
                    text_widget.insert(tk.END, "=" * 60 + "\n\n")

                    for detection in recent_detections:
                        timestamp = detection.detected_at.strftime("%Y-%m-%d %H:%M:%S") if detection.detected_at else "Unknown"
                        confidence_str = f"{detection.confidence:.2%}" if detection.confidence else "N/A"
                        notification_str = "✅" if detection.notification_sent else "❌"

                        text_widget.insert(tk.END, f"🕒 {timestamp}\n")
                        text_widget.insert(tk.END, f"🔍 Type: {detection.detection_type.title()}\n")
                        text_widget.insert(tk.END, f"👤 Entity: {detection.entity_name or 'Unknown'}\n")
                        text_widget.insert(tk.END, f"📊 Confidence: {confidence_str}\n")
                        text_widget.insert(tk.END, f"📱 Notification: {notification_str}\n")
                        text_widget.insert(tk.END, "-" * 40 + "\n\n")
                else:
                    text_widget.insert(tk.END, "No detections found in database.\n")
            else:
                text_widget.insert(tk.END, "Database connection not available.\n")

            text_widget.configure(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Error showing database log: {e}")

    def show_detection_stats(self):
        """Show detection statistics in a new window."""
        try:
            # Create new window
            stats_window = tk.Toplevel()
            stats_window.title("📈 Detection Statistics")
            stats_window.geometry("600x400")

            # Create text widget with scrollbar
            text_frame = ttk.Frame(stats_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            text_widget = tk.Text(text_frame, font=("Courier", 11))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Get detection statistics
            if hasattr(self, 'main_system') and self.main_system:
                stats_7d = self.main_system.get_detection_stats(7)
                stats_24h = self.main_system.get_detection_stats(1)

                text_widget.insert(tk.END, "📈 DETECTION STATISTICS\n")
                text_widget.insert(tk.END, "=" * 40 + "\n\n")

                text_widget.insert(tk.END, "📅 Last 24 Hours:\n")
                if stats_24h:
                    for detection_type, count in stats_24h.items():
                        text_widget.insert(tk.END, f"  • {detection_type.title()}: {count}\n")
                else:
                    text_widget.insert(tk.END, "  No detections\n")

                text_widget.insert(tk.END, "\n📅 Last 7 Days:\n")
                if stats_7d:
                    for detection_type, count in stats_7d.items():
                        text_widget.insert(tk.END, f"  • {detection_type.title()}: {count}\n")
                else:
                    text_widget.insert(tk.END, "  No detections\n")

                # Get metric averages if available
                metric_averages = self.main_system.get_metric_averages(24)
                if metric_averages:
                    text_widget.insert(tk.END, "\n📊 Performance (24h avg):\n")
                    for metric, value in metric_averages.items():
                        unit = "fps" if metric == "fps" else "%" if "usage" in metric else "ms" if "time" in metric else ""
                        text_widget.insert(tk.END, f"  • {metric.replace('_', ' ').title()}: {value}{unit}\n")
            else:
                text_widget.insert(tk.END, "Database connection not available.\n")

            text_widget.configure(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Error showing detection stats: {e}")

    def refresh_detection_log(self):
        """Refresh the current detection log display."""
        try:
            # Clear current log
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)

            # Add header
            self.log_text.insert(tk.END, "[System] Detection log refreshed...\n")

            # Get recent detections and add to log
            if hasattr(self, 'main_system') and self.main_system:
                recent_detections = self.main_system.get_recent_detections(10)

                for detection in recent_detections:
                    timestamp = detection.detected_at.strftime("%H:%M:%S") if detection.detected_at else "Unknown"
                    confidence_str = f"{detection.confidence:.1%}" if detection.confidence else "N/A"

                    log_entry = f"[{timestamp}] {detection.detection_type.title()}: {detection.entity_name or 'Unknown'} ({confidence_str})\n"
                    self.log_text.insert(tk.END, log_entry)

            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)

        except Exception as e:
            logger.error(f"Error refreshing detection log: {e}")
