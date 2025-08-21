"""
IP Camera Manager Module

Network camera configuration and management interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any

# Import backend components
from database.database_manager import DatabaseManager
from database.models import Device
from core.camera_manager import CameraManager
from config.camera_config import CameraConfig, CameraConfigManager

logger = logging.getLogger(__name__)


class IPCameraManager:
    """IP Camera configuration and management interface."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.title = "📹 IP Camera Manager"
        self.db_manager = db_manager or DatabaseManager()
        self.camera_manager = CameraManager()
        self.config_manager = CameraConfigManager()

        # GUI components
        self.camera_tree = None
        self.form_vars = {}
        self.results_text = None

        # Edit mode tracking
        self.editing_camera_id = None  # Track which camera is being edited
        self.is_edit_mode = False      # Track if we're in edit mode

        # Main system reference for camera reloading
        self.main_system = None

        # Load existing cameras
        self._load_cameras_from_db()

    def _load_cameras_from_db(self):
        """Load camera configurations from database."""
        try:
            self.config_manager.load_from_database(self.db_manager)
            logger.info(f"Loaded {len(self.config_manager.cameras)} cameras from database")
        except Exception as e:
            logger.error(f"Failed to load cameras from database: {e}")

    def get_title(self):
        return self.title

    def set_main_system(self, main_system):
        """Set reference to main system for camera reloading."""
        self.main_system = main_system
        logger.info("Main system reference set for IP Camera Manager")

    def show_in_frame(self, parent_frame):
        """Display the IP camera manager in the given frame."""
        # Main container with notebook for tabs
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Camera List Tab
        self._create_camera_list_tab(notebook)

        # Add Camera Tab
        self._create_add_camera_tab(notebook)

        # Test Connection Tab
        self._create_test_tab(notebook)
    
    def _create_camera_list_tab(self, notebook):
        """Create the camera list tab."""
        list_frame = ttk.Frame(notebook)
        notebook.add(list_frame, text="Camera List")

        # Toolbar
        toolbar = ttk.Frame(list_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(toolbar, text="🔄 Refresh", command=self._refresh_camera_list).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="➕ Add Camera", command=lambda: notebook.select(1)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="✏️ Edit", command=self._edit_selected_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="🗑️ Delete", command=self._delete_selected_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="🔢 Fix IDs", command=self._reorganize_camera_ids).pack(side=tk.LEFT, padx=(0, 5))

        # Camera list with treeview
        columns = ("ID", "IP Address", "Port", "Protocol", "Status", "Last Seen")
        self.camera_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)

        for col in columns:
            self.camera_tree.heading(col, text=col)
            self.camera_tree.column(col, width=120)

        # Load cameras from database
        self._refresh_camera_list()

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.camera_tree.yview)
        self.camera_tree.configure(yscrollcommand=scrollbar.set)

        self.camera_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_add_camera_tab(self, notebook):
        """Create the add camera tab."""
        add_frame = ttk.Frame(notebook)
        notebook.add(add_frame, text="Add Camera")

        # Form container
        form_frame = ttk.LabelFrame(add_frame, text="Camera Configuration", padding=20)
        form_frame.pack(fill=tk.X, padx=20, pady=20)

        # Store form variables
        self.form_vars = {}

        # IP Address
        ttk.Label(form_frame, text="IP Address:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.form_vars['ip_entry'] = ttk.Entry(form_frame, width=30)
        self.form_vars['ip_entry'].grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        self.form_vars['ip_entry'].insert(0, "192.168.1.100")

        # Port
        ttk.Label(form_frame, text="Port:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.form_vars['port_entry'] = ttk.Entry(form_frame, width=30)
        self.form_vars['port_entry'].grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        self.form_vars['port_entry'].insert(0, "8080")

        # Protocol
        ttk.Label(form_frame, text="Protocol:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.form_vars['protocol_var'] = tk.StringVar(value="HTTP")
        protocol_combo = ttk.Combobox(form_frame, textvariable=self.form_vars['protocol_var'],
                                    values=["HTTP", "HTTPS"], width=27)
        protocol_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # URL suffix
        ttk.Label(form_frame, text="URL Suffix:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.form_vars['suffix_entry'] = ttk.Entry(form_frame, width=30)
        self.form_vars['suffix_entry'].grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        self.form_vars['suffix_entry'].insert(0, "/video")

        # Options
        options_frame = ttk.LabelFrame(add_frame, text="Options", padding=20)
        options_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.form_vars['auto_connect_var'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Auto-connect on startup",
                       variable=self.form_vars['auto_connect_var']).pack(anchor=tk.W)

        self.form_vars['fallback_var'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable local camera fallback",
                       variable=self.form_vars['fallback_var']).pack(anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(add_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        ttk.Button(button_frame, text="🧪 Test Connection", style="Accent.TButton",
                  command=self._test_form_connection).pack(side=tk.LEFT, padx=(0, 10))

        # Store reference to save button so we can update its text
        self.save_button = ttk.Button(button_frame, text="💾 Save Camera",
                                     command=self._save_camera)
        self.save_button.pack(side=tk.LEFT)

        ttk.Button(button_frame, text="🔄 Reset Form",
                  command=self._reset_form).pack(side=tk.RIGHT)
    
    def _create_test_tab(self, notebook):
        """Create the connection test tab."""
        test_frame = ttk.Frame(notebook)
        notebook.add(test_frame, text="Test Connection")

        # Test form
        test_form = ttk.LabelFrame(test_frame, text="Connection Test", padding=20)
        test_form.pack(fill=tk.X, padx=20, pady=20)

        ttk.Label(test_form, text="Camera URL:").pack(anchor=tk.W)
        self.form_vars['test_url_entry'] = ttk.Entry(test_form, width=50)
        self.form_vars['test_url_entry'].pack(fill=tk.X, pady=(5, 10))
        self.form_vars['test_url_entry'].insert(0, "http://192.168.1.100:8080/video")

        ttk.Button(test_form, text="🧪 Test Connection", style="Accent.TButton",
                  command=self._test_url_connection).pack()

        # Results area
        results_frame = ttk.LabelFrame(test_frame, text="Test Results", padding=20)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.results_text = tk.Text(results_frame, height=15, font=("Courier", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial message
        self._log_test_result("Ready to test camera connections...")
        self.results_text.configure(state=tk.DISABLED)

    # Backend functionality methods
    def _refresh_camera_list(self):
        """Refresh the camera list from database."""
        try:
            # Clear existing items
            if self.camera_tree:
                for item in self.camera_tree.get_children():
                    self.camera_tree.delete(item)

                # Load cameras from database
                devices = self.db_manager.get_all_devices()

                for device in devices:
                    protocol = "HTTPS" if device.use_https else "HTTP"
                    status = "Active" if device.status == "active" else "Inactive"
                    last_seen = "Now" if device.status == "active" else "Unknown"

                    self.camera_tree.insert("", tk.END, values=(
                        device.id,
                        device.ip_address,
                        device.port,
                        protocol,
                        status,
                        last_seen
                    ))

                logger.info(f"Refreshed camera list with {len(devices)} cameras")
        except Exception as e:
            logger.error(f"Failed to refresh camera list: {e}")
            messagebox.showerror("Error", f"Failed to refresh camera list: {e}")

    def _save_camera(self):
        """Save camera configuration to database."""
        try:
            # Get form values
            ip_address = self.form_vars['ip_entry'].get().strip()
            port = int(self.form_vars['port_entry'].get().strip())
            use_https = self.form_vars['protocol_var'].get() == "HTTPS"
            url_suffix = self.form_vars['suffix_entry'].get().strip()
            # Determine if URL ends with video based on suffix (for backward compatibility)
            end_with_video = url_suffix.endswith('/video') or url_suffix == '/video'

            # Validate inputs
            if not ip_address:
                messagebox.showerror("Error", "IP Address is required")
                return

            if port <= 0 or port > 65535:
                messagebox.showerror("Error", "Port must be between 1 and 65535")
                return

            if self.is_edit_mode and self.editing_camera_id:
                # Update existing camera
                device = Device(
                    id=self.editing_camera_id,
                    ip_address=ip_address,
                    port=port,
                    use_https=use_https,
                    end_with_video=end_with_video,
                    status="active"
                )

                success = self.db_manager.update_device(device)
                if success:
                    messagebox.showinfo("Success", f"Camera {self.editing_camera_id} updated successfully!")
                    logger.info(f"Updated camera {self.editing_camera_id}: {ip_address}:{port}")
                else:
                    messagebox.showerror("Error", "Failed to update camera")
                    return

            else:
                # Create new camera
                device = Device(
                    ip_address=ip_address,
                    port=port,
                    use_https=use_https,
                    end_with_video=end_with_video,
                    status="active"
                )

                device_id = self.db_manager.create_device(device)
                messagebox.showinfo("Success", f"Camera saved successfully with ID: {device_id}")
                logger.info(f"Created new camera {device_id}: {ip_address}:{port}")

            # Refresh camera list
            self._refresh_camera_list()

            # Trigger camera reload in main system if available
            self._trigger_camera_reload()

            # Reset form and exit edit mode
            self._reset_form()
            self._exit_edit_mode()

        except ValueError:
            messagebox.showerror("Error", "Port must be a valid number")
        except Exception as e:
            logger.error(f"Failed to save camera: {e}")
            messagebox.showerror("Error", f"Failed to save camera: {e}")

    def _reset_form(self):
        """Reset the camera form to default values."""
        try:
            self.form_vars['ip_entry'].delete(0, tk.END)
            self.form_vars['ip_entry'].insert(0, "192.168.1.100")

            self.form_vars['port_entry'].delete(0, tk.END)
            self.form_vars['port_entry'].insert(0, "8080")

            self.form_vars['protocol_var'].set("HTTP")

            self.form_vars['suffix_entry'].delete(0, tk.END)
            self.form_vars['suffix_entry'].insert(0, "/video")

            self.form_vars['auto_connect_var'].set(True)
            self.form_vars['fallback_var'].set(True)

            # Exit edit mode when resetting form
            self._exit_edit_mode()

        except Exception as e:
            logger.error(f"Failed to reset form: {e}")

    def _exit_edit_mode(self):
        """Exit edit mode and reset tracking variables."""
        self.is_edit_mode = False
        self.editing_camera_id = None

        # Reset button text
        if hasattr(self, 'save_button'):
            self.save_button.config(text="💾 Save Camera")

        logger.info("Exited edit mode")

    def _edit_selected_camera(self):
        """Edit the selected camera."""
        try:
            selection = self.camera_tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a camera to edit")
                return

            item = self.camera_tree.item(selection[0])
            camera_id = item['values'][0]

            # Get camera from database
            device = self.db_manager.get_device(camera_id)
            if not device:
                messagebox.showerror("Error", "Camera not found in database")
                return

            # Enter edit mode
            self.is_edit_mode = True
            self.editing_camera_id = camera_id

            # Populate form with camera data
            self.form_vars['ip_entry'].delete(0, tk.END)
            self.form_vars['ip_entry'].insert(0, device.ip_address)

            self.form_vars['port_entry'].delete(0, tk.END)
            self.form_vars['port_entry'].insert(0, str(device.port))

            self.form_vars['protocol_var'].set("HTTPS" if device.use_https else "HTTP")

            # Update URL suffix field based on device settings
            if 'suffix_entry' in self.form_vars:
                self.form_vars['suffix_entry'].delete(0, tk.END)
                self.form_vars['suffix_entry'].insert(0, "/video" if device.end_with_video else "")

            # Update button text to indicate edit mode
            if hasattr(self, 'save_button'):
                self.save_button.config(text=f"✏️ Update Camera {camera_id}")

            # Switch to Add Camera tab for editing
            messagebox.showinfo("Edit Mode",
                              f"📝 Editing Camera ID {camera_id}\n\n"
                              f"Camera data loaded in Add Camera tab.\n"
                              f"Modify the settings and click 'Update Camera' to save changes.\n\n"
                              f"The camera will be UPDATED, not created as new.")

            logger.info(f"Entered edit mode for camera {camera_id}")

        except Exception as e:
            logger.error(f"Failed to edit camera: {e}")
            messagebox.showerror("Error", f"Failed to edit camera: {e}")

    def _delete_selected_camera(self):
        """Delete the selected camera."""
        try:
            selection = self.camera_tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a camera to delete")
                return

            item = self.camera_tree.item(selection[0])
            camera_id = item['values'][0]
            camera_ip = item['values'][1]

            # Confirm deletion
            result = messagebox.askyesno("Confirm Delete",
                                       f"Are you sure you want to delete camera {camera_ip} (ID: {camera_id})?")
            if not result:
                return

            # Delete from database
            success = self.db_manager.delete_device(camera_id)
            if success:
                self._refresh_camera_list()

                # Trigger camera reload in main system if available
                self._trigger_camera_reload()

                messagebox.showinfo("Success", "Camera deleted successfully")
                logger.info(f"Deleted camera {camera_id}")
            else:
                messagebox.showerror("Error", "Failed to delete camera")

        except Exception as e:
            logger.error(f"Failed to delete camera: {e}")
            messagebox.showerror("Error", f"Failed to delete camera: {e}")

    def _reorganize_camera_ids(self):
        """Reorganize camera IDs to be sequential (1, 2, 3, ...)."""
        try:
            # Confirm with user
            result = messagebox.askyesno(
                "Reorganize Camera IDs",
                "This will reorganize camera IDs to be sequential (1, 2, 3, ...).\n\n"
                "⚠️ Warning: This will change camera IDs!\n"
                "Any external references to camera IDs will need to be updated.\n\n"
                "Do you want to continue?"
            )

            if not result:
                return

            # Perform reorganization
            success = self.db_manager.reorganize_device_ids()

            if success:
                # Refresh the camera list to show new IDs
                self._refresh_camera_list()

                # Trigger camera reload in main system if available
                self._trigger_camera_reload()

                messagebox.showinfo("Success",
                                  "✅ Camera IDs reorganized successfully!\n\n"
                                  "IDs are now sequential (1, 2, 3, ...)")
                logger.info("Camera IDs reorganized successfully")
            else:
                messagebox.showerror("Error", "Failed to reorganize camera IDs")

        except Exception as e:
            logger.error(f"Failed to reorganize camera IDs: {e}")
            messagebox.showerror("Error", f"Failed to reorganize camera IDs: {e}")

    def _trigger_camera_reload(self):
        """Trigger camera configuration reload in main system."""
        try:
            if self.main_system and hasattr(self.main_system, 'reload_camera_configurations'):
                logger.info("Triggering camera configuration reload in main system...")
                success = self.main_system.reload_camera_configurations()
                if success:
                    logger.info("Camera configurations reloaded successfully in main system")
                else:
                    logger.warning("Camera configuration reload failed in main system")
            else:
                logger.info("Main system not available for camera reload - restart application to use new cameras")
        except Exception as e:
            logger.error(f"Error triggering camera reload: {e}")

    def _test_form_connection(self):
        """Test connection using form data."""
        try:
            ip_address = self.form_vars['ip_entry'].get().strip()
            port_str = self.form_vars['port_entry'].get().strip()

            # Validate inputs
            if not ip_address:
                messagebox.showerror("Validation Error", "IP Address is required")
                return

            if not port_str:
                messagebox.showerror("Validation Error", "Port is required")
                return

            try:
                port = int(port_str)
                if port < 1 or port > 65535:
                    messagebox.showerror("Validation Error", "Port must be between 1 and 65535")
                    return
            except ValueError:
                messagebox.showerror("Validation Error", "Port must be a valid number")
                return

            use_https = self.form_vars['protocol_var'].get() == "HTTPS"
            url_suffix = self.form_vars['suffix_entry'].get().strip()

            # Construct URL
            protocol = 'https' if use_https else 'http'
            camera_url = f"{protocol}://{ip_address}:{port}{url_suffix}"

            # Update test URL field if it exists
            if 'test_url_entry' in self.form_vars:
                self.form_vars['test_url_entry'].delete(0, tk.END)
                self.form_vars['test_url_entry'].insert(0, camera_url)

            # Show immediate feedback
            if hasattr(self, 'results_text') and self.results_text:
                self.results_text.configure(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self._log_test_result(f"[{self._get_timestamp()}] Starting connection test...")
                self._log_test_result(f"[{self._get_timestamp()}] Testing URL: {camera_url}")
                self.results_text.configure(state=tk.DISABLED)

            # Run test in background thread
            threading.Thread(target=self._run_form_connection_test,
                           args=(ip_address, port, use_https, url_suffix),
                           daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to test form connection: {e}")
            messagebox.showerror("Error", f"Failed to test connection: {e}")

    def _run_form_connection_test(self, ip_address: str, port: int, use_https: bool, url_suffix: str):
        """Run connection test from form data in background thread."""
        try:
            # Construct URL
            protocol = 'https' if use_https else 'http'
            camera_url = f"{protocol}://{ip_address}:{port}{url_suffix}"

            self._log_test_result(f"[{self._get_timestamp()}] Testing connection to {camera_url}")

            # Test using camera manager
            config = {
                'ip_address': ip_address,
                'port': port,
                'use_https': use_https,
                'end_with_video': url_suffix.endswith('/video') or url_suffix == '/video'
            }

            # Test connection
            success = self.camera_manager.test_camera_connection(config)

            if success:
                self._log_test_result(f"[{self._get_timestamp()}] ✅ Connection test PASSED")
                self._log_test_result(f"[{self._get_timestamp()}] Camera is accessible and responding")

                # Show success message in main thread
                self.form_vars['ip_entry'].after(0, lambda: messagebox.showinfo(
                    "Connection Test", "✅ Connection successful! Camera is accessible."))
            else:
                self._log_test_result(f"[{self._get_timestamp()}] ❌ Connection test FAILED")
                self._log_test_result(f"[{self._get_timestamp()}] Camera is not accessible")

                # Show error message in main thread
                self.form_vars['ip_entry'].after(0, lambda: messagebox.showerror(
                    "Connection Test", "❌ Connection failed! Camera is not accessible."))

        except Exception as e:
            self._log_test_result(f"[{self._get_timestamp()}] ❌ Test error: {e}")
            logger.error(f"Connection test error: {e}")

            # Show error message in main thread
            self.form_vars['ip_entry'].after(0, lambda: messagebox.showerror(
                "Connection Test", f"❌ Test error: {e}"))

    def _test_url_connection(self):
        """Test connection to the specified URL."""
        if 'test_url_entry' not in self.form_vars:
            messagebox.showerror("Error", "Test URL entry not found")
            return

        camera_url = self.form_vars['test_url_entry'].get().strip()

        if not camera_url:
            messagebox.showerror("Error", "Camera URL is required")
            return

        # Clear previous results
        if hasattr(self, 'results_text') and self.results_text:
            self.results_text.configure(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

        # Start test in background thread
        threading.Thread(target=self._run_connection_test, args=(camera_url,), daemon=True).start()

    def _run_connection_test(self, camera_url: str):
        """Run connection test in background thread."""
        try:
            self._log_test_result(f"[{self._get_timestamp()}] Testing connection to {camera_url}")

            # Test using camera manager
            config = {
                'ip_address': camera_url.split('://')[1].split(':')[0],
                'port': int(camera_url.split(':')[-1].split('/')[0]),
                'use_https': camera_url.startswith('https'),
                'end_with_video': camera_url.endswith('/video')
            }

            # Test connection
            success = self.camera_manager.test_camera_connection(config)

            if success:
                self._log_test_result(f"[{self._get_timestamp()}] ✅ Connection test PASSED")
                self._log_test_result(f"[{self._get_timestamp()}] Camera is accessible and responding")
            else:
                self._log_test_result(f"[{self._get_timestamp()}] ❌ Connection test FAILED")
                self._log_test_result(f"[{self._get_timestamp()}] Camera is not accessible")

        except Exception as e:
            self._log_test_result(f"[{self._get_timestamp()}] ❌ Test error: {e}")
            logger.error(f"Connection test error: {e}")

    def _log_test_result(self, message: str):
        """Log a test result to the results text area."""
        try:
            # Log to console as well (remove emoji for Windows compatibility)
            clean_message = message.replace('✅', '[PASS]').replace('❌', '[FAIL]')
            logger.info(clean_message)

            # Log to GUI if available
            if hasattr(self, 'results_text') and self.results_text:
                self.results_text.configure(state=tk.NORMAL)
                self.results_text.insert(tk.END, message + "\n")
                self.results_text.see(tk.END)
                self.results_text.configure(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Failed to log test result: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")
