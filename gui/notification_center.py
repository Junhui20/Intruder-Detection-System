"""
Notification Center Module

Telegram user management and bot configuration interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
import requests
import threading

logger = logging.getLogger(__name__)


class NotificationCenter:
    """Telegram notification management interface."""

    def __init__(self, database_manager=None, notification_system=None, config_manager=None):
        self.title = "📱 Notification Center"
        self.database_manager = database_manager
        self.notification_system = notification_system
        self.config_manager = config_manager

        # UI components
        self.users_tree = None
        self.user_details_vars = {}
        self.bot_config_vars = {}
        self.test_vars = {}
        self.test_results_text = None

        # Current selection
        self.selected_user = None

    def get_title(self):
        return self.title
    
    def show_in_frame(self, parent_frame):
        """Display the notification center interface."""
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Users tab
        self._create_users_tab(notebook)

        # Bot configuration tab
        self._create_bot_config_tab(notebook)

        # Test tab
        self._create_test_tab(notebook)

        # Load initial data
        self._load_users()
        self._load_bot_config()
    
    def _create_users_tab(self, notebook):
        """Create users management tab."""
        users_frame = ttk.Frame(notebook)
        notebook.add(users_frame, text="👥 Users")
        
        # Toolbar
        toolbar = ttk.Frame(users_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="➕ Add User", command=self._add_user).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="✏️ Edit", command=self._edit_user).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="🗑️ Remove", command=self._remove_user).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="🔄 Refresh", command=self._load_users).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="📤 Test Message", command=self._test_user_message).pack(side=tk.RIGHT)

        # Create tree frame first
        tree_frame = ttk.Frame(users_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Users list - create tree in the tree_frame
        columns = ("Username", "Chat ID", "Human Alerts", "Animal Alerts", "Status", "Last Seen")
        self.users_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)

        for col in columns:
            self.users_tree.heading(col, text=col)
            if col == "Chat ID":
                self.users_tree.column(col, width=100)
            elif col in ["Human Alerts", "Animal Alerts"]:
                self.users_tree.column(col, width=80)
            else:
                self.users_tree.column(col, width=120)

        # Bind selection event
        self.users_tree.bind('<<TreeviewSelect>>', self._on_user_select)
        self.users_tree.bind('<Double-1>', self._on_user_double_click)

        # Create scrollbar in tree_frame
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=tree_scroll.set)

        # Pack tree and scrollbar
        self.users_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # User details panel
        details_frame = ttk.LabelFrame(users_frame, text="User Details", padding=10)
        details_frame.pack(fill=tk.X, pady=(10, 0))

        # Form in two columns
        left_col = ttk.Frame(details_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        right_col = ttk.Frame(details_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Left column
        ttk.Label(left_col, text="Username:").pack(anchor=tk.W)
        self.user_details_vars['username'] = tk.StringVar()
        username_entry = ttk.Entry(left_col, textvariable=self.user_details_vars['username'], width=25)
        username_entry.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(left_col, text="Chat ID:").pack(anchor=tk.W)
        self.user_details_vars['chat_id'] = tk.StringVar()
        chat_id_entry = ttk.Entry(left_col, textvariable=self.user_details_vars['chat_id'], width=25)
        chat_id_entry.pack(fill=tk.X, pady=(0, 10))

        # Right column - notification preferences
        ttk.Label(right_col, text="Notification Preferences:").pack(anchor=tk.W)

        self.user_details_vars['human_alerts'] = tk.BooleanVar()
        ttk.Checkbutton(right_col, text="Human Detection Alerts",
                       variable=self.user_details_vars['human_alerts'],
                       command=self._update_user_details).pack(anchor=tk.W)

        self.user_details_vars['animal_alerts'] = tk.BooleanVar()
        ttk.Checkbutton(right_col, text="Animal Detection Alerts",
                       variable=self.user_details_vars['animal_alerts'],
                       command=self._update_user_details).pack(anchor=tk.W)

        self.user_details_vars['status'] = tk.StringVar()
        ttk.Label(right_col, text="Status:").pack(anchor=tk.W, pady=(10, 0))
        status_combo = ttk.Combobox(right_col, textvariable=self.user_details_vars['status'],
                                   values=["open", "close"], width=22, state="readonly")
        status_combo.pack(fill=tk.X)
        status_combo.bind('<<ComboboxSelected>>', lambda e: self._update_user_details())

        # Save button
        ttk.Button(right_col, text="💾 Save Changes", command=self._save_user_details).pack(pady=(10, 0))
    
    def _create_bot_config_tab(self, notebook):
        """Create bot configuration tab."""
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="🤖 Bot Config")
        
        # Bot token section
        token_frame = ttk.LabelFrame(config_frame, text="Bot Configuration", padding=20)
        token_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(token_frame, text="Bot Token:").pack(anchor=tk.W)
        self.bot_config_vars['token'] = tk.StringVar()
        self.bot_config_vars['token_hidden'] = tk.BooleanVar(value=True)
        token_entry = ttk.Entry(token_frame, textvariable=self.bot_config_vars['token'], width=60, show="*")
        token_entry.pack(fill=tk.X, pady=(5, 10))
        self.bot_config_vars['token_entry'] = token_entry

        token_buttons = ttk.Frame(token_frame)
        token_buttons.pack(fill=tk.X)

        ttk.Button(token_buttons, text="💾 Save Token", command=self._save_bot_token).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(token_buttons, text="🧪 Test Token", command=self._test_bot_token).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(token_buttons, text="👁️ Show/Hide", command=self._toggle_token_visibility).pack(side=tk.LEFT)
        
        # Bot settings
        settings_frame = ttk.LabelFrame(config_frame, text="Bot Settings", padding=20)
        settings_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Polling settings
        ttk.Label(settings_frame, text="Polling Interval (seconds):").pack(anchor=tk.W)
        self.bot_config_vars['polling_interval'] = tk.DoubleVar(value=1.0)

        # Frame for polling interval with value display
        polling_frame = ttk.Frame(settings_frame)
        polling_frame.pack(fill=tk.X, pady=(5, 10))

        polling_scale = ttk.Scale(polling_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                 variable=self.bot_config_vars['polling_interval'])
        polling_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.polling_value_label = ttk.Label(polling_frame, text=f"{self.bot_config_vars['polling_interval'].get():.1f}s")
        self.polling_value_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Update label when scale changes
        def update_polling_label(*args):
            self.polling_value_label.config(text=f"{self.bot_config_vars['polling_interval'].get():.1f}s")
        self.bot_config_vars['polling_interval'].trace('w', update_polling_label)

        # Command settings
        ttk.Label(settings_frame, text="Enabled Commands:").pack(anchor=tk.W)

        self.bot_config_vars['check_cmd'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="'check' - Manual photo capture",
                       variable=self.bot_config_vars['check_cmd']).pack(anchor=tk.W)

        self.bot_config_vars['status_cmd'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="'status' - System status",
                       variable=self.bot_config_vars['status_cmd']).pack(anchor=tk.W)

        self.bot_config_vars['help_cmd'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="'help' - Show help message",
                       variable=self.bot_config_vars['help_cmd']).pack(anchor=tk.W)
        
        # Notification settings
        notif_frame = ttk.LabelFrame(config_frame, text="Notification Settings", padding=20)
        notif_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        ttk.Label(notif_frame, text="Cooldown Period (seconds):").pack(anchor=tk.W)
        self.bot_config_vars['cooldown'] = tk.DoubleVar(value=20.0)

        # Frame for cooldown with value display
        cooldown_frame = ttk.Frame(notif_frame)
        cooldown_frame.pack(fill=tk.X, pady=(5, 10))

        cooldown_scale = ttk.Scale(cooldown_frame, from_=5, to=60, orient=tk.HORIZONTAL,
                                  variable=self.bot_config_vars['cooldown'])
        cooldown_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.cooldown_value_label = ttk.Label(cooldown_frame, text=f"{self.bot_config_vars['cooldown'].get():.0f}s")
        self.cooldown_value_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Update label when scale changes
        def update_cooldown_label(*args):
            self.cooldown_value_label.config(text=f"{self.bot_config_vars['cooldown'].get():.0f}s")
        self.bot_config_vars['cooldown'].trace('w', update_cooldown_label)

        # Unknown detection timer settings
        ttk.Label(notif_frame, text="Unknown Person Alert Timer (seconds):").pack(anchor=tk.W, pady=(15, 0))
        self.bot_config_vars['unknown_person_timer'] = tk.IntVar(value=5)

        unknown_person_frame = ttk.Frame(notif_frame)
        unknown_person_frame.pack(fill=tk.X, pady=(5, 10))

        unknown_person_scale = ttk.Scale(unknown_person_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                        variable=self.bot_config_vars['unknown_person_timer'])
        unknown_person_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.unknown_person_value_label = ttk.Label(unknown_person_frame, text=f"{self.bot_config_vars['unknown_person_timer'].get()}s")
        self.unknown_person_value_label.pack(side=tk.RIGHT, padx=(10, 0))

        def update_unknown_person_label(*args):
            self.unknown_person_value_label.config(text=f"{self.bot_config_vars['unknown_person_timer'].get()}s")

        self.bot_config_vars['unknown_person_timer'].trace('w', update_unknown_person_label)

        ttk.Label(notif_frame, text="Unknown Animal Alert Timer (seconds):").pack(anchor=tk.W, pady=(10, 0))
        self.bot_config_vars['unknown_animal_timer'] = tk.IntVar(value=5)

        unknown_animal_frame = ttk.Frame(notif_frame)
        unknown_animal_frame.pack(fill=tk.X, pady=(5, 10))

        unknown_animal_scale = ttk.Scale(unknown_animal_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                        variable=self.bot_config_vars['unknown_animal_timer'])
        unknown_animal_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.unknown_animal_value_label = ttk.Label(unknown_animal_frame, text=f"{self.bot_config_vars['unknown_animal_timer'].get()}s")
        self.unknown_animal_value_label.pack(side=tk.RIGHT, padx=(10, 0))

        def update_unknown_animal_label(*args):
            self.unknown_animal_value_label.config(text=f"{self.bot_config_vars['unknown_animal_timer'].get()}s")

        self.bot_config_vars['unknown_animal_timer'].trace('w', update_unknown_animal_label)

        self.bot_config_vars['send_photos'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(notif_frame, text="Include photos in notifications",
                       variable=self.bot_config_vars['send_photos']).pack(anchor=tk.W)

        ttk.Label(notif_frame, text="Photo Quality (1-100):").pack(anchor=tk.W)
        self.bot_config_vars['photo_quality'] = tk.DoubleVar(value=85.0)

        # Frame for photo quality with value display
        quality_frame = ttk.Frame(notif_frame)
        quality_frame.pack(fill=tk.X, pady=(5, 10))

        quality_scale = ttk.Scale(quality_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                 variable=self.bot_config_vars['photo_quality'])
        quality_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.quality_value_label = ttk.Label(quality_frame, text=f"{self.bot_config_vars['photo_quality'].get():.0f}%")
        self.quality_value_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Update label when scale changes
        def update_quality_label(*args):
            self.quality_value_label.config(text=f"{self.bot_config_vars['photo_quality'].get():.0f}%")
        self.bot_config_vars['photo_quality'].trace('w', update_quality_label)

        # Save all settings button
        ttk.Button(notif_frame, text="💾 Save All Settings", command=self._save_bot_config).pack(pady=(10, 0))
    
    def _create_test_tab(self, notebook):
        """Create testing tab."""
        test_frame = ttk.Frame(notebook)
        notebook.add(test_frame, text="🧪 Test")
        
        # Test message section
        message_frame = ttk.LabelFrame(test_frame, text="Send Test Message", padding=20)
        message_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(message_frame, text="Recipient:").pack(anchor=tk.W)
        self.test_vars['recipient'] = tk.StringVar(value="All Users")
        recipient_combo = ttk.Combobox(message_frame, textvariable=self.test_vars['recipient'],
                                      values=["All Users"], width=30, state="readonly")
        recipient_combo.pack(fill=tk.X, pady=(5, 10))
        self.test_vars['recipient_combo'] = recipient_combo

        ttk.Label(message_frame, text="Message:").pack(anchor=tk.W)
        message_text = tk.Text(message_frame, height=4, width=50)
        message_text.pack(fill=tk.X, pady=(5, 10))
        message_text.insert(tk.END, "🧪 This is a test message from the Intruder Detection System.")
        self.test_vars['message_text'] = message_text

        test_buttons = ttk.Frame(message_frame)
        test_buttons.pack(fill=tk.X)

        ttk.Button(test_buttons, text="📤 Send Message", command=self._send_test_message).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(test_buttons, text="📸 Send with Photo", command=self._send_test_message_with_photo).pack(side=tk.LEFT)
        
        # Test results
        results_frame = ttk.LabelFrame(test_frame, text="Test Results", padding=20)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.test_results_text = tk.Text(results_frame, height=15, font=("Courier", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.test_results_text.yview)
        self.test_results_text.configure(yscrollcommand=results_scrollbar.set)

        self.test_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Clear results button
        clear_button = ttk.Button(results_frame, text="🗑️ Clear Results", command=self._clear_test_results)
        clear_button.pack(side=tk.BOTTOM, pady=(10, 0))

        # Initial message
        self._log_test_result("Notification Center initialized. Ready for testing.")

    # Data loading methods
    def _load_users(self):
        """Load users from database."""
        if not self.database_manager:
            self._log_test_result("❌ Database manager not available")
            return

        if not self.users_tree:
            # Tree not created yet, skip loading
            return

        try:
            # Clear existing items
            for item in self.users_tree.get_children():
                self.users_tree.delete(item)

            # Load users from database
            users = self.database_manager.get_all_notification_settings()

            # Update recipient combo
            recipient_values = ["All Users"]

            for user in users:
                # Format status display
                status_display = "Active" if user.sendstatus == "open" else "Inactive"
                human_alerts = "✅" if user.notify_human_detection else "❌"
                animal_alerts = "✅" if user.notify_animal_detection else "❌"

                # Calculate last seen (placeholder for now)
                last_seen = "Unknown"
                if user.last_notification:
                    try:
                        if isinstance(user.last_notification, str):
                            last_notif = datetime.fromisoformat(user.last_notification.replace('Z', '+00:00'))
                        else:
                            last_notif = user.last_notification

                        time_diff = datetime.now() - last_notif.replace(tzinfo=None)
                        if time_diff.days > 0:
                            last_seen = f"{time_diff.days} days ago"
                        elif time_diff.seconds > 3600:
                            last_seen = f"{time_diff.seconds // 3600} hours ago"
                        else:
                            last_seen = f"{time_diff.seconds // 60} min ago"
                    except:
                        last_seen = "Unknown"

                # Insert into tree
                self.users_tree.insert("", tk.END, values=(
                    user.telegram_username,
                    user.chat_id,
                    human_alerts,
                    animal_alerts,
                    status_display,
                    last_seen
                ))

                # Add to recipient list
                recipient_values.append(user.telegram_username)

            # Update test tab recipient combo
            if 'recipient_combo' in self.test_vars:
                self.test_vars['recipient_combo']['values'] = recipient_values

            self._log_test_result(f"✅ Loaded {len(users)} users from database")

        except Exception as e:
            logger.error(f"Error loading users: {e}")
            self._log_test_result(f"❌ Error loading users: {e}")
            messagebox.showerror("Error", f"Failed to load users: {e}")

    def _load_bot_config(self):
        """Load bot configuration."""
        if not self.config_manager:
            # Set default values
            self.bot_config_vars['token'].set("7314134012:AAEn1bd8B4AbUja4rVAHZFGTqkbyJB-Envw")
            return

        try:
            # Load configuration from config manager
            # Use the correct method name - get() instead of get_config()

            # Set bot token
            bot_token = self.config_manager.get('telegram.bot_token', '')
            self.bot_config_vars['token'].set(bot_token)

            # Set other settings
            self.bot_config_vars['polling_interval'].set(self.config_manager.get('telegram.polling_interval', 1.0))
            self.bot_config_vars['cooldown'].set(self.config_manager.get('telegram.cooldown', 20.0))
            self.bot_config_vars['send_photos'].set(self.config_manager.get('telegram.send_photos', True))
            self.bot_config_vars['photo_quality'].set(self.config_manager.get('telegram.photo_quality', 85.0))

            # Command settings
            commands = self.config_manager.get('telegram.enabled_commands', ['check', 'status', 'help'])
            self.bot_config_vars['check_cmd'].set('check' in commands)
            self.bot_config_vars['status_cmd'].set('status' in commands)
            self.bot_config_vars['help_cmd'].set('help' in commands)

        except Exception as e:
            logger.error(f"Error loading bot config: {e}")
            self._log_test_result(f"❌ Error loading bot config: {e}")

    # User management event handlers
    def _on_user_select(self, event):
        """Handle user selection in tree."""
        selection = self.users_tree.selection()
        if not selection:
            self._clear_user_details()
            return

        item = self.users_tree.item(selection[0])
        values = item['values']

        if len(values) >= 5:
            # Update user details form
            self.user_details_vars['username'].set(values[0])
            self.user_details_vars['chat_id'].set(values[1])
            self.user_details_vars['human_alerts'].set(values[2] == "✅")
            self.user_details_vars['animal_alerts'].set(values[3] == "✅")
            self.user_details_vars['status'].set("open" if values[4] == "Active" else "close")

            # Store current selection
            self.selected_user = {
                'username': values[0],
                'chat_id': values[1],
                'human_alerts': values[2] == "✅",
                'animal_alerts': values[3] == "✅",
                'status': "open" if values[4] == "Active" else "close"
            }

    def _on_user_double_click(self, event):
        """Handle double-click on user (edit user)."""
        self._edit_user()

    def _clear_user_details(self):
        """Clear user details form."""
        for var in self.user_details_vars.values():
            if isinstance(var, tk.StringVar):
                var.set("")
            elif isinstance(var, tk.BooleanVar):
                var.set(False)
        self.selected_user = None

    def _update_user_details(self):
        """Update user details when form changes."""
        if not self.selected_user:
            return

        # This method is called when checkboxes or combobox change
        # We'll update the database immediately for real-time changes
        self._save_user_details()

    def _save_user_details(self):
        """Save user details to database."""
        if not self.selected_user or not self.database_manager:
            return

        try:
            # Get current values from form
            username = self.user_details_vars['username'].get()
            chat_id = self.user_details_vars['chat_id'].get()
            human_alerts = self.user_details_vars['human_alerts'].get()
            animal_alerts = self.user_details_vars['animal_alerts'].get()
            status = self.user_details_vars['status'].get()

            if not chat_id:
                return

            # Create notification settings object
            from database.models import NotificationSettings
            settings = NotificationSettings(
                chat_id=int(chat_id),
                telegram_username=username,
                notify_human_detection=human_alerts,
                notify_animal_detection=animal_alerts,
                sendstatus=status
            )

            # Update in database
            success = self.database_manager.update_notification_settings(settings)

            if success:
                # Update notification system if available
                if self.notification_system:
                    user_data = {
                        'chat_id': int(chat_id),
                        'telegram_username': username,
                        'notify_human_detection': human_alerts,
                        'notify_animal_detection': animal_alerts,
                        'sendstatus': status
                    }
                    self.notification_system.add_user(user_data)

                # Refresh the tree view
                self._load_users()
                self._log_test_result(f"✅ Updated user {username}")
            else:
                self._log_test_result(f"❌ Failed to update user {username}")

        except Exception as e:
            logger.error(f"Error saving user details: {e}")
            self._log_test_result(f"❌ Error saving user: {e}")
            messagebox.showerror("Error", f"Failed to save user details: {e}")

    def _add_user(self):
        """Add a new user."""
        self._show_user_dialog()

    def _edit_user(self):
        """Edit selected user."""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a user to edit.")
            return

        item = self.users_tree.item(selection[0])
        values = item['values']

        if len(values) >= 5:
            user_data = {
                'username': values[0],
                'chat_id': values[1],
                'human_alerts': values[2] == "✅",
                'animal_alerts': values[3] == "✅",
                'status': "open" if values[4] == "Active" else "close"
            }
            self._show_user_dialog(user_data)

    def _remove_user(self):
        """Remove selected user."""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a user to remove.")
            return

        item = self.users_tree.item(selection[0])
        values = item['values']
        username = values[0]
        chat_id = values[1]

        if messagebox.askyesno("Confirm Deletion",
                              f"Are you sure you want to remove user '{username}'?\n"
                              f"Chat ID: {chat_id}"):
            try:
                if self.database_manager:
                    success = self.database_manager.delete_notification_settings(int(chat_id))
                    if success:
                        # Also remove from notification system
                        if self.notification_system:
                            self.notification_system.remove_user(int(chat_id))

                        self._load_users()
                        self._clear_user_details()
                        self._log_test_result(f"✅ Removed user {username}")
                        messagebox.showinfo("Success", f"User '{username}' removed successfully.")
                    else:
                        self._log_test_result(f"❌ Failed to remove user {username}")
                        messagebox.showerror("Error", f"Failed to remove user '{username}'.")
                else:
                    messagebox.showerror("Error", "Database manager not available.")

            except Exception as e:
                logger.error(f"Error removing user: {e}")
                self._log_test_result(f"❌ Error removing user: {e}")
                messagebox.showerror("Error", f"Failed to remove user: {e}")

    def _test_user_message(self):
        """Send test message to selected user."""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a user to test.")
            return

        item = self.users_tree.item(selection[0])
        values = item['values']
        username = values[0]
        chat_id = values[1]

        self._send_test_to_user(username, chat_id)

    def _show_user_dialog(self, user_data=None):
        """Show add/edit user dialog."""
        dialog = tk.Toplevel()
        dialog.title("Add User" if user_data is None else "Edit User")
        dialog.geometry("500x400")
        dialog.transient()
        dialog.grab_set()
        dialog.resizable(False, False)

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"500x400+{x}+{y}")

        # Form variables
        username_var = tk.StringVar(value=user_data['username'] if user_data else "")
        chat_id_var = tk.StringVar(value=str(user_data['chat_id']) if user_data else "")
        human_alerts_var = tk.BooleanVar(value=user_data.get('human_alerts', True) if user_data else True)
        animal_alerts_var = tk.BooleanVar(value=user_data.get('animal_alerts', True) if user_data else True)
        status_var = tk.StringVar(value=user_data.get('status', 'open') if user_data else 'open')

        # Form layout
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Username
        ttk.Label(main_frame, text="Username:").pack(anchor=tk.W)
        username_entry = ttk.Entry(main_frame, textvariable=username_var, width=40)
        username_entry.pack(fill=tk.X, pady=(5, 10))

        # Chat ID
        ttk.Label(main_frame, text="Chat ID:").pack(anchor=tk.W)
        chat_id_entry = ttk.Entry(main_frame, textvariable=chat_id_var, width=40)
        chat_id_entry.pack(fill=tk.X, pady=(5, 10))

        # Notification preferences
        ttk.Label(main_frame, text="Notification Preferences:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Checkbutton(main_frame, text="Human Detection Alerts", variable=human_alerts_var).pack(anchor=tk.W)
        ttk.Checkbutton(main_frame, text="Animal Detection Alerts", variable=animal_alerts_var).pack(anchor=tk.W)

        # Status
        ttk.Label(main_frame, text="Status:").pack(anchor=tk.W, pady=(10, 5))
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        ttk.Radiobutton(status_frame, text="Active", variable=status_var, value="open").pack(side=tk.LEFT)
        ttk.Radiobutton(status_frame, text="Inactive", variable=status_var, value="close").pack(side=tk.LEFT, padx=(20, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        def save_user():
            username = username_var.get().strip()
            chat_id_str = chat_id_var.get().strip()

            if not username or not chat_id_str:
                messagebox.showerror("Validation Error", "Username and Chat ID are required.")
                return

            try:
                chat_id = int(chat_id_str)
            except ValueError:
                messagebox.showerror("Validation Error", "Chat ID must be a number.")
                return

            try:
                from database.models import NotificationSettings
                settings = NotificationSettings(
                    chat_id=chat_id,
                    telegram_username=username,
                    notify_human_detection=human_alerts_var.get(),
                    notify_animal_detection=animal_alerts_var.get(),
                    sendstatus=status_var.get()
                )

                if user_data is None:
                    # Add new user
                    settings_id = self.database_manager.create_notification_settings(settings)
                    if settings_id:
                        self._log_test_result(f"✅ Added user {username}")
                        messagebox.showinfo("Success", "User added successfully.")
                    else:
                        self._log_test_result(f"❌ Failed to add user {username}")
                        messagebox.showerror("Error", "Failed to add user.")
                        return
                else:
                    # Update existing user
                    success = self.database_manager.update_notification_settings(settings)
                    if success:
                        self._log_test_result(f"✅ Updated user {username}")
                        messagebox.showinfo("Success", "User updated successfully.")
                    else:
                        self._log_test_result(f"❌ Failed to update user {username}")
                        messagebox.showerror("Error", "Failed to update user.")
                        return

                # Save to database first
                if self.database_manager:
                    from database.models import NotificationSettings
                    settings = NotificationSettings(
                        chat_id=chat_id,
                        telegram_username=username,
                        notify_human_detection=human_alerts_var.get(),
                        notify_animal_detection=animal_alerts_var.get(),
                        sendstatus=status_var.get()
                    )

                    # Check if user exists, update or create
                    existing = self.database_manager.get_notification_settings(chat_id)
                    if existing:
                        self.database_manager.update_notification_settings(settings)
                    else:
                        self.database_manager.create_notification_settings(settings)

                # Also update notification system for in-memory management
                if self.notification_system:
                    user_data_dict = {
                        'chat_id': chat_id,
                        'telegram_username': username,
                        'notify_human_detection': human_alerts_var.get(),
                        'notify_animal_detection': animal_alerts_var.get(),
                        'sendstatus': status_var.get()
                    }
                    self.notification_system.add_user(user_data_dict)

                # Refresh and close
                dialog.destroy()
                self._load_users()  # Refresh after dialog closes

            except Exception as e:
                logger.error(f"Error saving user: {e}")
                self._log_test_result(f"❌ Error saving user: {e}")
                messagebox.showerror("Error", f"Failed to save user: {e}")

        ttk.Button(button_frame, text="Save", command=save_user).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

        # Focus on username entry
        username_entry.focus_set()

    # Bot configuration methods
    def _save_bot_token(self):
        """Save bot token to configuration."""
        token = self.bot_config_vars['token'].get().strip()
        if not token:
            messagebox.showerror("Validation Error", "Bot token is required.")
            return

        try:
            if self.config_manager:
                # Use the set method to update the configuration
                self.config_manager.set('telegram.bot_token', token)
                self.config_manager.save_config()

            self._log_test_result("✅ Bot token saved successfully")
            messagebox.showinfo("Success", "Bot token saved successfully.")

        except Exception as e:
            logger.error(f"Error saving bot token: {e}")
            self._log_test_result(f"❌ Error saving bot token: {e}")
            messagebox.showerror("Error", f"Failed to save bot token: {e}")

    def _test_bot_token(self):
        """Test bot token validity."""
        token = self.bot_config_vars['token'].get().strip()
        if not token:
            messagebox.showerror("Validation Error", "Bot token is required.")
            return

        self._log_test_result("🧪 Testing bot token...")

        def test_token():
            try:
                # Test bot token by getting bot info
                url = f"https://api.telegram.org/bot{token}/getMe"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        bot_info = data.get('result', {})
                        bot_name = bot_info.get('first_name', 'Unknown')
                        bot_username = bot_info.get('username', 'Unknown')

                        self._log_test_result(f"✅ Bot token valid")
                        self._log_test_result(f"   Bot name: {bot_name}")
                        self._log_test_result(f"   Bot username: @{bot_username}")

                        messagebox.showinfo("Success", f"Bot token is valid!\n\nBot: {bot_name}\nUsername: @{bot_username}")
                    else:
                        error_msg = data.get('description', 'Unknown error')
                        self._log_test_result(f"❌ Bot token invalid: {error_msg}")
                        messagebox.showerror("Error", f"Bot token is invalid: {error_msg}")
                else:
                    self._log_test_result(f"❌ HTTP error {response.status_code}")
                    messagebox.showerror("Error", f"HTTP error {response.status_code}")

            except requests.exceptions.Timeout:
                self._log_test_result("❌ Request timeout")
                messagebox.showerror("Error", "Request timeout. Please check your internet connection.")
            except Exception as e:
                self._log_test_result(f"❌ Error testing token: {e}")
                messagebox.showerror("Error", f"Error testing token: {e}")

        # Run test in background thread
        threading.Thread(target=test_token, daemon=True).start()

    def _toggle_token_visibility(self):
        """Toggle bot token visibility."""
        entry = self.bot_config_vars['token_entry']
        if self.bot_config_vars['token_hidden'].get():
            entry.config(show="")
            self.bot_config_vars['token_hidden'].set(False)
        else:
            entry.config(show="*")
            self.bot_config_vars['token_hidden'].set(True)

    def _save_bot_config(self):
        """Save all bot configuration settings."""
        try:
            if self.config_manager:
                # Use the set method to update all configuration values
                self.config_manager.set('telegram.bot_token', self.bot_config_vars['token'].get())
                self.config_manager.set('telegram.polling_interval', self.bot_config_vars['polling_interval'].get())
                self.config_manager.set('telegram.cooldown', self.bot_config_vars['cooldown'].get())
                self.config_manager.set('telegram.send_photos', self.bot_config_vars['send_photos'].get())
                self.config_manager.set('telegram.photo_quality', self.bot_config_vars['photo_quality'].get())

                # Enabled commands
                enabled_commands = []
                if self.bot_config_vars['check_cmd'].get():
                    enabled_commands.append('check')
                if self.bot_config_vars['status_cmd'].get():
                    enabled_commands.append('status')
                if self.bot_config_vars['help_cmd'].get():
                    enabled_commands.append('help')
                self.config_manager.set('telegram.enabled_commands', enabled_commands)

                # Save the configuration
                self.config_manager.save_config()

            self._log_test_result("✅ Bot configuration saved successfully")
            messagebox.showinfo("Success", "Bot configuration saved successfully.")

        except Exception as e:
            logger.error(f"Error saving bot config: {e}")
            self._log_test_result(f"❌ Error saving bot config: {e}")
            messagebox.showerror("Error", f"Failed to save bot configuration: {e}")

    # Test message methods
    def _send_test_message(self):
        """Send test message without photo."""
        self._send_test_message_internal(with_photo=False)

    def _send_test_message_with_photo(self):
        """Send test message with photo."""
        self._send_test_message_internal(with_photo=True)

    def _send_test_message_internal(self, with_photo=False):
        """Internal method to send test messages."""
        recipient = self.test_vars['recipient'].get()
        message = self.test_vars['message_text'].get("1.0", tk.END).strip()

        if not message:
            messagebox.showerror("Validation Error", "Message is required.")
            return

        token = self.bot_config_vars['token'].get().strip()
        if not token:
            messagebox.showerror("Configuration Error", "Bot token is required. Please configure it in the Bot Config tab.")
            return

        self._log_test_result(f"🧪 Sending test message to {recipient}...")
        if with_photo:
            self._log_test_result("   Including photo attachment")

        def send_message():
            try:
                sent_count = 0
                failed_count = 0

                if recipient == "All Users":
                    # Send to all active users
                    if self.database_manager:
                        users = self.database_manager.get_all_notification_settings()
                        active_users = [u for u in users if u.sendstatus == 'open']

                        for user in active_users:
                            success = self._send_to_chat_id(token, user.chat_id, message, with_photo)
                            if success:
                                sent_count += 1
                                self._log_test_result(f"   ✅ Sent to {user.telegram_username} ({user.chat_id})")
                            else:
                                failed_count += 1
                                self._log_test_result(f"   ❌ Failed to send to {user.telegram_username} ({user.chat_id})")
                    else:
                        self._log_test_result("❌ Database manager not available")
                        return
                else:
                    # Send to specific user
                    if self.database_manager:
                        users = self.database_manager.get_all_notification_settings()
                        target_user = next((u for u in users if u.telegram_username == recipient), None)

                        if target_user:
                            success = self._send_to_chat_id(token, target_user.chat_id, message, with_photo)
                            if success:
                                sent_count = 1
                                self._log_test_result(f"   ✅ Sent to {recipient} ({target_user.chat_id})")
                            else:
                                failed_count = 1
                                self._log_test_result(f"   ❌ Failed to send to {recipient} ({target_user.chat_id})")
                        else:
                            self._log_test_result(f"❌ User {recipient} not found")
                            return
                    else:
                        self._log_test_result("❌ Database manager not available")
                        return

                # Summary
                total = sent_count + failed_count
                self._log_test_result(f"📊 Results: {sent_count}/{total} messages sent successfully")

                if sent_count > 0:
                    messagebox.showinfo("Success", f"Test message sent successfully!\n\nSent: {sent_count}\nFailed: {failed_count}")
                else:
                    messagebox.showerror("Error", "Failed to send test message to any recipients.")

            except Exception as e:
                self._log_test_result(f"❌ Error sending test message: {e}")
                messagebox.showerror("Error", f"Error sending test message: {e}")

        # Run in background thread
        threading.Thread(target=send_message, daemon=True).start()

    def _send_test_to_user(self, username, chat_id):
        """Send test message to specific user."""
        token = self.bot_config_vars['token'].get().strip()
        if not token:
            messagebox.showerror("Configuration Error", "Bot token is required. Please configure it in the Bot Config tab.")
            return

        message = f"🧪 Test notification for {username}\n\nThis is a test message from the Intruder Detection System."

        self._log_test_result(f"🧪 Sending test message to {username} ({chat_id})...")

        def send_message():
            try:
                success = self._send_to_chat_id(token, chat_id, message, with_photo=False)
                if success:
                    self._log_test_result(f"   ✅ Test message sent to {username}")
                    messagebox.showinfo("Success", f"Test message sent successfully to {username}!")
                else:
                    self._log_test_result(f"   ❌ Failed to send test message to {username}")
                    messagebox.showerror("Error", f"Failed to send test message to {username}.")

            except Exception as e:
                self._log_test_result(f"❌ Error sending test message: {e}")
                messagebox.showerror("Error", f"Error sending test message: {e}")

        # Run in background thread
        threading.Thread(target=send_message, daemon=True).start()

    def _send_to_chat_id(self, token, chat_id, message, with_photo=False):
        """Send message to specific chat ID."""
        try:
            if with_photo:
                # For demo purposes, we'll send a placeholder photo
                # In real implementation, this would use an actual photo file
                url = f'https://api.telegram.org/bot{token}/sendMessage'
                payload = {
                    'chat_id': chat_id,
                    'text': f"📸 {message}\n\n(Photo would be attached in real implementation)"
                }
            else:
                url = f'https://api.telegram.org/bot{token}/sendMessage'
                payload = {
                    'chat_id': chat_id,
                    'text': message
                }

            response = requests.post(url, data=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('ok', False)
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                return False

        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    def _clear_test_results(self):
        """Clear test results text area."""
        self.test_results_text.config(state=tk.NORMAL)
        self.test_results_text.delete(1.0, tk.END)
        self.test_results_text.config(state=tk.DISABLED)
        self._log_test_result("Test results cleared.")

    def _log_test_result(self, message):
        """Log a message to the test results area."""
        if not self.test_results_text:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        self.test_results_text.config(state=tk.NORMAL)
        self.test_results_text.insert(tk.END, log_message)
        self.test_results_text.see(tk.END)
        self.test_results_text.config(state=tk.DISABLED)

    # Public methods for integration
    def set_database_manager(self, database_manager):
        """Set the database manager."""
        self.database_manager = database_manager
        if hasattr(self, 'users_tree') and self.users_tree:
            self._load_users()

    def set_notification_system(self, notification_system):
        """Set the notification system."""
        self.notification_system = notification_system

    def set_config_manager(self, config_manager):
        """Set the configuration manager."""
        self.config_manager = config_manager
        if hasattr(self, 'bot_config_vars') and self.bot_config_vars:
            self._load_bot_config()

    def refresh_data(self):
        """Refresh all data from database."""
        if hasattr(self, 'users_tree') and self.users_tree and self.database_manager:
            self._load_users()
        if hasattr(self, 'bot_config_vars') and self.bot_config_vars:
            self._load_bot_config()

    def force_refresh_users(self):
        """Force refresh users table even if conditions aren't met."""
        if self.database_manager:
            self._load_users()
