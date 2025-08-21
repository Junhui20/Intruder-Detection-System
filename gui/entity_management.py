"""
Entity Management Module

Human and animal registration interface with photo upload and management.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import os
import json
import csv
import shutil
import cv2
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import backend components
from database.database_manager import DatabaseManager
from database.models import WhitelistEntry
from core.camera_manager import CameraManager

logger = logging.getLogger(__name__)


class EntityManagement:
    """Entity management interface for humans and animals."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None, camera_manager: Optional[CameraManager] = None):
        self.title = "👥 Entity Management"
        self.db_manager = db_manager or DatabaseManager()
        self.camera_manager = camera_manager or CameraManager()

        # GUI components
        self.humans_tree = None
        self.animals_tree = None
        self.human_form_vars = {}
        self.animal_form_vars = {}

        # Camera capture variables
        self.capture_window = None
        self.capture_label = None
        self.capture_active = False
        self.capture_type = None  # 'human' or 'animal'

        # Common color options
        self.common_colors = [
            "Black", "White", "Brown", "Golden", "Gray", "Beige",
            "Red", "Orange", "Yellow", "Green", "Blue", "Purple",
            "Pink", "Silver", "Cream", "Tan", "Mixed"
        ]

        # Animal types
        self.animal_types = [
            "Dog", "Cat", "Horse", "Sheep", "Cow", "Elephant",
            "Bear", "Zebra", "Bird", "Rabbit", "Goat", "Pig"
        ]

        # Load existing data
        self._load_entities()

    def get_title(self):
        return self.title

    def _load_entities(self):
        """Load entities from database."""
        try:
            self.humans_data = self.db_manager.get_whitelist_entries(entity_type='human')
            self.animals_data = self.db_manager.get_whitelist_entries(entity_type='animal')
        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            self.humans_data = []
            self.animals_data = []

    def show_in_frame(self, parent_frame):
        """Display the entity management interface."""
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Humans tab
        self._create_humans_tab(notebook)

        # Animals tab
        self._create_animals_tab(notebook)

        # Bulk operations tab
        self._create_bulk_tab(notebook)
    
    def _create_humans_tab(self, notebook):
        """Create humans management tab."""
        humans_frame = ttk.Frame(notebook)
        notebook.add(humans_frame, text="👤 Humans")

        # Left panel - list
        left_panel = ttk.LabelFrame(humans_frame, text="Known Humans", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Human list
        columns = ("Name", "Photos", "Confidence", "Last Seen")
        self.humans_tree = ttk.Treeview(left_panel, columns=columns, show="headings", height=20)

        for col in columns:
            self.humans_tree.heading(col, text=col)
            self.humans_tree.column(col, width=100)

        # Bind selection event
        self.humans_tree.bind('<<TreeviewSelect>>', self._on_human_select)

        # Load human data
        self._refresh_humans_list()

        self.humans_tree.pack(fill=tk.BOTH, expand=True)

        # Right panel - details/add
        right_panel = ttk.LabelFrame(humans_frame, text="Human Details", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialize form variables
        self.human_form_vars = {
            'name': tk.StringVar(),
            'photo_path': tk.StringVar(),
            'confidence': tk.DoubleVar(value=0.6),
            'selected_id': tk.StringVar()
        }

        # Form fields
        ttk.Label(right_panel, text="Name:").pack(anchor=tk.W)
        name_entry = ttk.Entry(right_panel, textvariable=self.human_form_vars['name'], width=25)
        name_entry.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(right_panel, text="Photo:").pack(anchor=tk.W)
        photo_frame = ttk.Frame(right_panel)
        photo_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(photo_frame, text="📁 Browse", command=self._browse_human_photo).pack(side=tk.LEFT)
        ttk.Button(photo_frame, text="📷 Capture", command=self._capture_human_photo).pack(side=tk.RIGHT)

        # Photo path display
        photo_label = ttk.Label(right_panel, textvariable=self.human_form_vars['photo_path'],
                               foreground="blue", wraplength=200)
        photo_label.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(right_panel, text="Confidence Threshold:").pack(anchor=tk.W)

        # Confidence threshold with percentage display
        conf_frame = ttk.Frame(right_panel)
        conf_frame.pack(fill=tk.X, pady=(0, 10))

        self.human_conf_label = ttk.Label(conf_frame, text="60%")
        self.human_conf_label.pack(side=tk.RIGHT)

        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                              variable=self.human_form_vars['confidence'],
                              command=self._update_human_conf_display)
        conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Buttons
        ttk.Button(right_panel, text="💾 Save", command=self._save_human,
                  style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(right_panel, text="🗑️ Delete", command=self._delete_human).pack(fill=tk.X, pady=2)
        ttk.Button(right_panel, text="🔄 Reset", command=self._reset_human_form).pack(fill=tk.X, pady=2)
    
    def _create_animals_tab(self, notebook):
        """Create animals management tab."""
        animals_frame = ttk.Frame(notebook)
        notebook.add(animals_frame, text="🐕 Animals")

        # Similar structure to humans but with animal-specific fields
        left_panel = ttk.LabelFrame(animals_frame, text="Known Animals", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        columns = ("Name", "Type", "Color", "Individual ID", "Method")
        self.animals_tree = ttk.Treeview(left_panel, columns=columns, show="headings", height=20)

        for col in columns:
            self.animals_tree.heading(col, text=col)
            self.animals_tree.column(col, width=80)

        # Bind selection event
        self.animals_tree.bind('<<TreeviewSelect>>', self._on_animal_select)

        # Load animal data
        self._refresh_animals_list()

        self.animals_tree.pack(fill=tk.BOTH, expand=True)

        # Right panel for animal details
        right_panel = ttk.LabelFrame(animals_frame, text="Animal Details", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialize animal form variables
        self.animal_form_vars = {
            'name': tk.StringVar(),
            'individual_id': tk.StringVar(),
            'animal_type': tk.StringVar(),
            'color': tk.StringVar(),
            'identification_method': tk.StringVar(value="Hybrid"),
            'photo_path': tk.StringVar(),
            'selected_id': tk.StringVar()
        }

        # Animal-specific fields
        ttk.Label(right_panel, text="Name:").pack(anchor=tk.W)
        ttk.Label(right_panel, text="(Display name for the animal)", font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        name_entry = ttk.Entry(right_panel, textvariable=self.animal_form_vars['name'], width=25)
        name_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(right_panel, text="Individual ID (e.g., 'jacky'):").pack(anchor=tk.W)
        ttk.Label(right_panel, text="(Unique identifier for specific pet)", font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        id_entry = ttk.Entry(right_panel, textvariable=self.animal_form_vars['individual_id'], width=25)
        id_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(right_panel, text="Animal Type:").pack(anchor=tk.W)
        type_combo = ttk.Combobox(right_panel, textvariable=self.animal_form_vars['animal_type'],
                                 values=self.animal_types, width=22, state="readonly")
        type_combo.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(right_panel, text="Color:").pack(anchor=tk.W)
        color_combo = ttk.Combobox(right_panel, textvariable=self.animal_form_vars['color'],
                                  values=self.common_colors, width=22)
        color_combo.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(right_panel, text="Identification Method:").pack(anchor=tk.W)
        ttk.Label(right_panel, text="(How to recognize this animal)", font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        method_combo = ttk.Combobox(right_panel, textvariable=self.animal_form_vars['identification_method'],
                                   values=["Color", "Face", "Hybrid"], width=22, state="readonly")
        method_combo.pack(fill=tk.X, pady=(0, 10))

        # Photo management
        ttk.Label(right_panel, text="Photos:").pack(anchor=tk.W)
        photo_frame = ttk.Frame(right_panel)
        photo_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(photo_frame, text="📁 Add", command=self._browse_animal_photo).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(photo_frame, text="� Capture", command=self._capture_animal_photo).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(photo_frame, text="�👁️ View", command=self._view_animal_photo).pack(side=tk.LEFT)

        # Photo path display
        photo_label = ttk.Label(right_panel, textvariable=self.animal_form_vars['photo_path'],
                               foreground="blue", wraplength=200)
        photo_label.pack(fill=tk.X, pady=(0, 10))

        # Buttons
        ttk.Button(right_panel, text="💾 Save Pet", command=self._save_animal,
                  style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(right_panel, text="🧪 Test Recognition", command=self._test_animal_recognition).pack(fill=tk.X, pady=2)
        ttk.Button(right_panel, text="🗑️ Delete", command=self._delete_animal).pack(fill=tk.X, pady=2)
    
    def _create_bulk_tab(self, notebook):
        """Create bulk operations tab."""
        bulk_frame = ttk.Frame(notebook)
        notebook.add(bulk_frame, text="📦 Bulk Operations")

        # Import/Export section
        import_frame = ttk.LabelFrame(bulk_frame, text="Import/Export", padding=20)
        import_frame.pack(fill=tk.X, padx=20, pady=20)

        ttk.Button(import_frame, text="📥 Import from CSV", command=self._import_from_csv).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(import_frame, text="📤 Export to CSV", command=self._export_to_csv).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(import_frame, text="💾 Backup Database", command=self._backup_database).pack(side=tk.LEFT)

        # Batch processing
        batch_frame = ttk.LabelFrame(bulk_frame, text="Batch Processing", padding=20)
        batch_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        ttk.Button(batch_frame, text="🔄 Regenerate Face Encodings", command=self._regenerate_encodings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(batch_frame, text="🧹 Clean Invalid Entries", command=self._clean_invalid_entries).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(batch_frame, text="📊 Generate Report", command=self._generate_report).pack(side=tk.LEFT)

    # Human management methods
    def _refresh_humans_list(self):
        """Refresh the humans list from database."""
        # Clear existing items
        for item in self.humans_tree.get_children():
            self.humans_tree.delete(item)

        # Load from database
        try:
            humans = self.db_manager.get_whitelist_entries(entity_type='human')
            for human in humans:
                # Count photos (main + additional)
                photo_count = 1 if human.image_path else 0
                if human.multiple_photos:
                    try:
                        additional_photos = json.loads(human.multiple_photos)
                        photo_count += len(additional_photos)
                    except:
                        pass

                # Format confidence as percentage
                confidence = f"{human.confidence_threshold * 100:.1f}%"

                # Calculate last seen (placeholder for now)
                last_seen = "Never"
                if human.updated_at:
                    time_diff = datetime.now() - human.updated_at
                    if time_diff.days == 0:
                        if time_diff.seconds < 3600:
                            last_seen = f"{time_diff.seconds // 60} min ago"
                        else:
                            last_seen = f"{time_diff.seconds // 3600} hour ago"
                    else:
                        last_seen = f"{time_diff.days} days ago"

                self.humans_tree.insert("", tk.END, iid=human.id,
                                       values=(human.name, photo_count, confidence, last_seen))
        except Exception as e:
            logger.error(f"Failed to refresh humans list: {e}")
            messagebox.showerror("Error", f"Failed to load humans: {e}")

    def _on_human_select(self, event):
        """Handle human selection in tree."""
        selection = self.humans_tree.selection()
        if not selection:
            return

        try:
            human_id = selection[0]
            human = self.db_manager.get_whitelist_entry(int(human_id))
            if human:
                self.human_form_vars['selected_id'].set(str(human.id))
                self.human_form_vars['name'].set(human.name)
                self.human_form_vars['photo_path'].set(human.image_path)
                self.human_form_vars['confidence'].set(human.confidence_threshold)
                self._update_human_conf_display()
        except Exception as e:
            logger.error(f"Failed to load human details: {e}")

    def _update_human_conf_display(self, value=None):
        """Update human confidence threshold display."""
        conf_value = self.human_form_vars['confidence'].get()
        self.human_conf_label.config(text=f"{conf_value * 100:.0f}%")

    def _browse_human_photo(self):
        """Browse for human photo."""
        file_path = filedialog.askopenfilename(
            title="Select Human Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.human_form_vars['photo_path'].set(file_path)

    def _capture_human_photo(self):
        """Capture human photo from camera."""
        self.capture_type = 'human'
        self._open_camera_capture_window()

    def _save_human(self):
        """Save human to database."""
        try:
            name = self.human_form_vars['name'].get().strip()
            photo_path = self.human_form_vars['photo_path'].get().strip()
            confidence = self.human_form_vars['confidence'].get()
            selected_id = self.human_form_vars['selected_id'].get()

            if not name:
                messagebox.showwarning("Validation Error", "Please enter a name.")
                return

            if not photo_path or not os.path.exists(photo_path):
                messagebox.showwarning("Validation Error", "Please select a valid photo.")
                return

            # Compute face encodings from the image
            face_encodings_data = None
            try:
                # Use the core face recognition system to compute encodings
                from core.face_recognition import FaceRecognitionSystem
                import pickle

                # Create a temporary face recognition system to compute encodings
                temp_face_system = FaceRecognitionSystem()

                # Use the same method as the main system
                temp_face_system._load_face_from_image(name, photo_path)

                # Extract the encodings that were just loaded
                if temp_face_system.known_face_encodings:
                    # Get the encodings for this person (they will be the last ones added)
                    person_encodings = []
                    for i, face_name in enumerate(temp_face_system.known_face_names):
                        if face_name == name:
                            person_encodings.append(temp_face_system.known_face_encodings[i])

                    if person_encodings:
                        # Serialize the encodings for database storage
                        face_encodings_data = pickle.dumps(person_encodings)
                        logger.info(f"Computed {len(person_encodings)} face encodings for {name}")
                    else:
                        logger.warning(f"No face encodings computed for {name}")
                        # Continue saving without encodings
                else:
                    logger.warning(f"No faces detected in image for {name}")
                    # Continue saving without encodings

            except Exception as e:
                logger.error(f"Error computing face encodings for {name}: {e}")
                # Continue saving without encodings - face recognition can be fixed later

            # Create or update human entry
            human = WhitelistEntry(
                id=int(selected_id) if selected_id else None,
                name=name,
                entity_type='human',
                image_path=photo_path,
                confidence_threshold=confidence,
                face_encodings=face_encodings_data
            )

            if selected_id:
                self.db_manager.update_whitelist_entry(human)
                messagebox.showinfo("Success", "Human updated successfully!")
            else:
                self.db_manager.create_whitelist_entry(human)
                messagebox.showinfo("Success", "Human added successfully!")

            self._refresh_humans_list()
            self._reset_human_form()

        except Exception as e:
            logger.error(f"Failed to save human: {e}")
            messagebox.showerror("Error", f"Failed to save human: {e}")

    def _delete_human(self):
        """Delete selected human."""
        selected_id = self.human_form_vars['selected_id'].get()
        if not selected_id:
            messagebox.showwarning("Selection Error", "Please select a human to delete.")
            return

        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this human?"):
            try:
                self.db_manager.delete_whitelist_entry(int(selected_id))
                messagebox.showinfo("Success", "Human deleted successfully!")
                self._refresh_humans_list()
                self._reset_human_form()
            except Exception as e:
                logger.error(f"Failed to delete human: {e}")
                messagebox.showerror("Error", f"Failed to delete human: {e}")

    def _reset_human_form(self):
        """Reset human form."""
        for var in self.human_form_vars.values():
            if isinstance(var, tk.StringVar):
                var.set("")
            elif isinstance(var, tk.DoubleVar):
                var.set(0.6)
        self._update_human_conf_display()

    # Animal management methods
    def _refresh_animals_list(self):
        """Refresh the animals list from database."""
        # Clear existing items
        for item in self.animals_tree.get_children():
            self.animals_tree.delete(item)

        # Load from database
        try:
            animals = self.db_manager.get_whitelist_entries(entity_type='animal')
            for animal in animals:
                # Get animal type name
                animal_type = "Unknown"
                if animal.coco_class_id:
                    # Map COCO class ID to animal type name
                    coco_to_animal = {
                        16: "Dog", 17: "Cat", 18: "Horse", 19: "Sheep",
                        20: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra"
                    }
                    animal_type = coco_to_animal.get(animal.coco_class_id, "Unknown")

                self.animals_tree.insert("", tk.END, iid=animal.id,
                                        values=(animal.name, animal_type, animal.color or "",
                                               animal.individual_id or "", animal.identification_method))
        except Exception as e:
            logger.error(f"Failed to refresh animals list: {e}")
            messagebox.showerror("Error", f"Failed to load animals: {e}")

    def _on_animal_select(self, event):
        """Handle animal selection in tree."""
        selection = self.animals_tree.selection()
        if not selection:
            return

        try:
            animal_id = selection[0]
            animal = self.db_manager.get_whitelist_entry(int(animal_id))
            if animal:
                self.animal_form_vars['selected_id'].set(str(animal.id))
                self.animal_form_vars['name'].set(animal.name)
                self.animal_form_vars['individual_id'].set(animal.individual_id or "")
                self.animal_form_vars['color'].set(animal.color or "")
                self.animal_form_vars['identification_method'].set(animal.identification_method)
                self.animal_form_vars['photo_path'].set(animal.image_path)

                # Set animal type
                if animal.coco_class_id:
                    coco_to_animal = {
                        16: "Dog", 17: "Cat", 18: "Horse", 19: "Sheep",
                        20: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra"
                    }
                    animal_type = coco_to_animal.get(animal.coco_class_id, "Dog")
                    self.animal_form_vars['animal_type'].set(animal_type)
        except Exception as e:
            logger.error(f"Failed to load animal details: {e}")

    def _browse_animal_photo(self):
        """Browse for animal photo."""
        file_path = filedialog.askopenfilename(
            title="Select Animal Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.animal_form_vars['photo_path'].set(file_path)

    def _capture_animal_photo(self):
        """Capture animal photo from camera."""
        self.capture_type = 'animal'
        self._open_camera_capture_window()

    def _view_animal_photo(self):
        """View animal photo."""
        photo_path = self.animal_form_vars['photo_path'].get()
        if photo_path and os.path.exists(photo_path):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(photo_path)
                else:  # macOS and Linux
                    os.system(f'open "{photo_path}"' if os.name == 'posix' else f'xdg-open "{photo_path}"')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open photo: {e}")
        else:
            messagebox.showwarning("No Photo", "No photo selected or file not found.")

    def _save_animal(self):
        """Save animal to database."""
        try:
            name = self.animal_form_vars['name'].get().strip()
            individual_id = self.animal_form_vars['individual_id'].get().strip()
            animal_type = self.animal_form_vars['animal_type'].get()
            color = self.animal_form_vars['color'].get().strip()
            identification_method = self.animal_form_vars['identification_method'].get()
            photo_path = self.animal_form_vars['photo_path'].get().strip()
            selected_id = self.animal_form_vars['selected_id'].get()

            if not name:
                messagebox.showwarning("Validation Error", "Please enter a name.")
                return

            if not animal_type:
                messagebox.showwarning("Validation Error", "Please select an animal type.")
                return

            if not photo_path or not os.path.exists(photo_path):
                messagebox.showwarning("Validation Error", "Please select a valid photo.")
                return

            # Map animal type to COCO class ID
            animal_to_coco = {
                "Dog": 16, "Cat": 17, "Horse": 18, "Sheep": 19,
                "Cow": 20, "Elephant": 22, "Bear": 23, "Zebra": 24
            }
            coco_class_id = animal_to_coco.get(animal_type, 16)  # Default to dog

            # Create or update animal entry
            animal = WhitelistEntry(
                id=int(selected_id) if selected_id else None,
                name=name,
                entity_type='animal',
                individual_id=individual_id if individual_id else None,
                color=color if color else None,
                identification_method=identification_method,
                image_path=photo_path,
                coco_class_id=coco_class_id
            )

            if selected_id:
                self.db_manager.update_whitelist_entry(animal)
                messagebox.showinfo("Success", "Animal updated successfully!")
            else:
                self.db_manager.create_whitelist_entry(animal)
                messagebox.showinfo("Success", "Animal added successfully!")

            self._refresh_animals_list()
            self._reset_animal_form()

        except Exception as e:
            logger.error(f"Failed to save animal: {e}")
            messagebox.showerror("Error", f"Failed to save animal: {e}")

    def _test_animal_recognition(self):
        """Test animal recognition."""
        messagebox.showinfo("Test Recognition", "Animal recognition testing will be implemented.")

    def _delete_animal(self):
        """Delete selected animal."""
        selected_id = self.animal_form_vars['selected_id'].get()
        if not selected_id:
            messagebox.showwarning("Selection Error", "Please select an animal to delete.")
            return

        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this animal?"):
            try:
                self.db_manager.delete_whitelist_entry(int(selected_id))
                messagebox.showinfo("Success", "Animal deleted successfully!")
                self._refresh_animals_list()
                self._reset_animal_form()
            except Exception as e:
                logger.error(f"Failed to delete animal: {e}")
                messagebox.showerror("Error", f"Failed to delete animal: {e}")

    def _reset_animal_form(self):
        """Reset animal form."""
        for key, var in self.animal_form_vars.items():
            if key == 'identification_method':
                var.set("Hybrid")
            else:
                var.set("")

    # Bulk operations methods
    def _import_from_csv(self):
        """Import entities from CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select CSV file to import",
            filetypes=[("CSV files", "*.csv")]
        )

        if not file_path:
            return

        try:
            imported_count = 0
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    try:
                        # Validate required fields
                        if not row.get('name') or not row.get('entity_type'):
                            continue

                        # Create entity
                        entity = WhitelistEntry(
                            name=row['name'],
                            entity_type=row['entity_type'],
                            individual_id=row.get('individual_id'),
                            color=row.get('color'),
                            identification_method=row.get('identification_method', 'color'),
                            image_path=row.get('image_path', ''),
                            confidence_threshold=float(row.get('confidence_threshold', 0.6))
                        )

                        # Map animal type to COCO class ID if needed
                        if entity.entity_type == 'animal' and row.get('animal_type'):
                            animal_to_coco = {
                                "Dog": 16, "Cat": 17, "Horse": 18, "Sheep": 19,
                                "Cow": 20, "Elephant": 22, "Bear": 23, "Zebra": 24
                            }
                            entity.coco_class_id = animal_to_coco.get(row['animal_type'], 16)

                        self.db_manager.create_whitelist_entry(entity)
                        imported_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to import row {row}: {e}")
                        continue

            messagebox.showinfo("Import Complete", f"Successfully imported {imported_count} entities.")
            self._refresh_humans_list()
            self._refresh_animals_list()

        except Exception as e:
            logger.error(f"Failed to import CSV: {e}")
            messagebox.showerror("Import Error", f"Failed to import CSV: {e}")

    def _export_to_csv(self):
        """Export entities to CSV file."""
        file_path = filedialog.asksaveasfilename(
            title="Save CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not file_path:
            return

        try:
            # Get all entities
            humans = self.db_manager.get_whitelist_entries(entity_type='human')
            animals = self.db_manager.get_whitelist_entries(entity_type='animal')
            all_entities = humans + animals

            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'name', 'entity_type', 'individual_id', 'animal_type', 'color',
                    'identification_method', 'image_path', 'confidence_threshold'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for entity in all_entities:
                    # Map COCO class ID to animal type
                    animal_type = ""
                    if entity.coco_class_id:
                        coco_to_animal = {
                            16: "Dog", 17: "Cat", 18: "Horse", 19: "Sheep",
                            20: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra"
                        }
                        animal_type = coco_to_animal.get(entity.coco_class_id, "")

                    writer.writerow({
                        'name': entity.name,
                        'entity_type': entity.entity_type,
                        'individual_id': entity.individual_id or "",
                        'animal_type': animal_type,
                        'color': entity.color or "",
                        'identification_method': entity.identification_method,
                        'image_path': entity.image_path,
                        'confidence_threshold': entity.confidence_threshold
                    })

            messagebox.showinfo("Export Complete", f"Successfully exported {len(all_entities)} entities to CSV.")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            messagebox.showerror("Export Error", f"Failed to export CSV: {e}")

    def _backup_database(self):
        """Backup the database."""
        backup_path = filedialog.asksaveasfilename(
            title="Save database backup",
            defaultextension=".db",
            filetypes=[("Database files", "*.db"), ("All files", "*.*")]
        )

        if not backup_path:
            return

        try:
            # Copy the database file
            db_path = self.db_manager.db_path
            shutil.copy2(db_path, backup_path)
            messagebox.showinfo("Backup Complete", f"Database backed up to: {backup_path}")

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            messagebox.showerror("Backup Error", f"Failed to backup database: {e}")

    def _check_face_recognition_status(self):
        """Check the status of the face recognition system."""
        try:
            from core.face_recognition import FaceRecognitionSystem

            # Create a test face recognition system
            test_system = FaceRecognitionSystem()

            if hasattr(test_system, 'use_opencv_fallback') and test_system.use_opencv_fallback:
                if hasattr(test_system, 'opencv_face_system') and test_system.opencv_face_system:
                    status_text = "✅ OpenCV Face Recognition (Working)"
                    status_color = "green"
                else:
                    status_text = "⚠️ OpenCV Face Recognition (Not Available)"
                    status_color = "orange"
            else:
                status_text = "✅ Standard Face Recognition (Working)"
                status_color = "green"

        except Exception as e:
            status_text = f"❌ Face Recognition Error: {str(e)[:50]}..."
            status_color = "red"

        # Update status if the label exists
        if hasattr(self, 'face_status_label'):
            self.face_status_label.config(text=status_text, foreground=status_color)

    def _regenerate_encodings(self):
        """Regenerate face encodings for all humans."""
        if messagebox.askyesno("Confirm", "This will regenerate face encodings for all humans. Continue?"):
            try:
                messagebox.showinfo("Processing", "Face encoding regeneration started. This may take a while.")

                # Get all human entries
                humans = self.db_manager.get_whitelist_entries(entity_type="human")

                success_count = 0
                error_count = 0

                for human in humans:
                    try:
                        # Use the core face recognition system to compute encodings
                        from core.face_recognition import FaceRecognitionSystem
                        import pickle

                        # Create a temporary face recognition system
                        temp_face_system = FaceRecognitionSystem()

                        # Use the same method as the main system
                        temp_face_system._load_face_from_image(human.name, human.image_path)

                        # Extract the encodings that were just loaded
                        if temp_face_system.known_face_encodings:
                            person_encodings = []
                            for i, face_name in enumerate(temp_face_system.known_face_names):
                                if face_name == human.name:
                                    person_encodings.append(temp_face_system.known_face_encodings[i])

                            if person_encodings:
                                # Serialize and update the database entry
                                face_encodings_data = pickle.dumps(person_encodings)
                                human.face_encodings = face_encodings_data
                                self.db_manager.update_whitelist_entry(human)
                                success_count += 1
                                logger.info(f"Updated face encodings for {human.name}")
                            else:
                                error_count += 1
                                logger.warning(f"No face encodings computed for {human.name}")
                        else:
                            error_count += 1
                            logger.warning(f"No faces detected for {human.name}")

                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing {human.name}: {e}")

                # Show results
                if success_count > 0:
                    messagebox.showinfo("Complete",
                                      f"Face encoding regeneration completed!\n"
                                      f"Successfully processed: {success_count}\n"
                                      f"Errors: {error_count}")
                else:
                    messagebox.showwarning("No Success",
                                         f"No face encodings were successfully generated.\n"
                                         f"Errors: {error_count}\n"
                                         "Please check the image files and face_recognition library installation.")

                # Refresh the display
                self._refresh_humans_list()

            except Exception as e:
                logger.error(f"Failed to regenerate encodings: {e}")
                messagebox.showerror("Error", f"Failed to regenerate encodings: {e}")

    def _clean_invalid_entries(self):
        """Clean invalid entries from database."""
        if messagebox.askyesno("Confirm", "This will remove entries with missing or invalid image files. Continue?"):
            try:
                cleaned_count = 0
                all_entities = (self.db_manager.get_whitelist_entries(entity_type='human') +
                               self.db_manager.get_whitelist_entries(entity_type='animal'))

                for entity in all_entities:
                    if not entity.image_path or not os.path.exists(entity.image_path):
                        self.db_manager.delete_whitelist_entry(entity.id)
                        cleaned_count += 1

                messagebox.showinfo("Cleanup Complete", f"Removed {cleaned_count} invalid entries.")
                self._refresh_humans_list()
                self._refresh_animals_list()

            except Exception as e:
                logger.error(f"Failed to clean entries: {e}")
                messagebox.showerror("Error", f"Failed to clean entries: {e}")

    def _generate_report(self):
        """Generate entity management report."""
        try:
            humans = self.db_manager.get_whitelist_entries(entity_type='human')
            animals = self.db_manager.get_whitelist_entries(entity_type='animal')

            report = f"""Entity Management Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Total Humans: {len(humans)}
- Total Animals: {len(animals)}
- Total Entities: {len(humans) + len(animals)}

Humans:
"""
            for human in humans:
                report += f"  - {human.name} (Confidence: {human.confidence_threshold:.1%})\n"

            report += "\nAnimals:\n"
            for animal in animals:
                animal_type = "Unknown"
                if animal.coco_class_id:
                    coco_to_animal = {
                        16: "Dog", 17: "Cat", 18: "Horse", 19: "Sheep",
                        20: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra"
                    }
                    animal_type = coco_to_animal.get(animal.coco_class_id, "Unknown")

                report += f"  - {animal.name} ({animal_type}, {animal.identification_method})\n"

            # Save report
            report_path = filedialog.asksaveasfilename(
                title="Save report",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")]
            )

            if report_path:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Report Generated", f"Report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            messagebox.showerror("Error", f"Failed to generate report: {e}")

    # Camera capture functionality
    def _open_camera_capture_window(self):
        """Open camera capture window."""
        if self.capture_window is not None:
            self.capture_window.destroy()

        self.capture_window = tk.Toplevel()
        self.capture_window.title("📷 Camera Capture")
        self.capture_window.geometry("800x700")
        self.capture_window.resizable(True, True)

        # Make window modal
        self.capture_window.transient()
        self.capture_window.grab_set()

        # Main frame
        main_frame = ttk.Frame(self.capture_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_text = f"Capture Photo for {self.capture_type.title()}"
        ttk.Label(main_frame, text=title_text, font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Video display label
        self.capture_label = ttk.Label(video_frame, text="Initializing camera...",
                                      background="black", foreground="white")
        self.capture_label.pack(fill=tk.BOTH, expand=True)

        # Control buttons frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(controls_frame, text="📷 Capture Photo",
                  command=self._capture_photo, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="🔄 Retry Camera",
                  command=self._retry_camera).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="❌ Cancel",
                  command=self._close_capture_window).pack(side=tk.RIGHT)

        # Status label
        self.capture_status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.capture_status_label.pack(pady=(0, 10))

        # Handle window close
        self.capture_window.protocol("WM_DELETE_WINDOW", self._close_capture_window)

        # Start camera feed
        self._start_camera_feed()

    def _start_camera_feed(self):
        """Start the camera feed."""
        try:
            # Try to setup cameras if not already done
            if not self.camera_manager.cameras:
                self.camera_manager.setup_cameras()

            if not self.camera_manager.cameras:
                self.capture_status_label.config(text="No camera available", foreground="red")
                self.capture_label.config(text="No camera detected\n\nPlease connect a camera and click 'Retry Camera'")
                return

            self.capture_active = True
            self.capture_status_label.config(text="Camera active", foreground="green")
            self._update_camera_feed()

        except Exception as e:
            logger.error(f"Failed to start camera feed: {e}")
            self.capture_status_label.config(text=f"Camera error: {e}", foreground="red")
            self.capture_label.config(text=f"Camera Error\n\n{e}")

    def _update_camera_feed(self):
        """Update camera feed in capture window."""
        if not self.capture_active or self.capture_window is None:
            return

        try:
            frame = self.camera_manager.capture_frame()
            if frame is not None:
                # Resize frame to fit display
                display_height = 480
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_width = int(display_height * aspect_ratio)

                frame_resized = cv2.resize(frame, (display_width, display_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Convert to PhotoImage
                from PIL import Image, ImageTk
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)

                # Update label
                self.capture_label.config(image=photo, text="")
                self.capture_label.image = photo  # Keep a reference

                self.capture_status_label.config(text="Camera feed active", foreground="green")
            else:
                self.capture_status_label.config(text="No frame received", foreground="orange")

            # Schedule next update
            if self.capture_active and self.capture_window:
                self.capture_window.after(33, self._update_camera_feed)  # ~30 FPS

        except Exception as e:
            logger.error(f"Camera feed update error: {e}")
            self.capture_status_label.config(text=f"Feed error: {e}", foreground="red")
            if self.capture_active and self.capture_window:
                self.capture_window.after(1000, self._update_camera_feed)  # Retry in 1 second

    def _capture_photo(self):
        """Capture and save photo."""
        try:
            frame = self.camera_manager.capture_frame()
            if frame is None:
                messagebox.showerror("Capture Error", "Failed to capture frame from camera")
                return

            # Create photos directory if it doesn't exist
            photos_dir = "data/photos"
            os.makedirs(photos_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.capture_type}_{timestamp}.jpg"
            filepath = os.path.join(photos_dir, filename)

            # Save image
            success = cv2.imwrite(filepath, frame)
            if success:
                # Update the appropriate form variable
                if self.capture_type == 'human':
                    self.human_form_vars['photo_path'].set(filepath)
                else:
                    self.animal_form_vars['photo_path'].set(filepath)

                messagebox.showinfo("Success", f"Photo captured and saved to:\n{filepath}")
                self._close_capture_window()
            else:
                messagebox.showerror("Save Error", "Failed to save captured photo")

        except Exception as e:
            logger.error(f"Photo capture error: {e}")
            messagebox.showerror("Capture Error", f"Failed to capture photo: {e}")

    def _retry_camera(self):
        """Retry camera connection."""
        self.capture_active = False
        self.camera_manager.release_all_cameras()
        self.capture_status_label.config(text="Retrying camera...", foreground="orange")
        self.capture_label.config(text="Reconnecting to camera...")

        # Wait a moment then restart
        self.capture_window.after(1000, self._start_camera_feed)

    def _close_capture_window(self):
        """Close camera capture window."""
        self.capture_active = False
        if self.capture_window:
            self.capture_window.destroy()
            self.capture_window = None
        self.capture_label = None
        
        ttk.Button(batch_frame, text="🔄 Regenerate Face Encodings").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(batch_frame, text="🧹 Clean Invalid Entries").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(batch_frame, text="📊 Generate Report").pack(side=tk.LEFT)
