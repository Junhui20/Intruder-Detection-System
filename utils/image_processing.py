"""
Image Processing Utilities

This module provides image processing functions for the detection system.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processing utilities for detection and analysis.
    
    Features:
    - Color space conversions
    - Image enhancement
    - Geometric transformations
    - Color detection for animal identification
    """
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame: Input frame
            target_size: (width, height) target size
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized frame
        """
        if maintain_aspect:
            h, w = frame.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize and pad if necessary
            resized = cv2.resize(frame, (new_w, new_h))
            
            if new_w != target_w or new_h != target_h:
                # Create padded frame
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                return padded
            
            return resized
        else:
            return cv2.resize(frame, target_size)
    
    @staticmethod
    def enhance_image(frame: np.ndarray, brightness: float = 0, 
                     contrast: float = 1.0, gamma: float = 1.0) -> np.ndarray:
        """
        Enhance image with brightness, contrast, and gamma correction.
        
        Args:
            frame: Input frame
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 3.0)
            gamma: Gamma correction (0.5 to 2.0)
            
        Returns:
            Enhanced frame
        """
        # Apply brightness and contrast
        enhanced = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        # Apply gamma correction
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    @staticmethod
    def detect_dominant_color(image: np.ndarray, k: int = 5) -> Tuple[str, np.ndarray]:
        """
        Detect dominant color in image using K-means clustering.
        
        Args:
            image: Input image
            k: Number of clusters
            
        Returns:
            Tuple of (color_name, dominant_color_bgr)
        """
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the most frequent cluster
        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique[np.argmax(counts)]
        dominant_color = centers[dominant_cluster].astype(int)
        
        # Convert BGR to color name
        color_name = ImageProcessor._bgr_to_color_name(dominant_color)
        
        return color_name, dominant_color
    
    @staticmethod
    def _bgr_to_color_name(bgr_color: np.ndarray) -> str:
        """
        Convert BGR color to color name.
        
        Args:
            bgr_color: BGR color array [B, G, R]
            
        Returns:
            Color name string
        """
        b, g, r = bgr_color
        
        # Convert to HSV for better color classification
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv
        
        # Color classification based on HSV
        if v < 50:
            return "black"
        elif v > 200 and s < 50:
            return "white"
        elif s < 50:
            return "gray"
        elif 0 <= h <= 10 or 170 <= h <= 180:
            return "red"
        elif 10 < h <= 25:
            return "orange"
        elif 25 < h <= 35:
            return "yellow"
        elif 35 < h <= 85:
            return "green"
        elif 85 < h <= 125:
            return "blue"
        elif 125 < h <= 150:
            return "purple"
        elif 150 < h <= 170:
            return "pink"
        else:
            return "unknown"
    
    @staticmethod
    def extract_color_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract color features for animal identification.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of color features
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        features = {}
        
        # HSV statistics
        features['hue_mean'] = np.mean(hsv[:, :, 0])
        features['hue_std'] = np.std(hsv[:, :, 0])
        features['saturation_mean'] = np.mean(hsv[:, :, 1])
        features['saturation_std'] = np.std(hsv[:, :, 1])
        features['value_mean'] = np.mean(hsv[:, :, 2])
        features['value_std'] = np.std(hsv[:, :, 2])
        
        # LAB statistics
        features['l_mean'] = np.mean(lab[:, :, 0])
        features['a_mean'] = np.mean(lab[:, :, 1])
        features['b_mean'] = np.mean(lab[:, :, 2])
        
        # Color histogram features
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        features['hue_peak'] = np.argmax(hist_h)
        
        return features
    
    @staticmethod
    def apply_color_mask(image: np.ndarray, color_range: List[List[int]]) -> np.ndarray:
        """
        Apply color mask to image.
        
        Args:
            image: Input image
            color_range: [[lower_bound], [upper_bound]] in HSV
            
        Returns:
            Binary mask
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(color_range[0])
        upper = np.array(color_range[1])
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    @staticmethod
    def draw_detection_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                          label: str, confidence: float, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw detection bounding box with label.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            label: Detection label
            confidence: Detection confidence
            color: Box color (B, G, R)
            
        Returns:
            Frame with drawn box
        """
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Use the label as-is (it already contains confidence)
        label_text = label
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Ensure label doesn't go outside frame boundaries
        frame_height, frame_width = frame.shape[:2]
        label_y = max(text_height + baseline + 5, y1)  # Don't go above frame
        label_x = min(x1, frame_width - text_width)    # Don't go beyond right edge

        # Draw label background
        bg_y1 = label_y - text_height - baseline - 5
        bg_y2 = label_y
        bg_x1 = label_x
        bg_x2 = min(label_x + text_width, frame_width)

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Draw label text
        cv2.putText(frame, label_text, (label_x, label_y - baseline - 2),
                   font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    @staticmethod
    def create_detection_overlay(frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Create overlay with all detection information.
        
        Args:
            frame: Input frame
            detections: Detection results dictionary
            
        Returns:
            Frame with overlay
        """
        overlay_frame = frame.copy()
        
        # Draw human detections
        for human in detections.get('humans', []):
            bbox = human.get('bbox', (0, 0, 0, 0))
            identity = human.get('identity', 'Unknown')
            face_confidence = human.get('face_confidence', 0)
            yolo_confidence = human.get('confidence', 0)

            # Choose appropriate confidence to display
            if identity == 'Unknown' or identity == 'No Face Detected':
                # For unknown humans, show YOLO detection confidence
                display_confidence = yolo_confidence
                label = f"Unknown Human ({display_confidence * 100:.1f}%)"
                color = (0, 0, 255)  # Red for unknown
            else:
                # For known humans, show face recognition confidence
                display_confidence = face_confidence
                label = f"{identity} ({display_confidence * 100:.1f}%)"
                color = (0, 255, 0)  # Green for known

            overlay_frame = ImageProcessor.draw_detection_box(
                overlay_frame, bbox, label, display_confidence, color
            )
        
        # Draw animal detections
        for animal in detections.get('animals', []):
            bbox = animal.get('bbox', (0, 0, 0, 0))
            pet_identity = animal.get('pet_identity', 'Unknown Animal')
            confidence = animal.get('identification_confidence', animal.get('confidence', 0))
            detected_color = animal.get('detected_color', '')

            # Convert confidence to percentage if it's in decimal format
            if confidence <= 1.0:
                confidence_pct = confidence * 100
            else:
                confidence_pct = confidence

            # Format label according to user requirements
            if 'Unknown' in pet_identity or pet_identity == 'Unknown Animal':
                if detected_color and detected_color != 'unknown':
                    label = f"Unknown Animal ({detected_color}) ({confidence_pct:.1f}%)"
                else:
                    label = f"Unknown Animal ({confidence_pct:.1f}%)"
                color = (0, 255, 255)  # Cyan for unknown animal
            else:
                if detected_color and detected_color != 'unknown':
                    label = f"{pet_identity} ({detected_color}) ({confidence_pct:.1f}%)"
                else:
                    label = f"{pet_identity} ({confidence_pct:.1f}%)"
                color = (255, 0, 0)  # Blue for known pet

            overlay_frame = ImageProcessor.draw_detection_box(
                overlay_frame, bbox, label, confidence, color
            )
        
        # Add frame info
        frame_info = detections.get('frame_info', {})
        fps = 1.0 / frame_info.get('processing_time', 1.0) if frame_info.get('processing_time', 0) > 0 else 0
        
        info_text = f"FPS: {fps:.1f} | Detections: {frame_info.get('total_detections', 0)}"
        cv2.putText(overlay_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay_frame
    
    @staticmethod
    def save_detection_image(frame: np.ndarray, detections: Dict, 
                           output_path: str, include_overlay: bool = True) -> bool:
        """
        Save detection image with optional overlay.
        
        Args:
            frame: Input frame
            detections: Detection results
            output_path: Output file path
            include_overlay: Whether to include detection overlay
            
        Returns:
            True if successful
        """
        try:
            if include_overlay:
                save_frame = ImageProcessor.create_detection_overlay(frame, detections)
            else:
                save_frame = frame
            
            success = cv2.imwrite(output_path, save_frame)
            if success:
                logger.info(f"Detection image saved to {output_path}")
            else:
                logger.error(f"Failed to save image to {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving detection image: {e}")
            return False
