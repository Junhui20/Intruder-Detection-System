"""
Model Optimization Module for Intruder Detection System

This module provides model quantization, TensorRT acceleration, and optimization
utilities for YOLO models to achieve maximum performance.

Features:
- INT8/FP16 quantization
- TensorRT acceleration for NVIDIA GPUs
- Automatic model format detection and fallback
- Performance benchmarking
- Model conversion utilities
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

# Core imports
from ultralytics import YOLO
import torch

# Optional imports for advanced features
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Advanced model optimization system for YOLO models.

    Features:
    - Automatic quantization (INT8/FP16)
    - TensorRT acceleration
    - Performance benchmarking
    - Model format management
    """

    def __init__(self, base_model_path: str = "yolo11n.pt", models_dir: str = "models"):
        """
        Initialize the model optimizer.

        Args:
            base_model_path: Path to the base YOLO model
            models_dir: Directory to store optimized models
        """
        self.base_model_path = base_model_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Available model formats and their priorities (higher = better performance)
        self.model_formats = {
            'tensorrt': {'priority': 5, 'extension': '.engine', 'available': TENSORRT_AVAILABLE},
            'onnx_fp16': {'priority': 4, 'extension': '_fp16.onnx', 'available': ONNX_AVAILABLE},
            'onnx_int8': {'priority': 3, 'extension': '_int8.onnx', 'available': ONNX_AVAILABLE},
            'torchscript': {'priority': 2, 'extension': '.torchscript', 'available': True},
            'pytorch': {'priority': 1, 'extension': '.pt', 'available': True}
        }

        # Performance tracking
        self.benchmark_results = {}
        self.current_model_info = {}

        # GPU detection
        self.gpu_available = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else None
        self.tensorrt_compatible = self._check_tensorrt_compatibility()

        logger.info(f"Model Optimizer initialized")
        logger.info(f"GPU Available: {self.gpu_available}")
        if self.gpu_available:
            logger.info(f"GPU: {self.gpu_name}")
        logger.info(f"TensorRT Compatible: {self.tensorrt_compatible}")

    def _check_tensorrt_compatibility(self) -> bool:
        """Check if system supports TensorRT acceleration."""
        if not TENSORRT_AVAILABLE or not self.gpu_available:
            return False

        try:
            # Check if GPU is NVIDIA and supports TensorRT
            if "nvidia" in self.gpu_name.lower():
                # Check TensorRT version
                logger.info(f"TensorRT version: {trt.__version__}")
                return True
        except Exception as e:
            logger.warning(f"TensorRT compatibility check failed: {e}")

        return False

    def create_quantized_models(self, calibration_images: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create quantized versions of the base model.

        Args:
            calibration_images: List of image paths for INT8 calibration

        Returns:
            Dictionary mapping format names to model paths
        """
        created_models = {}
        base_model = YOLO(self.base_model_path)

        logger.info("Creating quantized models...")

        # Create FP16 ONNX model
        if ONNX_AVAILABLE:
            try:
                fp16_path = self.models_dir / f"yolo11n_fp16.onnx"
                logger.info("Creating FP16 ONNX model...")
                base_model.export(format='onnx', half=True, dynamic=False, simplify=True)

                # Move to models directory
                default_onnx = Path(self.base_model_path).with_suffix('.onnx')
                if default_onnx.exists():
                    default_onnx.rename(fp16_path)
                    created_models['onnx_fp16'] = str(fp16_path)
                    logger.info(f"✅ FP16 ONNX model created: {fp16_path}")

            except Exception as e:
                logger.error(f"Failed to create FP16 ONNX model: {e}")

        # Create INT8 ONNX model
        if ONNX_AVAILABLE:
            try:
                int8_path = self.models_dir / f"yolo11n_int8.onnx"
                logger.info("Creating INT8 ONNX model...")
                base_model.export(format='onnx', int8=True, dynamic=False, simplify=True)

                # Move to models directory
                default_onnx = Path(self.base_model_path).with_suffix('.onnx')
                if default_onnx.exists():
                    default_onnx.rename(int8_path)
                    created_models['onnx_int8'] = str(int8_path)
                    logger.info(f"✅ INT8 ONNX model created: {int8_path}")

            except Exception as e:
                logger.error(f"Failed to create INT8 ONNX model: {e}")

        # Create TorchScript model
        try:
            torchscript_path = self.models_dir / f"yolo11n.torchscript"
            logger.info("Creating TorchScript model...")
            base_model.export(format='torchscript')

            # Move to models directory
            default_ts = Path(self.base_model_path).with_suffix('.torchscript')
            if default_ts.exists():
                default_ts.rename(torchscript_path)
                created_models['torchscript'] = str(torchscript_path)
                logger.info(f"✅ TorchScript model created: {torchscript_path}")

        except Exception as e:
            logger.error(f"Failed to create TorchScript model: {e}")

        # Create TensorRT engine (if compatible)
        if self.tensorrt_compatible:
            try:
                engine_path = self.models_dir / f"yolo11n.engine"
                logger.info("Creating TensorRT engine...")
                base_model.export(format='engine', half=True, dynamic=False, workspace=4)

                # Move to models directory
                default_engine = Path(self.base_model_path).with_suffix('.engine')
                if default_engine.exists():
                    default_engine.rename(engine_path)
                    created_models['tensorrt'] = str(engine_path)
                    logger.info(f"✅ TensorRT engine created: {engine_path}")

            except Exception as e:
                logger.error(f"Failed to create TensorRT engine: {e}")

        return created_models

    def benchmark_models(self, test_image_path: str, iterations: int = 100) -> Dict[str, Dict]:
        """
        Benchmark all available model formats.

        Args:
            test_image_path: Path to test image
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with benchmark results for each model format
        """
        results = {}

        # Find available models
        available_models = self._find_available_models()

        logger.info(f"Benchmarking {len(available_models)} model formats...")

        for format_name, model_path in available_models.items():
            try:
                logger.info(f"Benchmarking {format_name}...")

                # Load model
                model = YOLO(model_path)

                # Load test image
                import cv2
                test_image = cv2.imread(test_image_path)

                # Warmup
                for _ in range(5):
                    _ = model(test_image, verbose=False)

                # Benchmark
                times = []
                for _ in range(iterations):
                    start_time = time.time()
                    _ = model(test_image, verbose=False)
                    times.append(time.time() - start_time)

                # Calculate statistics
                avg_time = np.mean(times)
                fps = 1.0 / avg_time
                std_time = np.std(times)

                results[format_name] = {
                    'avg_inference_time': avg_time,
                    'fps': fps,
                    'std_time': std_time,
                    'model_path': model_path,
                    'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
                }

                logger.info(f"✅ {format_name}: {fps:.1f} FPS ({avg_time*1000:.1f}ms)")

            except Exception as e:
                logger.error(f"Failed to benchmark {format_name}: {e}")
                results[format_name] = {'error': str(e)}

        self.benchmark_results = results
        return results

    def get_optimal_model(self) -> Tuple[str, str]:
        """
        Get the optimal model format based on availability and performance.

        Returns:
            Tuple of (format_name, model_path)
        """
        available_models = self._find_available_models()

        if not available_models:
            # Fallback to base model
            return 'pytorch', self.base_model_path

        # Sort by priority (highest first)
        sorted_formats = sorted(
            available_models.items(),
            key=lambda x: self.model_formats[x[0]]['priority'],
            reverse=True
        )

        # Return the highest priority available model
        format_name, model_path = sorted_formats[0]
        logger.info(f"Selected optimal model: {format_name} ({model_path})")

        return format_name, model_path

    def _find_available_models(self) -> Dict[str, str]:
        """Find all available optimized models."""
        available = {}

        for format_name, format_info in self.model_formats.items():
            if not format_info['available']:
                continue

            # Check for model file
            if format_name == 'pytorch':
                model_path = self.base_model_path
            else:
                model_path = self.models_dir / f"yolo11n{format_info['extension']}"

            if Path(model_path).exists():
                available[format_name] = str(model_path)

        return available

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization report."""
        available_models = self._find_available_models()

        report = {
            'system_info': {
                'gpu_available': self.gpu_available,
                'gpu_name': self.gpu_name,
                'tensorrt_available': TENSORRT_AVAILABLE,
                'tensorrt_compatible': self.tensorrt_compatible,
                'onnx_available': ONNX_AVAILABLE
            },
            'available_models': available_models,
            'benchmark_results': self.benchmark_results,
            'optimal_model': self.get_optimal_model(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if not self.gpu_available:
            recommendations.append("Consider upgrading to a system with GPU for better performance")

        if self.gpu_available and not self.tensorrt_compatible:
            recommendations.append("GPU detected but TensorRT not available - install TensorRT for maximum performance")

        if not ONNX_AVAILABLE:
            recommendations.append("Install ONNX runtime for additional optimization options: pip install onnxruntime-gpu")

        available_models = self._find_available_models()
        if len(available_models) == 1:
            recommendations.append("Run create_quantized_models() to generate optimized model variants")

        return recommendations


class OptimizedDetectionEngine:
    """
    Enhanced detection engine with automatic model optimization.

    This class extends the base detection engine with automatic model
    selection and optimization capabilities.
    """

    def __init__(self, base_model_path: str = "yolo11n.pt", confidence: float = 0.5):
        """
        Initialize the optimized detection engine.

        Args:
            base_model_path: Path to the base YOLO model
            confidence: Detection confidence threshold
        """
        self.base_model_path = base_model_path
        self.confidence = confidence

        # Initialize optimizer
        self.optimizer = ModelOptimizer(base_model_path)

        # Get optimal model
        self.model_format, self.model_path = self.optimizer.get_optimal_model()

        # Load the optimal model
        self.model = YOLO(self.model_path)

        # Performance tracking
        self.detection_times = []
        self.frame_count = 0

        logger.info(f"Optimized Detection Engine initialized with {self.model_format} model")
        logger.info(f"Model path: {self.model_path}")

    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect objects using the optimized model.

        Args:
            frame: Input image frame

        Returns:
            Dictionary containing detection results with performance metrics
        """
        start_time = time.time()

        try:
            # Run detection
            results = self.model(frame, conf=self.confidence, verbose=False)

            # Process results (same as original detection engine)
            detections = {
                'humans': [],
                'animals': [],
                'frame_info': {
                    'timestamp': time.time(),
                    'processing_time': 0,
                    'total_detections': 0,
                    'model_format': self.model_format,
                    'model_path': self.model_path
                }
            }

            # Extract detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        detection_data = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls
                        }

                        if cls == 0:  # person
                            detections['humans'].append(detection_data)
                        elif cls in [15, 16, 17, 18, 19, 20, 21, 22]:  # animals
                            detections['animals'].append(detection_data)

            # Update performance metrics
            processing_time = time.time() - start_time
            detections['frame_info']['processing_time'] = processing_time
            detections['frame_info']['total_detections'] = len(detections['humans']) + len(detections['animals'])

            # Track performance
            self.detection_times.append(processing_time)
            self.frame_count += 1

            # Keep only last 100 times
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'humans': [],
                'animals': [],
                'frame_info': {
                    'timestamp': time.time(),
                    'processing_time': 0,
                    'total_detections': 0,
                    'error': str(e)
                }
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.detection_times:
            return {}

        avg_time = np.mean(self.detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_detection_time': avg_time,
            'fps': fps,
            'frame_count': self.frame_count,
            'model_format': self.model_format,
            'model_path': self.model_path
        }