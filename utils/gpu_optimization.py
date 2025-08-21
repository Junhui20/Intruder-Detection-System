"""
GPU Optimization and Hardware Detection

This module provides GPU optimization utilities and hardware detection.
"""

import logging
import platform
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    GPU optimization and hardware detection utilities.
    
    Features:
    - CUDA availability detection
    - GPU memory management
    - Performance optimization recommendations
    - Hardware capability assessment
    """
    
    def __init__(self):
        """Initialize GPU optimizer."""
        self.cuda_available = False
        self.gpu_count = 0
        self.gpu_info = []
        self.optimization_applied = False
        
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available GPU hardware."""
        try:
            # Try to import PyTorch for CUDA detection
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self.gpu_count = torch.cuda.device_count()
                
                for i in range(self.gpu_count):
                    gpu_props = torch.cuda.get_device_properties(i)
                    self.gpu_info.append({
                        'name': gpu_props.name,
                        'memory_total': gpu_props.total_memory / 1024**3,  # GB
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                        'multiprocessor_count': gpu_props.multi_processor_count
                    })
                
                logger.info(f"CUDA available: {self.gpu_count} GPU(s) detected")
                for i, info in enumerate(self.gpu_info):
                    logger.info(f"GPU {i}: {info['name']} ({info['memory_total']:.1f}GB)")
            else:
                logger.info("CUDA not available, using CPU")
                
        except ImportError:
            logger.warning("PyTorch not available for GPU detection")
        except Exception as e:
            logger.error(f"Error detecting GPU hardware: {e}")
    
    def get_hardware_info(self) -> Dict:
        """
        Get comprehensive hardware information.
        
        Returns:
            Hardware information dictionary
        """
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'cuda_available': self.cuda_available,
            'gpu_count': self.gpu_count,
            'gpu_info': self.gpu_info
        }
        
        # Add CPU information
        try:
            import psutil
            info['cpu_count'] = psutil.cpu_count()
            info['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else None
            info['memory_total'] = psutil.virtual_memory().total / 1024**3  # GB
        except ImportError:
            logger.warning("psutil not available for CPU info")
        
        return info
    
    def optimize_for_detection(self, model_type: str = "yolo11n") -> Dict[str, any]:
        """
        Optimize settings for detection models.
        
        Args:
            model_type: Type of model to optimize for
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'device': 'cpu',
            'batch_size': 1,
            'half_precision': False,
            'tensorrt': False,
            'quantization': False,
            'memory_optimization': True
        }
        
        if self.cuda_available and self.gpu_info:
            gpu = self.gpu_info[0]  # Use first GPU
            memory_gb = gpu['memory_total']
            
            # Set device to GPU
            recommendations['device'] = 'cuda:0'
            
            # Optimize based on GPU memory
            if memory_gb >= 8:
                # High-end GPU
                recommendations['batch_size'] = 4
                recommendations['half_precision'] = True
                recommendations['tensorrt'] = True
                logger.info("Optimized for high-end GPU")
                
            elif memory_gb >= 4:
                # Mid-range GPU
                recommendations['batch_size'] = 2
                recommendations['half_precision'] = True
                recommendations['quantization'] = True
                logger.info("Optimized for mid-range GPU")
                
            else:
                # Low-end GPU
                recommendations['batch_size'] = 1
                recommendations['quantization'] = True
                logger.info("Optimized for low-end GPU")
        
        else:
            # CPU optimization
            recommendations['batch_size'] = 1
            recommendations['memory_optimization'] = True
            logger.info("Optimized for CPU processing")
        
        self.optimization_applied = True
        return recommendations
    
    def setup_yolo_device(self) -> str:
        """
        Set up optimal device for YOLO models.
        
        Returns:
            Device string for YOLO
        """
        if self.cuda_available:
            return 'cuda:0'
        else:
            return 'cpu'
    
    def setup_torch_optimizations(self):
        """Set up PyTorch optimizations if available."""
        try:
            import torch
            
            if self.cuda_available:
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Set memory allocation strategy
                torch.cuda.empty_cache()
                
                logger.info("PyTorch CUDA optimizations enabled")
            
            # CPU optimizations
            torch.set_num_threads(4)  # Limit CPU threads for better performance
            
        except ImportError:
            logger.warning("PyTorch not available for optimizations")
        except Exception as e:
            logger.error(f"Error setting up PyTorch optimizations: {e}")
    
    def monitor_gpu_usage(self) -> Optional[Dict]:
        """
        Monitor GPU usage and memory.
        
        Returns:
            GPU usage statistics or None if not available
        """
        if not self.cuda_available:
            return None
        
        try:
            import torch
            
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                
                # Calculate utilization
                utilization = (memory_allocated / memory_total) * 100
                
                return {
                    'device': device,
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'memory_total_gb': memory_total,
                    'utilization_percent': utilization,
                    'temperature': self._get_gpu_temperature()
                }
        
        except Exception as e:
            logger.error(f"Error monitoring GPU usage: {e}")
        
        return None
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].temperature
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get GPU temperature: {e}")
        
        return None
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        if self.cuda_available:
            try:
                import torch
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared")
            except Exception as e:
                logger.error(f"Error cleaning GPU memory: {e}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get optimization recommendations based on hardware.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.cuda_available:
            recommendations.extend([
                "Consider upgrading to a CUDA-compatible GPU for better performance",
                "Use smaller model variants (e.g., YOLOv11n instead of YOLOv11x)",
                "Reduce input resolution to improve processing speed",
                "Enable multi-threading for CPU processing"
            ])
        
        elif self.gpu_info:
            gpu = self.gpu_info[0]
            memory_gb = gpu['memory_total']
            
            if memory_gb < 4:
                recommendations.extend([
                    "GPU memory is limited - consider enabling quantization",
                    "Use batch size of 1 to reduce memory usage",
                    "Enable memory optimization features"
                ])
            
            elif memory_gb >= 8:
                recommendations.extend([
                    "High-end GPU detected - enable TensorRT for maximum performance",
                    "Consider using half-precision (FP16) for faster inference",
                    "Increase batch size for better throughput"
                ])
        
        # General recommendations
        recommendations.extend([
            "Monitor GPU temperature to prevent thermal throttling",
            "Regularly clear GPU memory cache to prevent memory leaks",
            "Use appropriate model size for your use case"
        ])
        
        return recommendations
    
    def benchmark_performance(self, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark GPU/CPU performance.
        
        Args:
            iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics
        """
        results = {
            'device': 'cpu',
            'avg_inference_time': 0.0,
            'throughput_fps': 0.0,
            'memory_usage': 0.0
        }
        
        try:
            import torch
            import time
            
            device = 'cuda:0' if self.cuda_available else 'cpu'
            results['device'] = device
            
            # Create dummy tensor for benchmarking
            dummy_input = torch.randn(1, 3, 640, 640).to(device)
            
            # Warm up
            for _ in range(10):
                _ = torch.nn.functional.relu(dummy_input)
            
            # Benchmark
            start_time = time.time()
            
            for _ in range(iterations):
                _ = torch.nn.functional.relu(dummy_input)
                
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
            
            total_time = time.time() - start_time
            
            results['avg_inference_time'] = (total_time / iterations) * 1000  # ms
            results['throughput_fps'] = iterations / total_time
            
            # Memory usage
            if device.startswith('cuda'):
                results['memory_usage'] = torch.cuda.memory_allocated() / 1024**2  # MB
            
            logger.info(f"Benchmark results: {results['throughput_fps']:.1f} FPS on {device}")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
        
        return results
