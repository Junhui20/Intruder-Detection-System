#!/usr/bin/env python3
"""
Model Optimization Setup Script

This script sets up model quantization and optimization for the intruder detection system.
It creates optimized model variants and benchmarks their performance.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.model_optimization import ModelOptimizer, OptimizedDetectionEngine
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_system_requirements():
    """Check system requirements for optimization."""
    print("🔍 Checking System Requirements...")
    print("=" * 50)

    # Check PyTorch and CUDA
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check ONNX
    try:
        import onnx
        import onnxruntime as ort
        print(f"ONNX version: {onnx.__version__}")
        print(f"ONNX Runtime version: {ort.__version__}")

        # Check ONNX providers
        providers = ort.get_available_providers()
        print(f"ONNX providers: {providers}")
    except ImportError:
        print("❌ ONNX not available - install with: pip install onnx onnxruntime-gpu")

    # Check TensorRT
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not available - install for maximum GPU performance")

    print()


def create_test_image(output_path: str = "test_image.jpg"):
    """Create a test image for benchmarking."""
    # Create a test image with some objects
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Add some simple shapes to make it more realistic
    cv2.rectangle(test_image, (100, 100), (200, 300), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(test_image, (400, 200), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(test_image, (300, 400), (500, 500), (0, 0, 255), -1)  # Red rectangle

    cv2.imwrite(output_path, test_image)
    return output_path


def setup_optimization(base_model: str = "yolo11n.pt", benchmark: bool = True):
    """Set up model optimization."""
    print("🚀 Setting Up Model Optimization...")
    print("=" * 50)

    # Initialize optimizer
    optimizer = ModelOptimizer(base_model)

    # Get system report
    report = optimizer.get_optimization_report()

    print("📊 System Information:")
    for key, value in report['system_info'].items():
        print(f"  {key}: {value}")

    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")

    print("\n🔧 Creating Optimized Models...")

    # Create quantized models
    created_models = optimizer.create_quantized_models()

    if created_models:
        print("\n✅ Created Models:")
        for format_name, model_path in created_models.items():
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  • {format_name}: {model_path} ({file_size:.1f} MB)")
    else:
        print("\n❌ No optimized models were created")

    # Benchmark models if requested
    if benchmark and created_models:
        print("\n📈 Benchmarking Models...")

        # Create test image
        test_image_path = create_test_image()

        try:
            # Run benchmarks
            benchmark_results = optimizer.benchmark_models(test_image_path, iterations=50)

            print("\n📊 Benchmark Results:")
            print("-" * 60)
            print(f"{'Format':<15} {'FPS':<10} {'Time (ms)':<12} {'Size (MB)':<10}")
            print("-" * 60)

            # Sort by FPS (descending)
            sorted_results = sorted(
                benchmark_results.items(),
                key=lambda x: x[1].get('fps', 0),
                reverse=True
            )

            for format_name, results in sorted_results:
                if 'error' not in results:
                    fps = results['fps']
                    time_ms = results['avg_inference_time'] * 1000
                    size_mb = results['model_size_mb']
                    print(f"{format_name:<15} {fps:<10.1f} {time_ms:<12.1f} {size_mb:<10.1f}")
                else:
                    print(f"{format_name:<15} ERROR: {results['error']}")

            # Show optimal model
            optimal_format, optimal_path = optimizer.get_optimal_model()
            print(f"\n🎯 Optimal Model: {optimal_format} ({optimal_path})")

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
        finally:
            # Clean up test image
            if os.path.exists(test_image_path):
                os.remove(test_image_path)

    return optimizer


def test_optimized_detection():
    """Test the optimized detection engine."""
    print("\n🧪 Testing Optimized Detection Engine...")
    print("=" * 50)

    try:
        # Initialize optimized detection engine
        engine = OptimizedDetectionEngine()

        # Create test image
        test_image_path = create_test_image("detection_test.jpg")
        test_image = cv2.imread(test_image_path)

        # Run detection
        print("Running detection test...")
        results = engine.detect_objects(test_image)

        # Show results
        print(f"✅ Detection completed successfully")
        print(f"  • Humans detected: {len(results['humans'])}")
        print(f"  • Animals detected: {len(results['animals'])}")
        print(f"  • Processing time: {results['frame_info']['processing_time']*1000:.1f}ms")
        print(f"  • Model format: {results['frame_info']['model_format']}")

        # Get performance stats
        stats = engine.get_performance_stats()
        if stats:
            print(f"  • Average FPS: {stats['fps']:.1f}")

        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

        return True

    except Exception as e:
        logger.error(f"Optimized detection test failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up model optimization for intruder detection system")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model path")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmarking")
    parser.add_argument("--test-detection", action="store_true", help="Test optimized detection engine")

    args = parser.parse_args()

    print("🚀 Model Optimization Setup")
    print("=" * 50)

    # Check system requirements
    check_system_requirements()

    # Set up optimization
    optimizer = setup_optimization(args.model, not args.no_benchmark)

    # Test detection if requested
    if args.test_detection:
        test_optimized_detection()

    print("\n✅ Model optimization setup completed!")
    print("\n💡 Next steps:")
    print("1. Update your detection_config.py to use OptimizedDetectionEngine")
    print("2. Test the system with real camera input")
    print("3. Monitor performance improvements")


if __name__ == "__main__":
    main()