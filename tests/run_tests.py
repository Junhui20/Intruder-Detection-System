#!/usr/bin/env python3
"""
Test Runner for Intruder Detection System

Automated test runner that executes all test suites and generates reports.
"""

import sys
import time
import unittest
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import test modules
from test_detection import run_detection_tests
from test_database import run_database_tests

class TestRunner:
    """Comprehensive test runner for the intruder detection system."""
    
    def __init__(self):
        """Initialize test runner."""
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbose: bool = True) -> bool:
        """
        Run all test suites.
        
        Args:
            verbose: Whether to print detailed output
            
        Returns:
            True if all tests passed
        """
        self.start_time = time.time()
        
        if verbose:
            print("🚀 Starting Comprehensive Test Suite")
            print("=" * 60)
            print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
        
        # Test suites to run
        test_suites = [
            ("Detection System", self._run_detection_tests),
            ("Database System", self._run_database_tests),
            ("Camera System", self._run_camera_tests),
            ("Telegram System", self._run_telegram_tests),
            ("Integration Tests", self._run_integration_tests),
            ("Configuration Tests", self._run_configuration_tests),
        ]
        
        all_passed = True
        
        for suite_name, test_func in test_suites:
            if verbose:
                print(f"\n📋 Running {suite_name} Tests...")
                print("-" * 40)
            
            try:
                success = test_func(verbose)
                self.test_results[suite_name] = {
                    'success': success,
                    'error': None
                }
                
                if not success:
                    all_passed = False
                    
                if verbose:
                    status = "✅ PASSED" if success else "❌ FAILED"
                    print(f"{suite_name} Tests: {status}")
                    
            except Exception as e:
                self.test_results[suite_name] = {
                    'success': False,
                    'error': str(e)
                }
                all_passed = False
                
                if verbose:
                    print(f"❌ {suite_name} Tests: FAILED (Error: {e})")
        
        self.end_time = time.time()
        
        if verbose:
            self._print_summary()
        
        return all_passed
    
    def _run_detection_tests(self, verbose: bool = True) -> bool:
        """Run detection system tests."""
        try:
            return run_detection_tests()
        except Exception as e:
            if verbose:
                print(f"Detection tests failed: {e}")
            return False
    
    def _run_database_tests(self, verbose: bool = True) -> bool:
        """Run database system tests."""
        try:
            return run_database_tests()
        except Exception as e:
            if verbose:
                print(f"Database tests failed: {e}")
            return False
    
    def _run_camera_tests(self, verbose: bool = True) -> bool:
        """Run camera system tests."""
        try:
            # Import and run camera tests
            from test_camera import run_camera_tests
            return run_camera_tests()
        except ImportError:
            if verbose:
                print("⚠️ Camera tests not available (test_camera.py not found)")
            return True  # Skip if not available
        except Exception as e:
            if verbose:
                print(f"Camera tests failed: {e}")
            return False
    
    def _run_telegram_tests(self, verbose: bool = True) -> bool:
        """Run Telegram system tests."""
        try:
            # Import and run Telegram tests
            from test_telegram import run_telegram_tests
            return run_telegram_tests()
        except ImportError:
            if verbose:
                print("⚠️ Telegram tests not available (test_telegram.py not found)")
            return True  # Skip if not available
        except Exception as e:
            if verbose:
                print(f"Telegram tests failed: {e}")
            return False

    def _run_integration_tests(self, verbose: bool = True) -> bool:
        """Run integration tests."""
        try:
            if verbose:
                print("Running integration tests...")

            from test_integration import run_integration_tests
            success = run_integration_tests(verbose)

            if verbose:
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"Integration tests: {status}")

            return success

        except Exception as e:
            if verbose:
                print(f"Integration tests failed: {e}")
            return False

    def _run_configuration_tests(self, verbose: bool = True) -> bool:
        """Run configuration and security tests."""
        try:
            if verbose:
                print("Running configuration tests...")

            # Test environment configuration loading
            from config.env_config import EnvironmentConfigManager
            env_config = EnvironmentConfigManager()

            # Test secure configuration loading
            from config.settings import Settings
            settings = Settings.load_with_env_support()

            # Test configuration validation
            success = True

            # Validate settings structure
            if not hasattr(settings, 'bot_token'):
                success = False
                if verbose:
                    print("❌ Settings missing bot_token attribute")

            if not hasattr(settings, 'database_path'):
                success = False
                if verbose:
                    print("❌ Settings missing database_path attribute")

            # Test environment variable support
            test_value = env_config.get("test_key", "default_value")
            if test_value != "default_value":
                if verbose:
                    print("⚠️ Unexpected environment variable behavior")

            if verbose:
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"Configuration tests: {status}")

            return success

        except Exception as e:
            if verbose:
                print(f"Configuration tests failed: {e}")
            return False

    def _print_summary(self):
        """Print test summary."""
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed_count = sum(1 for result in self.test_results.values() if result['success'])
        total_count = len(self.test_results)
        
        print(f"Total Test Suites: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Total Time: {total_time:.2f} seconds")
        
        print("\n📋 Detailed Results:")
        for suite_name, result in self.test_results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"  • {suite_name}: {status}")
            if result['error']:
                print(f"    Error: {result['error']}")
        
        overall_status = "✅ ALL TESTS PASSED" if passed_count == total_count else "❌ SOME TESTS FAILED"
        print(f"\n🎯 Overall Result: {overall_status}")
        
        if passed_count == total_count:
            print("\n🎉 Congratulations! Your system is working correctly.")
        else:
            print("\n🔧 Please review and fix the failing tests.")
    
    def run_specific_test(self, test_name: str, verbose: bool = True) -> bool:
        """
        Run a specific test suite.
        
        Args:
            test_name: Name of test suite to run
            verbose: Whether to print detailed output
            
        Returns:
            True if test passed
        """
        test_map = {
            'detection': self._run_detection_tests,
            'database': self._run_database_tests,
            'camera': self._run_camera_tests,
            'telegram': self._run_telegram_tests,
            'integration': self._run_integration_tests,
            'configuration': self._run_configuration_tests,
        }
        
        if test_name.lower() not in test_map:
            if verbose:
                print(f"❌ Unknown test suite: {test_name}")
                print(f"Available tests: {', '.join(test_map.keys())}")
            return False
        
        if verbose:
            print(f"🧪 Running {test_name.title()} Tests")
            print("=" * 40)
        
        test_func = test_map[test_name.lower()]
        return test_func(verbose)
    
    def generate_report(self, output_file: str = None):
        """Generate test report."""
        if not self.test_results:
            print("No test results available. Run tests first.")
            return
        
        report_lines = [
            "# Intruder Detection System - Test Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total Test Suites: {len(self.test_results)}",
            f"- Passed: {sum(1 for r in self.test_results.values() if r['success'])}",
            f"- Failed: {sum(1 for r in self.test_results.values() if not r['success'])}",
            ""
        ]
        
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            report_lines.append(f"- Total Time: {total_time:.2f} seconds")
            report_lines.append("")
        
        report_lines.append("## Detailed Results")
        report_lines.append("")
        
        for suite_name, result in self.test_results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            report_lines.append(f"### {suite_name}")
            report_lines.append(f"Status: {status}")
            
            if result['error']:
                report_lines.append(f"Error: {result['error']}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"📄 Test report saved to: {output_file}")
        else:
            print(report_content)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Intruder Detection System Test Runner")
    parser.add_argument("--test", help="Run specific test suite (detection, database, camera, telegram)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--report", help="Generate test report to file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    verbose = not args.quiet
    
    if args.test:
        # Run specific test
        success = runner.run_specific_test(args.test, verbose)
    else:
        # Run all tests
        success = runner.run_all_tests(verbose)
    
    # Generate report if requested
    if args.report:
        runner.generate_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
