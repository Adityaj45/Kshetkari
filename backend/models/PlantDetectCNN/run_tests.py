#!/usr/bin/env python3
"""
Comprehensive test runner for Plant Detection Model
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_unit_tests():
    """Run unit tests"""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_model.py', 
            '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running unit tests: {e}")
        return False

def run_benchmarks():
    """Run performance benchmarks"""
    print("=" * 60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, 'tests/benchmark_model.py'
        ], cwd=os.path.dirname(__file__))
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return False

def start_web_app():
    """Start the web application"""
    print("=" * 60)
    print("STARTING WEB APPLICATION")
    print("=" * 60)
    print("Web app will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, 'web_app/app.py'
        ], cwd=os.path.dirname(__file__))
    except KeyboardInterrupt:
        print("\nWeb application stopped.")
    except Exception as e:
        print(f"Error starting web app: {e}")

def install_requirements():
    """Install required packages"""
    print("=" * 60)
    print("INSTALLING REQUIREMENTS")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], cwd=os.path.dirname(__file__))
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error installing requirements: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Plant Detection Model Test Runner')
    parser.add_argument('--install', action='store_true', help='Install requirements')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--web', action='store_true', help='Start web application')
    parser.add_argument('--all', action='store_true', help='Run tests and benchmarks')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # No arguments provided, show help
        parser.print_help()
        return
    
    print(f"Plant Detection Model Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    if args.install:
        success &= install_requirements()
    
    if args.test or args.all:
        success &= run_unit_tests()
    
    if args.benchmark or args.all:
        success &= run_benchmarks()
    
    if args.web:
        start_web_app()
    
    if args.all or args.test or args.benchmark:
        if success:
            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED!")
            print("=" * 60)
            sys.exit(1)

if __name__ == "__main__":
    main()