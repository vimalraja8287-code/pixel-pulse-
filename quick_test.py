#!/usr/bin/env python3
"""
Quick test to verify ParaDetect AI is working correctly
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if basic dependencies are available"""
    try:
        import flask
        import numpy
        from PIL import Image
        print("✅ All basic dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_app_import():
    """Test if the app can be imported without errors"""
    try:
        from app import app
        print("✅ App imports successfully")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        test_path = Path(__file__).parent / "test_sample.png"
        img.save(test_path)
        print(f"✅ Test image created: {test_path}")
        return test_path
    except Exception as e:
        print(f"❌ Could not create test image: {e}")
        return None

def main():
    print("🧪 ParaDetect AI - Quick Test")
    print("=" * 40)
    
    # Test 1: Check dependencies
    print("1. Checking dependencies...")
    if not check_dependencies():
        print("\n💡 Install missing dependencies with:")
        print("   pip install flask numpy pillow")
        return False
    
    # Test 2: Test app import
    print("\n2. Testing app import...")
    if not test_app_import():
        print("\n💡 Check for syntax errors in app.py")
        return False
    
    # Test 3: Create test image
    print("\n3. Creating test image...")
    test_image = create_test_image()
    if not test_image:
        return False
    
    # Test 4: Check static files
    print("\n4. Checking static files...")
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        print("✅ Static directory exists")
        
        bg_image = static_dir / "images" / "medical-bg.svg"
        if bg_image.exists():
            print("✅ Background image exists")
        else:
            print("⚠️  Background image missing (will use fallback)")
    else:
        print("⚠️  Static directory missing")
    
    print("\n" + "=" * 40)
    print("🎉 Basic tests passed!")
    print("\n💡 To start the application:")
    print("   python run_demo.py")
    print("\n🌐 Then open: http://localhost:5000")
    
    # Clean up test image
    if test_image and test_image.exists():
        test_image.unlink()
        print(f"\n🧹 Cleaned up test image")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)