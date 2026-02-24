#!/usr/bin/env python3
"""
Simple test script to verify the ParaDetect AI frontend is working correctly.
"""

import requests
import json
import time
from pathlib import Path

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Demo Mode: {data.get('demo_mode', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_frontend_pages():
    """Test that frontend pages load correctly"""
    pages = [
        ('/', 'Main page'),
        ('/about', 'About page'),
        ('/help', 'Help page')
    ]
    
    for path, name in pages:
        try:
            response = requests.get(f'http://localhost:5000{path}', timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} loads correctly")
            else:
                print(f"❌ {name} failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ {name} failed: {e}")
            return False
    
    return True

def create_test_image():
    """Create a simple test image for upload testing"""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple 128x128 RGB image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        test_path = Path(__file__).parent / "test_image.png"
        img.save(test_path)
        return test_path
    except Exception as e:
        print(f"❌ Could not create test image: {e}")
        return None

def test_prediction_endpoint():
    """Test the prediction endpoint with a test image"""
    test_image_path = create_test_image()
    if not test_image_path:
        return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': ('test_image.png', f, 'image/png')}
            response = requests.post('http://localhost:5000/api/predict', files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction endpoint working")
            print(f"   Result: {data.get('label')}")
            print(f"   Confidence: {data.get('confidence', 0) * 100:.1f}%")
            print(f"   Processing Time: {data.get('processing_time')}s")
            if data.get('demo_mode'):
                print("   🎭 Running in demo mode")
            
            # Clean up test image
            test_image_path.unlink()
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error')}")
            except:
                pass
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction test failed: {e}")
        return False
    finally:
        # Clean up test image if it exists
        if test_image_path and test_image_path.exists():
            test_image_path.unlink()

def main():
    print("🧪 Testing ParaDetect AI Frontend")
    print("=" * 40)
    print("Make sure the server is running first!")
    print("Run: python run_demo.py")
    print("=" * 40)
    
    # Wait a moment for user to start server
    print("Waiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Health endpoint
    print("\n1. Testing health endpoint...")
    if test_health_endpoint():
        tests_passed += 1
    
    # Test 2: Frontend pages
    print("\n2. Testing frontend pages...")
    if test_frontend_pages():
        tests_passed += 1
    
    # Test 3: Prediction endpoint
    print("\n3. Testing prediction endpoint...")
    if test_prediction_endpoint():
        tests_passed += 1
    
    # Results
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Frontend is working correctly.")
        print("\n💡 You can now:")
        print("   • Open http://localhost:5000 in your browser")
        print("   • Upload blood smear images for analysis")
        print("   • Explore the About and Help pages")
    else:
        print("❌ Some tests failed. Check the server logs.")
    
    print("=" * 40)

if __name__ == "__main__":
    main()