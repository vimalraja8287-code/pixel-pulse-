#!/usr/bin/env python3
"""
Quick start script for ParaDetect AI Demo
This script automatically detects if a trained model is available and runs the appropriate version.
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import flask
    except ImportError:
        missing_deps.append("flask")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    return missing_deps

def check_model_exists():
    """Check if a trained model file exists"""
    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        return False
    
    # Look for .keras model files
    model_files = list(models_dir.glob("*.keras"))
    return len(model_files) > 0

def check_port_available(port=5000):
    """Check if the port is available"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def open_browser_delayed(url, delay=3):
    """Open browser after a delay"""
    def open_browser():
        time.sleep(delay)
        print(f"🌐 Opening browser: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"📱 Please manually open: {url}")
    
    thread = threading.Thread(target=open_browser)
    thread.daemon = True
    thread.start()

def main():
    print("=" * 60)
    print("🔬 ParaDetect AI - Malaria Detection System")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\n💡 Install dependencies with:")
        print("   pip install -r requirements.txt")
        print("\n" + "=" * 60)
        sys.exit(1)
    
    # Check if port is available
    if not check_port_available(5000):
        print("❌ Port 5000 is already in use!")
        print("💡 Try one of these solutions:")
        print("   1. Close any other applications using port 5000")
        print("   2. Wait a moment and try again")
        print("   3. Use a different port")
        print("\n" + "=" * 60)
        sys.exit(1)
    
    # Check if model exists
    has_model = check_model_exists()
    
    if has_model:
        print("✅ Trained model detected!")
        print("🚀 Starting production application with AI model...")
        print("\nFeatures available:")
        print("  • Real AI-powered malaria detection")
        print("  • Trained CNN model predictions")
        print("  • Full accuracy metrics")
    else:
        print("⚠️  No trained model found in 'models/' directory")
        print("🎭 Starting demo application...")
        print("\nDemo features:")
        print("  • Simulated AI predictions")
        print("  • Full UI/UX testing")
        print("  • No model training required")
        print("\n💡 To use real AI predictions:")
        print("  1. Train a model using: python train.py")
        print("  2. Or place a .keras model file in the models/ directory")
    
    print("\n" + "=" * 60)
    
    # Start browser opening in background
    url = "http://localhost:5000"
    open_browser_delayed(url, delay=3)
    
    print("🔄 Starting server...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Browser will open automatically in 3 seconds...")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Import and run the app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("👋 Thank you for using ParaDetect AI!")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you're in the paradetect_ai directory")
        print("   2. Check that all dependencies are installed")
        print("   3. Try running: python app.py")
        sys.exit(1)

if __name__ == "__main__":
    main()