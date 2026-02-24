#!/usr/bin/env python3
"""
Enhanced startup script for ParaDetect AI that automatically opens the browser
"""

import os
import sys
import time
import webbrowser
import threading
import subprocess
from pathlib import Path

def check_port_available(port=5000):
    """Check if the port is available"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def wait_for_server(url, timeout=30):
    """Wait for the server to be ready"""
    import requests
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    
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
    print("🚀 Starting ParaDetect AI with Auto-Browser Opening")
    print("=" * 60)
    
    # Check if port is available
    if not check_port_available(5000):
        print("❌ Port 5000 is already in use!")
        print("💡 Try one of these solutions:")
        print("   1. Close any other applications using port 5000")
        print("   2. Wait a moment and try again")
        print("   3. Restart your computer if needed")
        return False
    
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
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you're in the paradetect_ai directory")
        print("   2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("   3. Try running: python app.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)