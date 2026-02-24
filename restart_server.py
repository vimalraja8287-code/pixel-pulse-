#!/usr/bin/env python3
"""
Restart server script to see changes immediately
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

def main():
    print("🔄 Restarting ParaDetect AI Server...")
    print("=" * 50)
    
    # Kill any existing processes on port 5000 (Windows)
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, check=False)
        time.sleep(1)
    except:
        pass
    
    print("🧹 Clearing any cached processes...")
    time.sleep(2)
    
    print("🚀 Starting fresh server...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Browser will open automatically...")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser after delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:5000")
            print("🌐 Browser opened!")
        except:
            print("⚠️  Please manually open: http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()