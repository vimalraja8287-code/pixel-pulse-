#!/usr/bin/env python3
"""
Troubleshooting script for ParaDetect AI
Helps diagnose and fix common issues
"""

import os
import sys
import socket
import webbrowser
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Python 3.8+ recommended")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_dependencies():
    """Check if all dependencies are installed"""
    required_packages = {
        'flask': 'Flask',
        'numpy': 'NumPy', 
        'PIL': 'Pillow',
        'requests': 'requests'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} missing")
            missing.append(name.lower())
    
    return missing

def check_port(port=5000):
    """Check if port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            print(f"✅ Port {port} is available")
            return True
    except OSError:
        print(f"❌ Port {port} is in use")
        return False

def check_browser():
    """Check if browser can be opened"""
    try:
        # Try to get default browser
        browser = webbrowser.get()
        print(f"✅ Default browser: {browser.name}")
        return True
    except Exception as e:
        print(f"⚠️  Browser issue: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'templates/index.html',
        'static/images/medical-bg.svg'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_server_start():
    """Test if server can start"""
    try:
        print("🧪 Testing server startup...")
        from app import app
        print("✅ App imports successfully")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def fix_suggestions(issues):
    """Provide fix suggestions based on issues found"""
    print("\n" + "=" * 50)
    print("🔧 TROUBLESHOOTING SUGGESTIONS")
    print("=" * 50)
    
    if 'python' in issues:
        print("📋 Python Version Issue:")
        print("   • Install Python 3.8 or higher")
        print("   • Download from: https://python.org/downloads/")
    
    if 'dependencies' in issues:
        print("📋 Missing Dependencies:")
        print("   • Run: pip install -r requirements.txt")
        print("   • Or install individually:")
        print("     pip install flask numpy pillow requests")
    
    if 'port' in issues:
        print("📋 Port 5000 In Use:")
        print("   • Close other applications using port 5000")
        print("   • Wait a few minutes and try again")
        print("   • Restart your computer if needed")
        print("   • Check for other Flask/web servers running")
    
    if 'files' in issues:
        print("📋 Missing Files:")
        print("   • Make sure you're in the paradetect_ai directory")
        print("   • Re-download the project files if needed")
        print("   • Check that all files were extracted properly")
    
    if 'server' in issues:
        print("📋 Server Startup Issues:")
        print("   • Check the error messages above")
        print("   • Try running: python app.py directly")
        print("   • Make sure all dependencies are installed")
    
    if 'browser' in issues:
        print("📋 Browser Opening Issues:")
        print("   • Browser will be opened automatically")
        print("   • If it doesn't open, manually go to: http://localhost:5000")
        print("   • Try a different browser if needed")
        print("   • Check your default browser settings")

def main():
    print("🔍 ParaDetect AI - Troubleshooting Tool")
    print("=" * 50)
    
    issues = []
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        issues.append('python')
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        issues.append('dependencies')
    
    # Check port availability
    print("\n3. Checking port availability...")
    if not check_port(5000):
        issues.append('port')
    
    # Check browser
    print("\n4. Checking browser...")
    if not check_browser():
        issues.append('browser')
    
    # Check required files
    print("\n5. Checking required files...")
    if not check_files():
        issues.append('files')
    
    # Test server startup
    print("\n6. Testing server startup...")
    if not test_server_start():
        issues.append('server')
    
    # Results
    print("\n" + "=" * 50)
    if not issues:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your system should work perfectly")
        print("\n💡 To start the application:")
        print("   • Run: python run_demo.py")
        print("   • Or double-click: start_app.bat")
        print("   • Browser should open automatically")
    else:
        print(f"⚠️  Found {len(issues)} issue(s)")
        fix_suggestions(issues)
    
    print("\n" + "=" * 50)
    print("🆘 If you still have issues:")
    print("   • Try running: python app.py")
    print("   • Then manually open: http://localhost:5000")
    print("   • Check the console for error messages")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")