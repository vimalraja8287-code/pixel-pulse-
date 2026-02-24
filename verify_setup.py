#!/usr/bin/env python3
"""
Setup verification script for ParaDetect AI
Checks all components and provides clear guidance
"""

import os
import sys
from pathlib import Path

def check_templates():
    """Check if all required templates exist"""
    required_templates = [
        'landing.html',
        'research.html', 
        'about.html',
        'help.html'
    ]
    
    templates_dir = Path(__file__).parent / "templates"
    missing = []
    
    for template in required_templates:
        template_path = templates_dir / template
        if template_path.exists():
            print(f"✅ {template}")
        else:
            print(f"❌ {template} missing")
            missing.append(template)
    
    return len(missing) == 0

def check_static_files():
    """Check if static files exist"""
    static_files = [
        'images/medical-hero-bg.svg',
        'images/medical-research-bg.svg',
        'images/medical-bg.svg'
    ]
    
    static_dir = Path(__file__).parent / "static"
    missing = []
    
    for file_path in static_files:
        full_path = static_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            missing.append(file_path)
    
    return len(missing) == 0

def check_app_routes():
    """Test if app routes work"""
    try:
        from app import app
        
        with app.test_client() as client:
            # Test landing page
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Landing page (/) works")
            else:
                print(f"❌ Landing page failed: {response.status_code}")
                return False
            
            # Test research page
            response = client.get('/research')
            if response.status_code == 200:
                print("✅ Research page (/research) works")
            else:
                print(f"❌ Research page failed: {response.status_code}")
                return False
            
            # Test API health
            response = client.get('/api/health')
            if response.status_code == 200:
                print("✅ API health check works")
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ App route test failed: {e}")
        return False

def main():
    print("🔍 ParaDetect AI - Setup Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check templates
    print("\n📄 Checking Templates...")
    if not check_templates():
        all_good = False
    
    # Check static files
    print("\n🖼️  Checking Static Files...")
    if not check_static_files():
        all_good = False
    
    # Check app routes
    print("\n🌐 Testing App Routes...")
    if not check_app_routes():
        all_good = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your ParaDetect AI setup is perfect!")
        print("\n🚀 Ready to start:")
        print("   python run_demo.py")
        print("\n🌐 Then open: http://localhost:5000")
        print("   • Landing page with beautiful animations")
        print("   • Research page for image analysis")
        print("   • Full medical-grade interface")
    else:
        print("⚠️  SOME ISSUES FOUND")
        print("💡 Please fix the missing files above")
        print("📧 If you need help, check the error messages")
    
    print("=" * 50)
    return all_good

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)