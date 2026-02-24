@echo off
echo ========================================
echo ParaDetect AI - Malaria Detection System
echo ========================================
echo.
echo Starting the web application...
echo Browser will open automatically in 3 seconds...
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python run_demo.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo Error occurred! Troubleshooting tips:
    echo ========================================
    echo 1. Make sure Python is installed
    echo 2. Install dependencies: pip install -r requirements.txt
    echo 3. Make sure you're in the paradetect_ai folder
    echo 4. Try running: python app.py
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo Server stopped successfully
    echo Thank you for using ParaDetect AI!
    echo ========================================
)

pause