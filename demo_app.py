"""
Demo version of ParaDetect AI that works without a trained model.
This is useful for testing the frontend and demonstrating the application.
"""

from flask import Flask, render_template, request, jsonify
import os
import time
import random
from datetime import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Demo mode - simulate model predictions
DEMO_MODE = True

@app.route("/")
def index():
    """Main diagnosis page"""
    return render_template("index.html")

@app.route("/about")
def about():
    """About page with project information"""
    return render_template("about.html")

@app.route("/help")
def help_page():
    """Help and FAQ page"""
    return render_template("help.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for malaria prediction - DEMO VERSION"""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext not in allowed_extensions:
        return jsonify({"error": "Invalid file format. Please upload PNG, JPG, or JPEG files."}), 400

    # Save to temp directory (for demo purposes)
    upload_dir = os.path.join(app.instance_path, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_filename = f"temp_{timestamp}_{file.filename}"
    temp_path = os.path.join(upload_dir, temp_filename)
    
    try:
        file.save(temp_path)
        
        # Simulate processing time
        start_time = time.time()
        time.sleep(random.uniform(1.0, 2.5))  # Simulate AI processing
        processing_time = time.time() - start_time
        
        # Generate demo predictions
        # Simulate realistic probability distributions
        if random.random() < 0.3:  # 30% chance of parasitized
            # Parasitized case
            parasitized_prob = random.uniform(0.65, 0.95)
            uninfected_prob = 1.0 - parasitized_prob
            label = "Parasitized"
            confidence = parasitized_prob
        else:
            # Uninfected case
            uninfected_prob = random.uniform(0.70, 0.98)
            parasitized_prob = 1.0 - uninfected_prob
            label = "Uninfected"
            confidence = uninfected_prob
        
        return jsonify({
            "label": label,
            "confidence": float(confidence),
            "probabilities": {
                "Uninfected": float(uninfected_prob),
                "Parasitized": float(parasitized_prob),
            },
            "processing_time": round(processing_time, 3),
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True
        })
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "demo_mode": True,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template("index.html"), 404

if __name__ == "__main__":
    print("Starting ParaDetect AI Web Application (DEMO MODE)...")
    print("This demo version simulates AI predictions for testing purposes.")
    print("To use the real model, run app.py instead.")
    app.run(debug=True, host='0.0.0.0', port=5000)