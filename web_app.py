from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import re
import os
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

# PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ PDF libraries not available. Install: pip install PyPDF2 pdfplumber PyMuPDF")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables
model = None
tfidf_vectorizer = None
label_encoder = None
model_loaded = False

class ResumeClassifier:
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        
    def preprocess_text(self, text):
        """Clean and preprocess resume text"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s.,;:()\-@]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_model(self, model_path, preprocessor_path):
        """Load the trained model and preprocessors"""
        try:
            self.model = keras.models.load_model(model_path)
            
            with open(preprocessor_path, 'rb') as f:
                data = pickle.load(f)
                self.tfidf_vectorizer = data['tfidf_vectorizer']
                self.label_encoder = data['label_encoder']
            
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, resume_text, top_k=5):
        """Predict resume category"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess text
        cleaned_text = self.preprocess_text(resume_text)
        
        if not cleaned_text:
            return [{"label": "Unknown", "confidence": 0.0}]
        
        # Transform to TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform([cleaned_text])
        
        # Predict
        predictions = self.model.predict(tfidf_features.toarray(), verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            category = self.label_encoder.classes_[idx]
            confidence = float(predictions[0][idx])
            results.append({
                'label': category,
                'confidence': confidence
            })
        
        return results

# Initialize classifier
classifier = ResumeClassifier()

# PDF extraction function
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using multiple methods"""
    if not PDF_SUPPORT:
        return None
    
    # Try PyMuPDF first
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text() + "\n"
        doc.close()
        if text.strip():
            return text
    except:
        pass
    
    # Try pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except:
        pass
    
    # Try PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        if text.strip():
            return text
    except:
        pass
    
    return None

# Load model on startup
def load_model_on_startup():
    """Try to load model from common locations"""
    global model_loaded
    
    possible_paths = [
        ("resume_classifier_nn_model.h5", "resume_classifier_nn_preprocessors.pkl"),
        ("my_resume_classifier_model.h5", "my_resume_classifier_preprocessors.pkl"),
        ("resume_classifier_v1_model.h5", "resume_classifier_v1_preprocessors.pkl"),
        (r"C:\Users\ASUS\Desktop\Neural Network\resume_classifier_nn_model.h5", 
         r"C:\Users\ASUS\Desktop\Neural Network\resume_classifier_nn_preprocessors.pkl"),
        ("models/resume_classifier_nn_model.h5", "models/resume_classifier_nn_preprocessors.pkl")
    ]
    
    for model_path, preprocessor_path in possible_paths:
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            print(f"🔍 Found model files: {model_path}")
            model_loaded = classifier.load_model(model_path, preprocessor_path)
            if model_loaded:
                print(f"✅ Model loaded from: {model_path}")
                return True
    
    print("❌ No model files found. Please ensure model files are in the correct location.")
    return False

# Load model
model_loaded = load_model_on_startup()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf']

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Resume Classification System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            padding: 30px;
            text-align: center;
            color: white;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .status-bar {
            background: {{ 'linear-gradient(135deg, #55a3ff 0%, #003d82 100%)' if model_loaded else 'linear-gradient(135deg, #ff9ff3 0%, #f368e0 100%)' }};
            padding: 15px;
            text-align: center;
            color: white;
            font-weight: 600;
        }

        .main-content {
            padding: 40px;
        }

        .input-methods {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .input-method {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }

        .input-method:hover {
            border-color: #ff6b6b;
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }

        .input-method h3 {
            color: #ff6b6b;
            margin-bottom: 15px;
            font-size: 1.3em;
        }

        .file-upload-area {
            border: 3px dashed #ff6b6b;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #fff;
        }

        .file-upload-area:hover {
            background: #fff5f5;
            border-color: #ee5a24;
        }

        .file-upload-area.dragover {
            background: #ffe6e6;
            border-color: #ff3838;
        }

        .upload-icon {
            font-size: 3em;
            color: #ff6b6b;
            margin-bottom: 15px;
        }

        .file-input {
            display: none;
        }

        .input-textarea {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s ease;
        }

        .input-textarea:focus {
            outline: none;
            border-color: #ff6b6b;
            box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1);
        }

        .predict-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
            display: block;
            margin: 20px auto;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(255, 107, 107, 0.4);
        }

        .predict-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .file-info {
            background: #e8f5e8;
            border: 1px solid #4caf50;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            display: none;
        }

        .results-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            border-left: 5px solid #ff6b6b;
        }

        .prediction-item {
            background: white;
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .prediction-item.top {
            border-left: 4px solid #ff6b6b;
            background: linear-gradient(135deg, #fff5f5 0%, #fff 100%);
        }

        .confidence-bar {
            width: 200px;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin-left: 20px;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #ee5a24);
            border-radius: 5px;
            transition: width 0.5s ease;
        }

        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #ff6b6b;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #ffcdd2;
        }

        .model-info {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }

        .example-section {
            background: #f0f8f0;
            border: 1px solid #4caf50;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }

        .example-text {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            border-left: 4px solid #4caf50;
            transition: all 0.3s ease;
        }

        .example-text:hover {
            background: #f0f8ff;
            transform: translateX(5px);
        }

        @media (max-width: 768px) {
            .input-methods {
                grid-template-columns: 1fr;
            }
            
            .container {
                margin: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐱 Resume Classification System</h1>
            <p>AI-powered professional category classification</p>
        </div>
        
        <div class="status-bar">
            {% if model_loaded %}
                ✅ Neural Network Model: Loaded & Ready | Test Accuracy: 61.77%
            {% else %}
                ⚠️ Model Not Loaded - Please check model files
            {% endif %}
        </div>
        
        <div class="main-content">
            {% if model_loaded %}
            <div class="model-info">
                <h3>🧠 Model Information</h3>
                <p><strong>Type:</strong> Custom Neural Network trained on your data</p>
                <p><strong>Categories:</strong> 24 professional fields (Healthcare, Engineering, IT, etc.)</p>
                <p><strong>Performance:</strong> 61.77% accuracy on test data</p>
                <p><strong>Last Updated:</strong> {{ current_time }}</p>
            </div>
            
            <div class="input-methods">
                <!-- PDF Upload Method -->
                <div class="input-method">
                    <h3>📄 Upload PDF Resume</h3>
                    {% if pdf_support %}
                    <div class="file-upload-area" onclick="document.getElementById('fileInput').click()" 
                         ondrop="dropHandler(event);" ondragover="dragOverHandler(event);" ondragleave="dragLeaveHandler(event);">
                        <div class="upload-icon">📄</div>
                        <p><strong>Click to browse</strong> or drag & drop PDF here</p>
                        <p style="color: #666; font-size: 0.9em; margin-top: 10px;">Max file size: 16MB</p>
                    </div>
                    <input type="file" id="fileInput" class="file-input" accept=".pdf" onchange="handleFileSelect(event)">
                    <div id="fileInfo" class="file-info"></div>
                    {% else %}
                    <div style="background: #ffebee; padding: 20px; border-radius: 8px; text-align: center;">
                        <p>PDF support not available. Install required packages:</p>
                        <code>pip install PyPDF2 pdfplumber PyMuPDF</code>
                    </div>
                    {% endif %}
                </div>
                
                <!-- Text Input Method -->
                <div class="input-method">
                    <h3>📝 Paste Resume Text</h3>
                    <textarea 
                        id="textInput" 
                        class="input-textarea" 
                        placeholder="Paste resume content here for classification..."
                    ></textarea>
                </div>
            </div>
            
            <div class="example-section">
                <h4>💡 Try these sample resumes:</h4>
                <div class="example-text" onclick="fillExample(0)">
                    <strong>Software Engineer:</strong> "Experienced software engineer with 5+ years in Python, JavaScript, and machine learning. Built scalable web applications, implemented CI/CD pipelines, and led development teams."
                </div>
                <div class="example-text" onclick="fillExample(1)">
                    <strong>Healthcare Professional:</strong> "Registered nurse with 8 years experience in critical care, patient assessment, and emergency procedures. Certified in ACLS and specializing in cardiac care."
                </div>
                <div class="example-text" onclick="fillExample(2)">
                    <strong>Financial Analyst:</strong> "Financial analyst with expertise in investment portfolio management, risk assessment, and financial modeling. CPA certified with experience in corporate finance."
                </div>
            </div>
            
            <button class="predict-btn" onclick="classifyResume()" id="predictBtn">
                🚀 Classify Resume
            </button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing resume with neural network...</p>
            </div>
            
            <div class="results-section" id="results" style="display: none;">
                <h3>🎯 Classification Results</h3>
                <div id="predictionResults"></div>
            </div>
            
            {% else %}
            <div class="error-message">
                <h3>❌ Model Not Available</h3>
                <p>The neural network model could not be loaded. Please ensure the following files exist:</p>
                <ul>
                    <li>resume_classifier_nn_model.h5</li>
                    <li>resume_classifier_nn_preprocessors.pkl</li>
                </ul>
                <p>Place these files in the same directory as this application.</p>
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        let selectedFile = null;

        const examples = [
            "Experienced software engineer with 5+ years in Python, JavaScript, and machine learning. Built scalable web applications, implemented CI/CD pipelines, and led development teams. Strong background in data structures, algorithms, and system design.",
            "Registered nurse with 8 years experience in critical care, patient assessment, and emergency procedures. Certified in ACLS and specializing in cardiac care. Experience in ICU, emergency department, and patient education.",
            "Financial analyst with expertise in investment portfolio management, risk assessment, and financial modeling. CPA certified with experience in corporate finance and regulatory compliance."
        ];

        function fillExample(index) {
            document.getElementById('textInput').value = examples[index];
            selectedFile = null;
            document.getElementById('fileInfo').style.display = 'none';
        }

        function dragOverHandler(ev) {
            ev.preventDefault();
            ev.currentTarget.classList.add('dragover');
        }

        function dragLeaveHandler(ev) {
            ev.currentTarget.classList.remove('dragover');
        }

        function dropHandler(ev) {
            ev.preventDefault();
            ev.currentTarget.classList.remove('dragover');
            
            const files = ev.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }

        function handleFile(file) {
            if (file.type !== 'application/pdf') {
                showError('Please select a PDF file.');
                return;
            }

            if (file.size > 16 * 1024 * 1024) {
                showError('File size must be less than 16MB.');
                return;
            }

            selectedFile = file;
            
            const fileInfo = document.getElementById('fileInfo');
            fileInfo.style.display = 'block';
            fileInfo.innerHTML = `
                <strong>📄 Selected File:</strong> ${file.name}<br>
                <strong>📏 Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                <strong>📅 Modified:</strong> ${new Date(file.lastModified).toLocaleDateString()}
            `;
            
            document.getElementById('textInput').value = '';
        }

        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('predictBtn').disabled = true;
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('predictBtn').disabled = false;
        }

        function showError(message) {
            const resultsDiv = document.getElementById('results');
            const errorHtml = `<div class="error-message">❌ ${message}</div>`;
            document.getElementById('predictionResults').innerHTML = errorHtml;
            resultsDiv.style.display = 'block';
        }

        function displayResults(predictions) {
            const resultsDiv = document.getElementById('results');
            const predictionResults = document.getElementById('predictionResults');
            
            let html = '<div style="margin-bottom: 15px;"><strong>🏆 Top Professional Categories:</strong></div>';
            predictions.forEach((pred, index) => {
                const confidence = (pred.confidence * 100).toFixed(1);
                const isTop = index === 0;
                html += `
                    <div class="prediction-item ${isTop ? 'top' : ''}">
                        <div>
                            <strong style="color: ${isTop ? '#ff6b6b' : '#333'}; font-size: ${isTop ? '1.2em' : '1em'};">
                                ${isTop ? '🥇 ' : `${index + 1}. `}${pred.label}
                            </strong>
                            <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                                Confidence: ${confidence}%
                            </div>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                `;
            });
            
            predictionResults.innerHTML = html;
            resultsDiv.style.display = 'block';
        }

        async function classifyResume() {
            const textInput = document.getElementById('textInput').value.trim();
            
            if (!selectedFile && !textInput) {
                showError('Please upload a PDF file or paste resume text.');
                return;
            }

            showLoading();

            try {
                let response;
                
                if (selectedFile) {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    
                    response = await fetch('/predict-pdf', {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: textInput })
                    });
                }

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const predictions = await response.json();
                
                if (predictions.error) {
                    throw new Error(predictions.error);
                }
                
                displayResults(predictions);
                
            } catch (error) {
                showError('An error occurred while classifying the resume. Please try again.');
                console.error('Classification error:', error);
            } finally {
                hideLoading();
            }
        }

        // Smooth page load animation
        document.addEventListener('DOMContentLoaded', function() {
            const container = document.querySelector('.container');
            container.style.opacity = '0';
            container.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                container.style.transition = 'all 0.6s ease';
                container.style.opacity = '1';
                container.style.transform = 'translateY(0)';
            }, 100);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return render_template_string(HTML_TEMPLATE, 
                                model_loaded=model_loaded, 
                                pdf_support=PDF_SUPPORT,
                                current_time=current_time)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Please check model files."}), 500
    
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "No 'text' field provided in request"}), 400
            
        resume_text = data['text'].strip()
        if not resume_text:
            return jsonify({"error": "Empty text provided"}), 400
        
        print(f"📝 Classifying text: {resume_text[:100]}...")
        
        predictions = classifier.predict(resume_text)
        
        print(f"🎯 Top prediction: {predictions[0]['label']} ({predictions[0]['confidence']:.3f})")
        
        return jsonify(predictions)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-pdf', methods=['POST'])
def predict_pdf():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Please check model files."}), 500
    
    if not PDF_SUPPORT:
        return jsonify({"error": "PDF support not available. Install required packages."}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        print(f"📄 Processing uploaded PDF: {filename}")
        
        try:
            raw_text = extract_text_from_pdf(temp_path)
            
            if not raw_text:
                return jsonify({"error": "Could not extract text from PDF. Please try a different file or paste text manually."}), 400
            
            print(f"📝 Extracted {len(raw_text)} characters from PDF")
            
            predictions = classifier.predict(raw_text)
            
            print(f"🎯 Top prediction: {predictions[0]['label']} ({predictions[0]['confidence']:.3f})")
            
            return jsonify(predictions)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

@app.route('/health')
def health():
    return {
        "status": "healthy" if model_loaded else "error",
        "model_loaded": model_loaded,
        "pdf_support": PDF_SUPPORT,
        "accuracy": "61.77%",
        "categories": len(classifier.label_encoder.classes_) if model_loaded else 0
    }

@app.route('/model-info')
def model_info():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": "Custom Neural Network",
        "test_accuracy": "61.77%",
        "categories": list(classifier.label_encoder.classes_),
        "num_categories": len(classifier.label_encoder.classes_),
        "features": "TF-IDF vectorization",
        "architecture": "Multi-layer dense neural network"
    })

if __name__ == '__main__':
    print("🚀 Starting Resume Classification Web Interface...")
    print("📊 Custom Neural Network Resume Classifier")
    
    if model_loaded:
        print("✅ Model loaded successfully!")
        print(f"📊 Categories: {len(classifier.label_encoder.classes_)}")
        print(f"🎯 Test Accuracy: 61.77%")
    else:
        print("⚠️ Model not loaded. Please check model files.")
    
    if PDF_SUPPORT:
        print("📄 PDF processing enabled")
    else:
        print("⚠️ PDF support disabled. Install: pip install PyPDF2 pdfplumber PyMuPDF")
    
    print("🌐 Server starting at: http://localhost:5000")
    print("💡 Make sure your model files are in the same directory!")
    app.run(debug=True, port=5000, host='127.0.0.1')