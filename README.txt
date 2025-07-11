# Resume Classification System 🐱

An AI-powered resume classification system that automatically categorizes resumes into 24 professional fields using a custom neural network.

## 🌟 Features

- **Custom Neural Network**: Built from scratch with TensorFlow/Keras
- **PDF Processing**: Automatic text extraction from PDF resumes
- **Web Interface**: User-friendly web application with drag & drop functionality
- **24 Categories**: Classifies into Healthcare, Engineering, IT, Finance, and 20+ other fields
- **Real-time Processing**: Instant classification with confidence scores
- **High Accuracy**: Achieved 61.77% accuracy on 24-class classification

## 🎯 Live Demo

*Upload a resume and get instant professional category predictions!*

## 📊 Model Performance

- **Test Accuracy**: 61.77%
- **Architecture**: Multi-layer Neural Network with TF-IDF features
- **Training Data**: PDF resumes across 24 professional categories
- **Performance**: ~15x better than random guessing (4.2% baseline)

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
# Clone the repository
git clone https://github.com/jrtyong15/resume-classification-system.git
cd resume-classification-system

# Install dependencies
pip install -r requirements.txt

# Run the web application
python web_app.py
```

### Usage
1. Open http://localhost:5000 in your browser
2. Upload a PDF resume or paste text
3. Get instant professional category classification
4. View confidence scores for top 5 predictions

## 🏗️ Architecture

### Neural Network Design
```
Input Layer (TF-IDF Features)
    ↓
Dense Layer (512 neurons) + ReLU + Dropout(0.3)
    ↓
Dense Layer (256 neurons) + ReLU + BatchNorm + Dropout(0.4)
    ↓
Dense Layer (128 neurons) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (64 neurons) + ReLU + Dropout(0.2)
    ↓
Output Layer (24 neurons) + Softmax
```

### Tech Stack
- **Backend**: Python, Flask, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Pipeline**: scikit-learn, pandas, numpy
- **PDF Processing**: PyMuPDF, pdfplumber, PyPDF2

## 📁 Project Structure

```
resume-classification-system/
├── 📓 notebooks/
│   └── Resume_Classifier_Trainer.ipynb    # Training notebook
├── 🧠 models/
│   ├── resume_classifier_nn_model.h5      # Trained model
│   └── resume_classifier_nn_preprocessors.pkl
├── 📄 data/
│   └── sample_resumes/                     # Sample data
├── 🌐 web_app.py                          # Flask web application
├── 🔧 train_model.py                      # Model training script
├── 📋 requirements.txt                    # Dependencies
├── 📖 README.md                           # This file
└── 🖼️ screenshots/                        # Demo images
```

## 📈 Training Process

### Data Preparation
1. **PDF Text Extraction**: Multi-method approach (PyMuPDF → pdfplumber → PyPDF2)
2. **Text Preprocessing**: Lowercasing, special character removal, whitespace normalization
3. **Feature Engineering**: TF-IDF vectorization with n-grams (1,2)
4. **Label Encoding**: 24 professional categories

### Model Training
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Regularization**: Dropout + Batch Normalization
- **Callbacks**: Early Stopping, Learning Rate Reduction
- **Validation**: 80/20 train-test split with stratification

## 🎯 Professional Categories

The system classifies resumes into these 24 categories:

| Category | Category | Category | Category |
|----------|----------|----------|----------|
| Accountant | Advocate | Agriculture | Apparel |
| Arts | Automobile | Aviation | Banking |
| BPO | Business-Development | Chef | Construction |
| Consultant | Designer | Digital-Media | Engineering |
| Finance | Healthcare | HR | Information-Technology |
| Public-Relations | Sales | Teacher | Fitness |

## 🔬 Technical Deep Dive

### Feature Engineering
- **TF-IDF Vectorization**: 10,000 max features
- **N-gram Range**: (1,2) for capturing phrases
- **Stop Words**: English stop words removed
- **Min/Max DF**: Balanced frequency filtering

### Model Architecture Decisions
- **Dense Layers**: Chosen for tabular TF-IDF features
- **Dropout**: Prevents overfitting on small dataset
- **Batch Normalization**: Stabilizes training
- **ReLU Activation**: Standard for hidden layers
- **Softmax Output**: Multi-class probability distribution

### Performance Metrics
```python
Classification Report:
                     precision    recall  f1-score   support
    Healthcare          0.75      0.82      0.78        22
    Engineering         0.68      0.71      0.69        28
    Information-Tech    0.59      0.61      0.60        31
    Finance             0.72      0.68      0.70        19
    ...
    
    macro avg           0.62      0.61      0.61       200
    weighted avg        0.62      0.62      0.62       200
```

## 🌐 Web Interface Features

### User Experience
- **Responsive Design**: Works on desktop and mobile
- **Drag & Drop**: Intuitive file upload
- **Real-time Feedback**: Loading indicators and progress
- **Visual Results**: Confidence bars and color-coded predictions

### Technical Features
- **Multi-format Support**: PDF upload or text input
- **Error Handling**: Graceful failure messages
- **File Validation**: Size and format checking
- **Secure Upload**: Temporary file handling

## 🔮 Future Enhancements

- [ ] **Improved Accuracy**: Experiment with state of the art transformer models (BERT, RoBERTa)
- [ ] **More Categories**: Expand to industry-specific roles
- [ ] **Large Dataset**: We can improve the accuracy of the model by adding more training data
- [ ] **Batch Processing**: Handle multiple resumes simultaneously
- [ ] **API Development**: RESTful API for integration
- [ ] **Cloud Deployment**: AWS/Azure hosting
- [ ] **Resume Parsing**: Extract structured information (skills, experience)
- [ ] **Similarity Search**: Find similar resumes in database
- [ ] **A/B Testing**: Compare model versions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

Mohammad Jumar Singh
- GitHub:https://github.com/jrtyong15
- Email: mohammadjumarsingh15@gmail.com

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing libraries
- Inspired by modern recruitment technology needs
- Built as part of machine learning portfolio development
- Thanks to the Kaggle and Reddit community for helping me to this project
---

⭐ **Star this repository if you found it helpful!** ⭐