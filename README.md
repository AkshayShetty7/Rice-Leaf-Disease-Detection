# 🌾 Rice Leaf Disease Detection System

An AI-powered web application for detecting rice leaf diseases using deep learning. Built with **Streamlit** and **TensorFlow**, it helps farmers and experts identify diseases quickly and accurately.

---


## 🎯 Overview

This application uses a **CNN model** trained on rice leaf images to detect **five major rice diseases** and healthy leaves.

**Key Benefits:**

* Instant image-based disease detection
* Confidence scores for predictions
* Treatment recommendations
* Educational content on symptoms and prevention

---

## ✨ Features

* **AI-Powered Detection**: CNN-based disease classification
* **Real-time Analysis**: Instant results with confidence scores
* **Web Interface**: Streamlit-based
* **Disease Guide**: Symptoms and treatment plans included

---

## 📁 Project Structure

```
rice_leaf_disease_app/
│
├── app.py                  # Main Streamlit application
├── prediction.py           # Model loading & prediction logic
├── rice_disease_model_2.h5 # Trained CNN model 
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── training.ipynb          # Model training notebooks 
   
  

```

---

## 🚀 Installation

### Prerequisites

* Python 3.8+
* pip
* 4GB+ RAM

### Steps

```bash
# Clone repo
git clone https://github.com/AkshayShetty7/Rice-Leaf-Disease-Detection
cd rice_leaf_disease_app

# Create virtual environment (optional)
python -m venv env
source env/bin/activate   # macOS/Linux
env\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Access at: [http://localhost:8501](http://localhost:8501)

---

## 💻 Usage

1. Upload a rice leaf image (PNG/JPG/JPEG)
2. Click **"Analyze Rice Leaf"**
3. View prediction, confidence, and treatment suggestions

---

## 🧠 Model Information

* **Architecture**: CNN (Keras/TensorFlow)
* **Input Size**: 224×224×3
* **Output Classes**: 6 (5 diseases + healthy)
* **Accuracy**: \~95–96% on test data

---

## 🦠 Detectable Diseases

* **Bacterial Leaf Blight** (High severity)
* **Brown Spot** (Medium severity)
* **Leaf Blast** (High severity)
* **Leaf Scald** (Medium severity)
* **Sheath Blight** (High severity)
* **Healthy Leaf** ✅

---

## 🤝 Contributing

Contributions are welcome!

1. Fork repo & create feature branch
2. Follow PEP 8 and add docstrings
3. Submit a pull request with description

---

## ⚖️ License

Licensed under the **MIT License**. See [LICENSE](LICENSE).

---

🌾 Made with ❤️ for the farming community — using AI to improve agriculture, one rice leaf at a time.
