# 🧠 Few-Shot Image Classifier with Flask

This project is a simple web application for **few-shot image classification** using a Prototypical Network-based encoder. It predicts whether an image belongs to **Good Responders** or **Bad Responders** using a small support set of labeled images.

The app is built with **PyTorch** for the model and **Flask** for the web interface.

---

## 🔧 Features

- Custom CNN-based encoder trained for few-shot learning
- Prototype computation from a small support set
- Flask-based web app for uploading and classifying images
- Works with two classes: `class_0` → **Good Responders**, `class_1` → **Bad Responders**

---



## 📦 Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/Amine136/Few-Shot-Learning-FSL-pred-anti-VEGF.git
cd few-shot-flask-app
```

The support/ folder contain subfolders for each class (e.g., class_0, class_1) with example images inside.

```bash

support/
├── class_0/   # Good Responders
│   ├── img1.jpg
│   └── ...
└── class_1/   # Bad Responders
    ├── img2.jpg
    └── ...
```
2. Run the Flask app
```bash   
python app.py
```
The app will start on http://127.0.0.1:5000/.

🌐 Web Interface
Open your browser and go to:

Upload an image and get a prediction:
✅ "Good responders"
❌ "Bad responders"

🧠 Model Details
The encoder is a 4-layer convolutional neural network trained for few-shot learning using Prototypical Networks. The .pth file is a pre-trained PyTorch model.

