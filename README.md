# 🩺 Skin Cancer Detection System

An end-to-end Deep Learning web application for detecting whether a skin lesion is **Benign** or **Malignant** using a custom Convolutional Neural Network (CNN) built from scratch.

The system performs real-time image classification through an interactive web interface and provides probability breakdown along with adjustable decision threshold for optimized recall–precision trade-off.

---

## 🚀 Key Highlights

- 🧠 Designed and trained a **custom CNN architecture from scratch**
- 📊 Achieved ~90% ROC-AUC on unseen test data demonstrating strong generalization
- 🎛 Adjustable decision threshold for recall vs precision control
- 🌐 Fully deployed using **Streamlit**
- ☁️ Model hosted externally and auto-downloaded at runtime
- 📈 Includes probability breakdown for prediction confidence

---

## 🛠 Tech Stack

- **TensorFlow / Keras** – Model development
- **NumPy** – Data processing
- **Pillow** – Image preprocessing
- **Streamlit** – Web application framework
- **Google Drive + gdown** – Model hosting

---

## 🧠 Model Details

- Input Size: 224x224 RGB
- Architecture: Multi-layer CNN (Conv2D → MaxPooling → Dense)
- Regularization: Dropout for overfitting control
- Activation: ReLU
- Output: Sigmoid (Binary Classification)
- Loss Function: Binary Crossentropy
- Evaluation Metrics:
  - Accuracy: ~88–90%
  - ROC-AUC: ~0.90

---

## 📊 Features

✔ Real-time image upload  
✔ Adjustable decision threshold  
✔ Probability breakdown  
✔ Clean UI with medical disclaimer  
✔ Model auto-download from Google Drive  
✔ Deployment-ready structure  

---

## 🚀 How It Works

1. User uploads a skin lesion image.
2. Image is resized to 224x224.
3. Model predicts probability of malignancy.
4. Decision threshold determines classification.
5. Results and confidence are displayed.

---

> Note: Model file (.h5) is hosted externally to avoid GitHub size limits.

> 👉 [Download Model Here] (https://drive.google.com/uc?id=1XyRpIGwXgfcyA_ERsqmlGdMIqUeig3dH)
>
> ## 🌐 Live Demo
👉 https://skin-cancer-detection-app-mxnpjr9nzszuwaiaims8yu.streamlit.app

## 🔗 Dataset Link
👉 https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset


