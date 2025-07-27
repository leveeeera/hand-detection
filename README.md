# 🤖 Hand Sign Detection with Real-Time Correction

A mini machine learning project that detects hand signs using MediaPipe, OpenCV, and a trained machine learning model. You can also correct wrong predictions and improve the model.

---

# Hand Sign Detection

This project implements real-time hand sign detection using MediaPipe for landmark extraction and a RandomForest machine learning model for classification. It supports:

- Collecting hand landmark data via webcam or IP camera
- Training a model on collected data
- Real-time gesture prediction with live video feed

Ideal for beginners learning computer vision and machine learning concepts applied to hand gesture recognition.

---

## 🧠 Features

- Collect hand landmark data from webcam or phone camera
- Train a machine learning model using collected data
- Predict real-time hand signs

---

## 🛠️ Files Description

| File | Description |
|------|-------------|
| `collect_data.py` | Collect hand landmarks data into `data.csv` |
| `train_model.py` | Train a model using the data collected |
| `model.pkl` | The saved trained model |
| `predict_realtime.py` | Run live camera and predict hand signs |
| `data.csv` | Your labeled dataset |

---

## 🚀 How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
