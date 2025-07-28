# 🤖 Hand Sign Detection with Real-Time Correction

A mini machine learning project that detects hand signs using MediaPipe, OpenCV, and a trained machine learning model. You can also correct wrong predictions and improve the model.

---

## 🧠 Features

- Collect hand landmark data from webcam or phone camera
- Train a machine learning model using collected data
- Predict real-time hand signs
- Correct predictions manually and save corrections in `corrected_data.csv`

---

## 🛠️ Files Description

| File | Description |
|------|-------------|
| `collect_data.py` | Collect hand landmarks data into `data.csv` |
| `train_model.py` | Train a model using the data collected |
| `model.pkl` | The saved trained model |
| `predict_realtime.py` | Run live camera and predict hand signs |
| `data.csv` | Your labeled dataset |
| `corrected_data.csv` | Stores corrections made during testing (optional) |

---

## 🚀 How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
