# Emotion Detector (Real-Time Inference)

AI-powered facial expression recognition using FastAI + TorchScript.
This repo contains the **inference-only** pipeline for real-time emotion detection from webcam video using a pretrained model.

---

## 🔍 What It Does

* Detects human faces using Haar Cascades (OpenCV) with an average accuracy of 70%.
* Applies preprocessing (CLAHE + blur) for contrast and noise reduction.
* Runs inference using a lightweight TorchScript model exported from FastAI.
* Smooths predictions temporally and spatially for visual stability.

---

## 🧠 Model Source

The model was trained with the FER-2013 dataset and exported as TorchScript.
👉 **Training notebook (Kaggle)**:
[Part 1: Emotion Recognition with FER-2013](https://www.kaggle.com/code/brunoolimpio/emotion-recognition-w-fer-2013)
Set **session options** to run with accelerator **GPU T4 x2** to better performance during model training.

---

## 🚀 Run Locally

### 1. Clone and Setup

```bash
git clone https://github.com/BOlimpio/emotion-detector-realtime.git
cd emotion-detector-realtime
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the model

Place the file `emotion_detector_ts.pt` in the root directory.
You can generate it using the Kaggle notebook or download from your own training pipeline.

### 3. Run the application

```bash
python emotion_detector.py
```

Press **`q`** to quit the webcam window.

---

## 📦 Requirements

Installed automatically with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
fastai==2.7.12
torch==2.0.1
torchvision==0.15.2
numpy==1.23.5
pillow==9.5.0
opencv-python==4.7.0.72
```

---

## 📌 Notes

* This project was created for **educational purposes** to explore how to build and use a custom-trained emotion recognition model end-to-end.
* The implementation prioritizes learning and readability; there may be **more efficient or production-ready alternatives**.
* The model uses **TorchScript** for portability and avoids `pickle` security issues.
* **CLAHE + Gaussian Blur** enhances face contrast and reduces noise, helping improve prediction stability.
* **Temporal and spatial smoothing** buffers reduce jitter in both face tracking and emotion prediction.

---

## 🤝 Contributing

Contributions are welcome!
If you have suggestions, improvements, or want to experiment with different models, preprocessing techniques, or visualizations — feel free to open an issue or submit a **pull request**.

