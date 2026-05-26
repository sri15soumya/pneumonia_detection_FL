# Pneumonia Detection Using Federated Learning

A privacy-preserving chest X-ray classifier that detects pneumonia without sharing raw patient data across hospitals — built using Federated Learning and CNNs.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-red?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow?style=flat-square)

---

## What This Project Does

Two simulated hospital clients each train a CNN on their own private chest X-ray dataset. Instead of sharing patient images, they only send their **model weights** to a central server, which merges them into one global model. That global model is sent back to each hospital for final fine-tuning.

We compare **3 different ways** of merging the models to see which gives the best and fairest result.

---

## The 3 Aggregation Strategies

| Model | Strategy | How it works |
|-------|----------|--------------|
| Model 1 | Accuracy-Weighted | Clients with better accuracy get more influence |
| Model 2 | Sample-Weighted (FedAvg) | Clients with more data get more influence |
| Model 3 | Uniform Averaging | Both clients treated equally |

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Binary classification: **Normal vs Pneumonia**
- ~5,800 chest X-ray images total
- Split between two simulated clients:
  - **Client A** — 5,216 training images, 624 test images
  - **Client B** — 4,684 training images, 1,172 test images

---

## Model Architecture

Same CNN used across all three models for a fair comparison:

```
Input (224 × 224 grayscale)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool + Dropout(0.3)
→ Conv2D(128) + MaxPool + Dropout(0.4)
→ Dense(128) + Dropout(0.5)
→ Output: Sigmoid (Normal / Pneumonia)
```

Trained for 20 epochs per client, fine-tuned for 20 epochs after federation.

---

## Results

### Before Federation (baseline — independent training)

| Client | Own Dataset Accuracy |
|--------|----------------------|
| Client A | 76.44% |
| Client B | 93.09% |

### After Federation (own dataset performance)

| Model | Strategy | Client A Accuracy | Client B Accuracy |
|-------|----------|-------------------|-------------------|
| Model 1 | Accuracy-Weighted | **88.30%** | **96.16%** |
| Model 2 | Sample-Weighted | 89.10% | 96.50% |
| Model 3 | Uniform | 86.38% | 96.84% |

**Model 1 wins overall** — it's the most balanced across both clients. Model 2 peaks higher but favours the larger client. Model 3 crashed to 38% accuracy before fine-tuning, making it unreliable at scale.

---

## Project Structure

```
pneumonia_detection_FL/
├── Base_model_A.ipynb      # Client A standalone training
├── Base_model_B.ipynb      # Client B standalone training
├── Model1.ipynb            # Federated: Accuracy-Weighted
├── Model2.ipynb            # Federated: Sample-Weighted (FedAvg)
├── Model3.ipynb            # Federated: Uniform Averaging
└── README.md
```

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/sri15soumya/pneumonia_detection_FL.git
cd pneumonia_detection_FL
```

**2. Install dependencies**
```bash
pip install tensorflow keras numpy matplotlib scikit-learn jupyter
```

**3. Download the dataset**

Get it from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in your working directory.

**4. Run notebooks in this order**
```
Base_model_A.ipynb  →  Base_model_B.ipynb  →  Model1 / Model2 / Model3
```


