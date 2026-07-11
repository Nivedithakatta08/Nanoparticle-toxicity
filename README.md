# Nanoparticle Toxicity Predictor

A supervised ML pipeline that classifies nanoparticles as toxic or non-toxic
based on physicochemical properties. Compares Random Forest, SVM, and KNN.
Random Forest achieves the highest accuracy at 98.87%.

---

## Background

Nanoparticles (1–100 nm) exhibit unique physicochemical behavior that makes
their toxicity difficult to predict from bulk material properties alone.
Key drivers of toxicity include particle size, surface charge, surface area,
and core material. The smaller particles with larger surface-to-volume ratios
tend to be significantly more reactive with biological tissue.

Computational prediction offers a faster, lower-cost alternative to in vitro
and in vivo testing, and aligns with established QSAR (Quantitative
Structure-Activity Relationship) approaches used in nanotoxicology.

---

## Dataset

Real experimental data sourced from Kaggle (UCI Machine Learning —
Nanoparticle Toxicity Dataset). 881 records, zero missing values.

| Column     | Description                                      |
|------------|--------------------------------------------------|
| coresize   | Core diameter (nm)                               |
| hydrosize  | Hydrodynamic size (nm)                           |
| surfcharge | Zeta potential (mV)                              |
| surfarea   | BET surface area (m²/g)                          |
| Ec         | Electronegativity of core metal                  |
| Expotime   | Exposure time (hours)                            |
| dosage     | Concentration (µg/mL)                            |
| e          | Electron configuration factor                    |
| NOxygen    | Number of oxygen atoms in formula                |
| NPs        | Particle type: ZnO, TiO2, CuO, Al2O3, Fe2O3     |

**Target:** Binary — Toxic / Non-Toxic (476 Toxic, 405 Non-Toxic)

---

## Models

| Model            | Test Accuracy | ROC-AUC | Cross-val (5-fold) |
|------------------|---------------|---------|-------------------|
| Random Forest    | 98.87%        | 0.9995  | 96.45%            |
| KNN (k=5)        | 94.92%        | 0.9915  | 90.06%            |
| SVM (RBF kernel) | 92.66%        | 0.9856  | 91.76%            |

Random Forest selected as final model. Confusion matrix: [[79, 2], [0, 96]]
on 177 test samples — 2 misclassifications.

---

## Project Structure

Nanoparticle-toxicity/
│
├── dataset/
│   └── nanotox_dataset.csv
│
├── model/
│   ├── best_model_rf.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── main.py
├── app.py
└── README.md
---

## Setup

```bash
git clone https://github.com/Nivedithakatta08/Nanoparticle-toxicity.git
cd Nanoparticle-toxicity
pip install -r requirements.txt
python main.py
```

---

## Inference

```python
from src.predict import predict_toxicity

sample = {
    "coresize":    45.3,
    "hydrosize":   327.0,
    "surfcharge":  -9.3,
    "surfarea":    200.0,
    "Ec":          1.65,
    "Expotime":    24,
    "dosage":      25.0,
    "e":           1.65,
    "NOxygen":     1,
    "NPs_encoded": 4,   # ZnO
}

result = predict_toxicity(sample)
# {'prediction': 'Toxic', 'probability': {'Non-Toxic': 0.1883, 'Toxic': 0.8117}}
```

---

## Stack

Python · Scikit-learn · Pandas · NumPy · Joblib · Flask
