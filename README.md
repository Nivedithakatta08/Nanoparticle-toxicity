# Nanoparticle Toxicity Predictor

A supervised machine learning pipeline that classifies nanoparticles as 
toxic or non-toxic based on physicochemical properties. Compares Random 
Forest, SVM, and KNN — Random Forest achieves the highest accuracy at ~87%.

> Note: Currently trained on synthetic data. Accuracy figures reflect 
> learned synthetic patterns, not validated toxicological outcomes. 
> Migration to real experimental data (Enano, NanoSafety Cluster) is planned.

---

## Background

Nanoparticles (1–100 nm) exhibit unique physicochemical behavior that 
makes bulk-material safety data unreliable for predicting their biological 
effects. Smaller particle size exponentially increases surface-area-to-volume 
ratio, which directly correlates with cellular reactivity and toxicity potential.

Lab-based toxicity testing is expensive and slow. Computational approaches — 
particularly ML models trained on physicochemical descriptors — offer a faster, 
lower-cost alternative for early-stage hazard screening.

---

## Input Features

| Feature | Description |
|---|---|
| particle_size_nm | Diameter in nanometers |
| zeta_potential_mV | Surface charge; extremes indicate higher reactivity |
| surface_area_m2g | BET surface area — higher = more biological contact |
| hydrophobicity_index | Water repellence (0–1); affects membrane interaction |
| charge_density | Surface charge per unit area |
| molecular_weight | Particle mass |
| core_material_encoded | Material type: Gold, Silver, TiO₂, ZnO, SiO₂ |
| coating_encoded | Surface coating: None, PEG, Citrate, Amine |

**Target:** Binary — Toxic (1) / Non-Toxic (0)

---

## Models

| Model | Test Accuracy | Cross-Validation |
|---|---|---|
| Random Forest | ~87% | reported |
| SVM (RBF kernel) | ~83% | reported |
| KNN (k=5) | ~79% | reported |

Evaluation includes accuracy, cross-validation score, ROC-AUC, and 
confusion matrices. Random Forest also provides feature importance rankings.

---

## Project Structure

├── src/           # training and prediction logic
├── model/         # serialized model (joblib)
├── data/          # dataset
├── results/       # output plots
├── api/           # API layer
├── templates/     # Flask HTML
├── static/        # CSS/JS
├── main.py        # entry point
└── app.py         # web application

## Setup

```bash
git clone https://github.com/Nivedithakatta08/Nanoparticle-toxicity.git
cd Nanoparticle-toxicity
pip install -r requirements.txt
python main.py
```

## Inference

```python
from src.predict import predict_toxicity

sample = {
    "particle_size_nm": 25.0,
    "zeta_potential_mV": -30.0,
    "surface_area_m2g": 200.0,
    "hydrophobicity_index": 0.6,
    "charge_density": 2.5,
    "molecular_weight": 5000.0,
    "core_material_encoded": 2,
    "coating_encoded": 1,
}

result = predict_toxicity(sample)
# {'prediction': 'Toxic', 'probability': {'Non-Toxic': 0.13, 'Toxic': 0.87}}
```

Web app: https://nanoparticle-toxicity.vercel.app

---

## Stack

Python · Scikit-learn · Pandas · NumPy · Matplotlib · Joblib · Flask

---

## Limitations & Roadmap

The dataset is synthetically generated — the model learns the rules used 
to produce the data, not real toxicity biology. This is the primary 
credibility gap.

Planned:
- Replace synthetic data with Enano / NanoSafety Cluster experimental datasets
- Retrain and re-evaluate all three models
- Add SHAP values for feature-level interpretability
