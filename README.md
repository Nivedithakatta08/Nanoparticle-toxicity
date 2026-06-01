# 🧬 Nanoparticle Toxicity Predictor

A machine learning project that predicts the toxicity of nanoparticles based on their physicochemical properties. Three classification algorithms are compared (Random Forest, SVM, and KNN) with Random Forest achieving the highest accuracy of **87%**.

---

## 📌 Project Overview

Nanoparticles are increasingly used in medicine, electronics, and manufacturing. Predicting their toxicity early using computational methods can reduce the need for expensive and time-consuming lab experiments.

This project builds a supervised ML pipeline that:
- Generates a dataset of nanoparticles with physicochemical features
- Trains and compares three ML classifiers
- Evaluates performance using accuracy, cross-validation, ROC-AUC, and confusion matrices
- Saves the best model for inference on new samples

---



## 🔬 Features Used

| Feature | Description |
|---|---|
| `particle_size_nm` | Particle size in nanometers |
| `zeta_potential_mV` | Surface charge in millivolts |
| `surface_area_m2g` | BET surface area (m²/g) |
| `hydrophobicity_index` | Hydrophobicity score (0–1) |
| `charge_density` | Surface charge density |
| `molecular_weight` | Molecular weight of the particle |
| `core_material_encoded` | Core material (Gold, Silver, TiO₂, ZnO, SiO₂) |
| `coating_encoded` | Surface coating (None, PEG, Citrate, Amine) |

---

## 🤖 Models Compared

| Model | Test Accuracy |
|---|---|
| ✅ Random Forest | **~87%** |
| SVM (RBF kernel) | ~83% |
| KNN (k=5) | ~79% |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Nanoparticle-toxicity.git
cd Nanoparticle-toxicity
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python main.py
```

---

## 📊 Results

After running, the following are generated automatically:

- **Accuracy Comparison** — Bar chart of test vs cross-validation accuracy
- **Confusion Matrices** — Side-by-side for all three models
- **ROC Curves** — AUC comparison across models
- **Feature Importance** — Top physicochemical predictors (Random Forest)

---

## 🔍 Run a Custom Prediction

```python
from src.predict import predict_toxicity

sample = {
    "particle_size_nm":      25.0,
    "zeta_potential_mV":    -30.0,
    "surface_area_m2g":     200.0,
    "hydrophobicity_index":   0.6,
    "charge_density":         2.5,
    "molecular_weight":    5000.0,
    "core_material_encoded":  2,   # TiO2
    "coating_encoded":         1,  # PEG
}

result = predict_toxicity(sample)
print(result)
# {'prediction': 'Toxic', 'probability': {'Non-Toxic': 0.13, 'Toxic': 0.87}}
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scikit-learn** — ML models and evaluation
- **Pandas & NumPy** — Data manipulation
- **Matplotlib** — Visualization
- **Joblib** — Model serialization

---

---

## 👩‍💻 Author

**Niveditha** — [GitHub](https://github.com/Nivedithakatta08)