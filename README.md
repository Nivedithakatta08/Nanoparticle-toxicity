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
