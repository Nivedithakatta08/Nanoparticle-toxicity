import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib


def predict_toxicity(sample: dict) -> dict:
    """
    sample: dict with keys matching the training features.
    Returns predicted label and probability.
    """
    model  = joblib.load("model/best_model_rf.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le     = joblib.load("model/label_encoder.pkl")

    features = [
        'coresize', 'hydrosize', 'surfcharge', 'surfarea',
        'Ec', 'Expotime', 'dosage', 'e', 'NOxygen', 'NPs_encoded'
    ]

    x = pd.DataFrame([[sample[f] for f in features]], columns=features)
    x_scaled = scaler.transform(x)

    label = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0]

    return {
        "prediction":  "Toxic" if label == 1 else "Non-Toxic",
        "probability": {"Non-Toxic": round(proba[0], 4),
                        "Toxic":     round(proba[1], 4)},
    }


if __name__ == "__main__":
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
        "NPs_encoded": 4,   # ZnO encoded value
    }
    result = predict_toxicity(sample)
    print(f"Prediction : {result['prediction']}")
    print(f"Probability: {result['probability']}")