import numpy as np
import joblib


def predict_toxicity(sample: dict) -> dict:
    """
    sample: dict with keys matching the training features.
    Returns predicted label and probability.
    """
    bundle   = joblib.load("model/best_model.pkl")
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features"]

    x = np.array([[sample[f] for f in features]])
    x_scaled = scaler.transform(x)

    label = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0]

    return {
        "prediction":   "Toxic" if label == 1 else "Non-Toxic",
        "probability":  {"Non-Toxic": round(proba[0], 4),
                         "Toxic":     round(proba[1], 4)},
    }


if __name__ == "__main__":
    sample = {
        "particle_size_nm":    25.0,
        "zeta_potential_mV":  -30.0,
        "surface_area_m2g":   200.0,
        "hydrophobicity_index": 0.6,
        "charge_density":       2.5,
        "molecular_weight":  5000.0,
        "core_material_encoded": 2,   # TiO2
        "coating_encoded":       1,   # PEG
    }
    result = predict_toxicity(sample)
    print(f"Prediction : {result['prediction']}")
    print(f"Probability: {result['probability']}")