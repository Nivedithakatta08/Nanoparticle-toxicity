import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_and_evaluate
from src.predict import predict_toxicity

if __name__ == "__main__":
    print("=" * 60)
    print("   NANOPARTICLE TOXICITY PREDICTOR")
    print("=" * 60)

    results = train_and_evaluate()

    print("\n── Demo Prediction ──")
    sample = {
        "particle_size_nm":      25.0,
        "zeta_potential_mV":    -30.0,
        "surface_area_m2g":     200.0,
        "hydrophobicity_index":   0.6,
        "charge_density":         2.5,
        "molecular_weight":    5000.0,
        "core_material_encoded":  2,
        "coating_encoded":        1,
    }
    result = predict_toxicity(sample)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Probability : {result['probability']}")