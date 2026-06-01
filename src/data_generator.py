import numpy as np
import pandas as pd

def generate_nanoparticle_data(n_samples=500, random_state=42):
    np.random.seed(random_state)

    data = {
        "particle_size_nm": np.random.uniform(1, 100, n_samples),
        "zeta_potential_mV": np.random.uniform(-60, 60, n_samples),
        "surface_area_m2g": np.random.uniform(10, 500, n_samples),
        "hydrophobicity_index": np.random.uniform(0, 1, n_samples),
        "charge_density": np.random.uniform(-5, 5, n_samples),
        "molecular_weight": np.random.uniform(100, 10000, n_samples),
        "core_material_encoded": np.random.randint(0, 5, n_samples),  # 0=Gold,1=Silver,2=TiO2,3=ZnO,4=SiO2
        "coating_encoded": np.random.randint(0, 4, n_samples),        # 0=None,1=PEG,2=Citrate,3=Amine
    }

    df = pd.DataFrame(data)

    # Simulate toxicity label based on physicochemical rules
    toxicity_score = (
        0.03 * df["particle_size_nm"] +
        0.02 * np.abs(df["zeta_potential_mV"]) -
        0.005 * df["surface_area_m2g"] +
        2.0 * df["hydrophobicity_index"] +
        1.5 * np.abs(df["charge_density"]) +
        0.0001 * df["molecular_weight"] +
        df["core_material_encoded"] * 0.8 +
        np.random.normal(0, 1.5, n_samples)
    )

    df["toxic"] = (toxicity_score > np.median(toxicity_score)).astype(int)
    return df