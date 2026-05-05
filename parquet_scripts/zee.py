import numpy as np
import pandas as pd

def selection(df):
    # Define individual selection criteria
    mass_criteria = (df["mass"] > 80) & (df["mass"] < 100)
    tag_pt_criteria = df["tag_pt"] > 40
    probe_pt_criteria = df["probe_pt"] > 22
    eta_criteria = np.abs(df["probe_ScEta"]) < 2.5
    phi_criteria = np.abs(df["probe_phi"]) < np.pi
    tag_mvaID_criteria = df["tag_mvaID"] > 0.0
    
    # Combine all criteria into a single mask
    mask = (
        mass_criteria &
        tag_pt_criteria &
        probe_pt_criteria &
        eta_criteria &
        phi_criteria &
        tag_mvaID_criteria
    )
    
    # Apply mask to filter DataFrame
    return df[mask]