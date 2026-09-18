import re
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np



# 1. Read metadata schema (instantaneous, zero data loaded into RAM)
print("step 1")
schema_names = pq.ParquetFile("../data/NMSSM_X600_Y150.parquet").schema.names
schema =  pq.ParquetFile("../data/NMSSM_X600_Y150.parquet").schema_arrow
#schema_names = pq.ParquetFile("../data/GGJets_MGG-80_Rescaled.parquet").schema.names

# 2. Match column names containing 'photon', 'jet', or 'gamma' (case-insensitive)
print("step 2")
include_pattern = re.compile(r"(^Res|^lead|^sublead)", re.IGNORECASE)
exclude_pattern = re.compile(r"(^nonRes|^Res_DNNpair|^fatjet|^jet|raw|genPart|MET|SCEta|charge|_eta|idx|hFlav|orig|genFlav|nano|PNet|gen|has|Res_M_X|Res_HHbbggCandidate_mass|WP80|WP90|^Res_chi_t)", re.IGNORECASE)

bool_cols = [field.name for field in schema if str(field.type) == "bool"]
#print("Boolean columns:", bool_cols)

features = [                                                                                                                   
    col for col in schema_names                                                                                                
    if include_pattern.search(col) and col not in ["weight", "target"] and not exclude_pattern.search(col) and col not in bool_cols                                                                                                                           
]       

print(f"Discovered {len(features)} physics features: {features}")


