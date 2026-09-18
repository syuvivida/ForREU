import re
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np


Luminosity=108.96 #/fb
sigXSec = 1 #fb
bkgXSec = 87.51*1000 # fb

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
print("Boolean columns:", bool_cols)

"""
features = [
    col for col in schema_names 
    if include_pattern.search(col) and col not in ["weight", "target"] and not exclude_pattern.search(col) and col not in bool_cols
]
"""

features = ["Res_pholead_PtOverM", "Res_lead_bjet_btagUParTAK4B", "Res_sublead_bjet_btagUParTAK4B", "Res_phosublead_PtOverM", "Res_DeltaR_jg_min","Res_lead_bjet_btagUParTAK4CvL", "Res_sublead_bjet_btagUParTAK4CvL", "lead_mvaID", "sublead_mvaID"]

"""
features = ["Res_pholead_PtOverM", "Res_dijet_pt", "Res_lead_bjet_btagUParTAK4B", "lead_pt", "Res_sublead_bjet_btagUParTAK4B", "Res_lead_bjet_pt", "Res_HHbbggCandidate_pt", "Res_phosublead_PtOverM", "sublead_pt", "sublead_pfRelIso03_all_quadratic", "Res_DeltaR_jg_min", "Res_lead_bjet_btagUParTAK4CvL", "sublead_cutBased", "sublead_mvaID", "lead_pfRelIso03_all_quadratic", "lead_sieie", "lead_mvaID", "lead_cutBased", "lead_trkSumPtSolidConeDR04", "Res_sublead_bjet_btagUParTAK4CvL", "lead_energyErr", "sublead_energyErr", "sublead_ecalPFClusterIso", "Res_sublead_bjet_pt", "Res_dijet_mass_DNNreg", "Res_DeltaR_j1g2", "sublead_trkSumPtSolidConeDR04", "sublead_sieie", "Res_sublead_bjet_mass", "Res_DeltaR_j1g1", "sublead_s4", "Res_dijet_mass", "Res_CosThetaStar_gg", "sublead_rho_smear", "sublead_pfRelIso03_chg_quadratic", "Res_SecondJet_PtOverM", "lead_rho_smear", "sublead_trkSumPtHollowConeDR03", "lead_ecalPFClusterIso", "Res_FirstJet_PtOverM", "lead_sipip", "lead_vidNestedWPBitmap", "Res_CosThetaStar_CS", "lead_s4", "sublead_sipip", "sublead_phi", "sublead_sieip", "lead_hcalPFClusterIso", "Res_DeltaR_j2g2", "sublead_hoe_Tower"]
"""

print(f"Discovered {len(features)} physics features: {features}")

# 3. Load only filtered features + weight column
print("step 3")
columns_to_load = features + ["weight"]

filters = [
    ("lead_mvaID",    ">", -0.7),
    ("sublead_mvaID", ">", -0.7),
    ("lead_pt",       ">", 35),
    ("lead_eta",      "<", 2.5),
    ("lead_eta",      ">",-2.5),
    ("sublead_pt",    ">", 20),
    ("sublead_eta",   "<", 2.5),
    ("sublead_eta",   ">",-2.5)
]


df_signal = pd.read_parquet("../data/NMSSM_X600_Y150.parquet", columns=columns_to_load, filters=filters)
df_bkg1 = pd.read_parquet("../data/GGJets_MGG-80_Rescaled.parquet", columns=columns_to_load, filters=filters)
df_bkg2 = pd.read_parquet("../data/DDQCDGJets_Rescaled.parquet", columns=columns_to_load, filters=filters)

for df in [df_signal, df_bkg1, df_bkg2]:
    df[features] = df[features].replace(-999, np.nan)

df_signal["weight"] *= Luminosity * sigXSec
df_bkg1["weight"]   *= Luminosity * bkgXSec


# Add target labels
df_signal["target"] = 1
df_bkg1["target"] = 0
df_bkg2["target"] = 0


df_bkg = pd.concat([df_bkg1, df_bkg2], ignore_index=True)
df_all = pd.concat([df_signal, df_bkg], ignore_index=True)


# 4. Extract features, targets, and weights
print("step 4")
X = df_all[features]
y = df_all["target"]
weights = df_all["weight"]

# 5. Train/Test Split
print("step 5")
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42, stratify=y
)


# 6. Weight Normalization + abs-weight fix for NLO negative weights
w_train = w_train.copy()   # avoid SettingWithCopyWarning

# Diagnostic: how many background events have negative weights?
neg_frac = (w_train[y_train == 0] < 0).mean()
print(f"Negative weight fraction in bkg train set: {neg_frac:.3%}")

# Take absolute values — standard practice for BDTs with NLO samples
#w_train_abs = np.clip(w_train, 0.0, None)
#w_test_abs = np.clip(w_test,0.0,None)

w_train_abs = w_train.abs()
w_test_abs = w_test.abs()
w_train_abs[y_train==1] *= len(w_train_abs[y_train==1])/w_train_abs[y_train==1].sum()
w_train_abs[y_train==0] *= len(w_train_abs[y_train==0])/w_train_abs[y_train==0].sum()
#w_train_abs[y_train==1] *= Luminosity * sigXSec/w_train_abs[y_train==1].sum()
#w_train_abs[y_train==0] *= Luminosity * bkgXSec/w_train_abs[y_train==0].sum()

print(f"length of signal w_train_abs={len(w_train_abs[y_train==1])}")
print(f"sum of signal weights = {w_train_abs[y_train==1].sum()}")
print(f"length of background w_train_abs={len(w_train_abs[y_train==0])}")
print(f"sum of background weights = {w_train_abs[y_train==0].sum()}")



# Then use scale_pos_weight to tell XGBoost about the class imbalance

n_sig = (y_train == 1).sum()
n_bkg = (y_train == 0).sum()
spw = n_bkg / n_sig
print(f"scale_pos_weight = {spw:.2f}")



import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# 1. Initialize the BDT model
bdt_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,                # Shallow depth (3-5) prevents overtraining on MC statistical fluctuations
    learning_rate=0.05,
    subsample=0.8,              # Train on 80% of events per tree
    colsample_bytree=0.8,       # Train on 80% of features per tree
    scale_pos_weight=spw, 
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=30,   # Stop training when validation AUC plateaus
    random_state=42
)

# 2. Fit the model using weighted clean features
bdt_model.fit(
    X_train, y_train,
    sample_weight=w_train_abs,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    sample_weight_eval_set=[w_train_abs, w_test_abs],
    verbose=50                  # Print progress every 50 trees
)

# 3. Predict probabilities on train and test sets
y_pred_train = bdt_model.predict_proba(X_train)[:, 1]
y_pred_test = bdt_model.predict_proba(X_test)[:, 1]

# 4. Calculate ROC AUC Scores
auc_train = roc_auc_score(y_train, y_pred_train, sample_weight=w_train_abs)
auc_test = roc_auc_score(y_test, y_pred_test, sample_weight=w_test_abs)

print(f"\nTraining ROC AUC Score: {auc_train:.4f}")
print(f"Testing  ROC AUC Score: {auc_test:.4f}")
if abs(auc_train - auc_test) > 0.03:
    print("Warning: Potential overtraining detected (|AUC_train - AUC_test| > 0.03).")
else:
    print("Model generalization looks good!")

print("Best iteration:", bdt_model.best_iteration)
print("Number of trees:", bdt_model.n_estimators)

# Check 1: NaN rates
print("=== Signal NaN rates ===")
print(df_signal[features].isna().mean().sort_values(ascending=False).head(10))

print("\n=== Background NaN rates ===")
print(df_bkg1[features].isna().mean().sort_values(ascending=False).head(10))
print(df_bkg2[features].isna().mean().sort_values(ascending=False).head(10))

# Check 2: Weight sanity
print("\n=== Weight diagnostics ===")
print(f"Signal weights:     min={w_train_abs[y_train==1].min():.7f}, max={w_train_abs[y_train==1].max():.7f}, sum={w_train_abs[y_train==1].sum():.7f}")
print(f"Background weights: min={w_train_abs[y_train==0].min():.7f}, max={w_train_abs[y_train==0].max():.7f}, sum={w_train_abs[y_train==0].sum():.7f}")
print(f"Signal events: {(y_train==1).sum()},  Background events: {(y_train==0).sum()}")

    
# 5. Diagnostic Plot 1: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_test, sample_weight=w_test_abs)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Test Set (AUC = {auc_test:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlabel("False Positive Rate (Background Efficiency)", fontsize=11)
plt.ylabel("True Positive Rate (Signal Efficiency)", fontsize=11)
plt.title("XGBoost ROC Curve", fontsize=12)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("AUC.png", bbox_inches="tight")
plt.show()

# 6. Diagnostic Plot 2: Overtraining Check (BDT Score Distribution)
bins = np.linspace(0, 1, 30)

plt.figure(figsize=(8, 6))

# Background distributions
plt.hist(y_pred_train[y_train == 0], bins=bins, weights=w_train_abs[y_train == 0], 
         density=True, alpha=0.4, color="blue", label="Bkg (Train)")
counts_bkg, _ = np.histogram(y_pred_test[y_test == 0], bins=bins, weights=w_test_abs[y_test == 0], density=True)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
plt.errorbar(bin_centers, counts_bkg, fmt="o", color="blue", label="Bkg (Test)")

# Signal distributions
plt.hist(y_pred_train[y_train == 1], bins=bins, weights=w_train_abs[y_train == 1], 
         density=True, alpha=0.4, color="red", label="Signal (Train)")
counts_sig, _ = np.histogram(y_pred_test[y_test == 1], bins=bins, weights=w_test_abs[y_test == 1], density=True)
plt.errorbar(bin_centers, counts_sig, fmt="o", color="red", label="Signal (Test)")

plt.xlabel("BDT Probability Output", fontsize=11)
plt.ylabel("Normalized Yield", fontsize=11)
plt.title("BDT Output Score & Overtraining Check", fontsize=12)
plt.yscale("log")
plt.legend(loc="upper center")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sig_vs_bkg.png", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt


check_features = ["Res_pholead_PtOverM", "Res_lead_bjet_btagUParTAK4B", "Res_dijet_pt", "Res_sublead_bjet_btagUParTAK4B"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feat in zip(axes.flatten(), check_features):
    sig_vals = df_signal[feat].dropna()
    bkg_vals = df_bkg[feat].dropna()
    ax.hist(sig_vals, bins=50, density=True, alpha=0.5, color="red",  label="Signal")
    ax.hist(bkg_vals, bins=50, density=True, alpha=0.5, color="blue", label="Background")
    ax.set_title(feat)
    ax.legend()
plt.tight_layout()
plt.savefig("feature.png", bbox_inches="tight")
plt.show()

# Get feature importances
fi = pd.Series(bdt_model.feature_importances_, index=features).sort_values(ascending=True)

# Plot
plt.figure(figsize=(8, 12))
plt.barh(fi.index, fi.values, color="steelblue")
plt.xlabel("Feature Importance (weight)", fontsize=11)
plt.title("XGBoost Feature Ranking", fontsize=12)
plt.tight_layout()
plt.savefig("feature_weight.png", bbox_inches="tight")
plt.show()

# 'gain'   — average improvement in loss when this feature is used (most informative)
# 'cover'  — average number of events affected by splits on this feature
# 'weight' — number of times the feature appears in trees (default)

fi = pd.Series(bdt_model.get_booster().get_score(importance_type="gain"),
               name="importance"
               ).sort_values(ascending=False)


plt.figure(figsize=(8, 12))
plt.barh(fi.index, fi.values, color="steelblue")
plt.xlabel("Feature Importance (gain)", fontsize=11)
plt.title("XGBoost Feature Ranking", fontsize=12)
plt.tight_layout()
plt.savefig("feature_gain.png", bbox_inches="tight")
plt.show()

with open("top50_features.txt", "w") as f:
    for feature, importance in fi.head(50).items():
        f.write(f"{feature}\t{importance:.6f}\n")

