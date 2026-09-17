import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no windows, saves to file
 
import pandas as pd
import matplotlib.pyplot as plt
import os
 
sig_path = "/Users/yush/ML/data/NMSSM_X600_Y150.parquet"
bkg_path = "/Users/yush/ML/data/DDQCDGJets_Rescaled.parquet"
 
df_sig = pd.read_parquet(sig_path)
df_bkg = pd.read_parquet(bkg_path)
pd.set_option("display.max_columns", None)
 
sig_cols = df_sig.columns.tolist()
bkg_cols = df_bkg.columns.tolist()
 
# Columns in both files
common  = [c for c in sig_cols if c in bkg_cols]
only_sig = [c for c in sig_cols if c not in bkg_cols]
only_bkg = [c for c in bkg_cols if c not in sig_cols]
 
print(f"Columns in BOTH  ({len(common)}):   {common}")
print(f"Only in signal   ({len(only_sig)}): {only_sig}")
print(f"Only in bkg      ({len(only_bkg)}): {only_bkg}")
 
plotDir = "plots"
os.makedirs(plotDir, exist_ok=True)
 
for col in common:
    if not pd.api.types.is_numeric_dtype(df_sig[col]):
        continue  # skip non-numeric columns

    # Cast booleans to int so numpy can compute histogram ranges
    x_sig = df_sig[col].astype(int) if df_sig[col].dtype == bool else df_sig[col]
    x_bkg = df_bkg[col].astype(int) if df_bkg[col].dtype == bool else df_bkg[col]
 
    # Skip columns where either file has all NaN or a single unique value
    if x_sig.dropna().nunique() < 2 or x_bkg.dropna().nunique() < 2:
        print(f"Skipping {col} (insufficient unique values)")
        continue
 
    fig, ax = plt.subplots()  # fresh figure per column
    ax.hist(x_sig, bins=50, density=True, alpha=0.5, label="sig")
    ax.hist(x_bkg, bins=50, density=True, alpha=0.5, label="bkg")
    ax.set_xlabel(col)
    ax.set_title(col)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plotDir, f"{col}.pdf"))
    plt.close(fig)  # free memory immediately
    print(f"Saved: {col}.pdf")
 
print("Done.")
