import pandas as pd
import matplotlib.pyplot as plt

sig_path = "/Users/yush/ML/DNN_project/1000_test_sig.parquet"    
bkg_path = "/Users/yush/ML/DNN_project/1000_test_bkg.parquet"

df_sig = pd.read_parquet(sig_path)
pd.set_option("display.max_columns", None)
df_bkg = pd.read_parquet(bkg_path)
pd.set_option("display.max_columns", None)

plt.figure()
for col in df_sig.columns:
    if pd.api.types.is_numeric_dtype(df_sig[col]):
        x_sig = df_sig[col]
    if pd.api.types.is_numeric_dtype(df_bkg[col]):
        x_bkg = df_bkg[col]
    plt.hist(x_sig, bins=50, density=True, alpha=0.5, label="sig")
    plt.hist(x_bkg, bins=50, density=True, alpha=0.5, label="bkg")        
    plt.xlabel(col)
    plt.title(f"{col}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(col+".pdf")
    plt.show()
