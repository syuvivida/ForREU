import argparse
import pandas as pd
import glob
import matplotlib.pyplot as plt


from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def _read_parquet_file(path: str) -> pd.DataFrame:
    """Helper to read a single Parquet file."""
    return pd.read_parquet(path)


def read_parquet_files(patterns: list[str], max_workers: int | None = None, type = "data", max_read_files: int | None = None) -> pd.DataFrame:
    """
    Parallel reading of multiple Parquet files matching given glob patterns.
    Returns a concatenated DataFrame, with a progress bar.
    """
    # Expand all glob patterns into real file paths
    files = [f for pat in patterns for f in glob.glob(pat)]     # Remove this limitation ... for testing purposes only
    #### only for testing, to speed up the process, we will read only a few files. Remove this line later (randomly select 100 files if there are more than 100)
    if max_read_files is not None and type == "data" and len(files) > max_read_files:
        files = np.random.choice(files, size=max_read_files, replace=False).tolist()
    elif max_read_files is not None and type == "MC" and len(files) > max_read_files:
        files = np.random.choice(files, size=max_read_files, replace=False).tolist()
    
    if not files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    # Read in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for df in tqdm(
            executor.map(_read_parquet_file, files),
            total=len(files),
            desc="Reading parquet files",
        ):
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
            description='Read parquet files and plots')
    parser.add_argument("-i", "--input",
        dest="file_path", type=list[str], default=["/Users/yush/OneFlow/TnP/DY_postEE_2022/nominal/"], help="path of directory of parquet files")

    parser.add_argument("-v", "--variable",
        dest="var", type=str, default="tag_pt", help="variables to plot")

    options = parser.parse_args()
    

#files = glob.glob(options.file_path+"/*.parquet")
#df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = read_parquet_files(options.file_path, 1, "data", 1)


pd.set_option("display.max_columns", None)

for col in df.columns:
    print(col)

column_to_plot = options.var
if column_to_plot not in df.columns:
    raise ValueError(f"Column '{column_to_plot}' not found in dataset")

plt.figure()

if pd.api.types.is_numeric_dtype(df[column_to_plot]):
    df[column_to_plot].plot(kind="hist", bins=50)
    plt.ylabel("A. U.")

plt.title(f"{column_to_plot}")
plt.xlabel(column_to_plot)
plt.tight_layout()
plt.show()
