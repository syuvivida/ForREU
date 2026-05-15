import argparse
import pandas as pd
import re


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
            description='Read parquet files and plots')
    parser.add_argument("-i", "--input",
        dest="file_path", type=str, default="/Users/yush/Downloads/hepmass/1000_test.csv", help="path of a single csv file")


    options = parser.parse_args()

    df = pd.read_csv(options.file_path)
    first_col_values = df[df.columns[0]]
    print(first_col_values)
    df_signal = df[first_col_values==1]
    df_background = df[first_col_values == 0]
    result=re.sub(r"/.*/","",options.file_path).replace(".csv","")

    df_signal.to_parquet(result+"_sig.parquet", engine="pyarrow", index=False, compression="snappy")
    df_background.to_parquet(result+"_bkg.parquet", engine="pyarrow", index=False, compression="snappy")

