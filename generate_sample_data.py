"""
generate_sample_data.py
=======================
Data preparation and format documentation for TFIF-ssTransformer.

Since the original Liuzhou Steel industrial dataset is proprietary, a small
subset of real industrial data is provided in the `data/` directory for code
verification. This script documents the expected data format and can be used
to extract subsets from the full dataset if available.

Expected data files in `data/`:
  - ss_did3_data_all.csv          (time-series data, N rows × 20 columns)
  - feo_ss_3_dataset_train_len9.pth  (training set TensorDataset)
  - feo_ss_3_dataset_val_len9.pth    (validation set TensorDataset)

CSV format (20 columns):
  - First 17 columns: process parameters (standardized):
      temp_huan1, temp_21#bellow_sul, temp_20#bellow_sul,
      temp_24#bellow_nor, temp_23#bellow_nor, temp_22#bellow_nor,
      temp_21#bellow_nor, temp_20#bellow_nor, prequency_fan_2,
      pressure_flue_sul, pressure_flue_nor, temp_flue_nor,
      thickness, hun2_water, hun2_waterq, zhonghe, fankuang
  - Next 2 columns: decoder auxiliary inputs (z-score normalized prior FeO)
  - Last column: FeO target value (z-score normalized)

TensorDataset format:
  - Each sample: (x_ts [9, 19], X_fig [1, 384, 1920], y_ts [1, 1])
  - x_ts: time-series window (seq_len=9, 17 features + 2 prior columns)
  - X_fig: thermal image tensor (384 × 1920 = 128*3 × 640*3)
  - y_ts: FeO label

Usage to extract subsets from the full dataset:
    python generate_sample_data.py --csv_rows 200 --train_samples 50 --val_samples 30

Code verification after data preparation:
    python run_test.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import os
import argparse


def extract_csv_subset(src_csv, dst_csv, n_rows=200):
    """Extract the first n_rows from the source CSV."""
    df = pd.read_csv(src_csv)
    df_subset = df.iloc[:n_rows]
    df_subset.to_csv(dst_csv, index=False)
    print(f"  CSV subset: {df_subset.shape} → {dst_csv}")
    return df_subset


def extract_tensor_subset(src_pth, dst_pth, n_samples=50):
    """Extract the first n_samples from a TensorDataset."""
    ds = torch.load(src_pth, weights_only=False)
    n = min(n_samples, len(ds))
    tensors = [ds[i] for i in range(n)]
    ts_list = [[], [], []]
    for t in tensors:
        for j in range(3):
            ts_list[j].append(t[j])
    subset = TensorDataset(
        torch.stack(ts_list[0]),
        torch.stack(ts_list[1]),
        torch.stack(ts_list[2])
    )
    torch.save(subset, dst_pth)
    print(f"  Tensor subset: {n} samples → {dst_pth}")
    return subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract data subsets for TFIF-ssTransformer")
    parser.add_argument("--csv_rows", type=int, default=200,
                        help="Number of CSV rows to extract")
    parser.add_argument("--train_samples", type=int, default=50,
                        help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=30,
                        help="Number of validation samples")
    parser.add_argument("--source_csv", type=str, default=None,
                        help="Path to source CSV file (full dataset)")
    parser.add_argument("--source_train", type=str, default=None,
                        help="Path to source training TensorDataset (full dataset)")
    parser.add_argument("--source_val", type=str, default=None,
                        help="Path to source validation TensorDataset (full dataset)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TFIF-ssTransformer: Data Subset Extractor")
    print("=" * 60)
    print()
    print("This script extracts small subsets from the full real industrial")
    print("dataset for code verification. The full dataset is proprietary")
    print("and cannot be publicly released.")
    print()

    if args.source_csv and os.path.exists(args.source_csv):
        print("[1/3] Extracting CSV subset...")
        extract_csv_subset(args.source_csv,
                           os.path.join(args.output_dir, "ss_did3_data_all.csv"),
                           args.csv_rows)
    else:
        print("[1/3] Source CSV not specified or not found. Skipping.")

    if args.source_train and os.path.exists(args.source_train):
        print("[2/3] Extracting training set subset...")
        extract_tensor_subset(args.source_train,
                              os.path.join(args.output_dir,
                                           "feo_ss_3_dataset_train_len9.pth"),
                              args.train_samples)
    else:
        print("[2/3] Source training set not specified or not found. Skipping.")

    if args.source_val and os.path.exists(args.source_val):
        print("[3/3] Extracting validation set subset...")
        extract_tensor_subset(args.source_val,
                              os.path.join(args.output_dir,
                                           "feo_ss_3_dataset_val_len9.pth"),
                              args.val_samples)
    else:
        print("[3/3] Source validation set not specified or not found. Skipping.")

    print()
    print("=" * 60)
    print("Data preparation complete.")
    print("Run 'python run_test.py' to verify the code.")
    print("=" * 60)
