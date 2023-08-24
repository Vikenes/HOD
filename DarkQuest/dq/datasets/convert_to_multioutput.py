from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import random


def pivot_k(df, column='pk', column_to_pivot = 'k', new_column_prefix=None):
    if new_column_prefix is None:
        new_column_prefix = column_to_pivot
    r = sorted(df[column_to_pivot].unique())
    r_mapping = {r[i]: i for i in range(len(r))}
    df[f"{column_to_pivot}_value"] = df[column_to_pivot].apply(lambda x: f"{new_column_prefix}_{r_mapping[x]}")
    df = df.fillna(0.)
    df = pd.pivot_table(
        df,
        values=column,
        index=[col for col in df.columns if col not in (column, column_to_pivot, f"{column_to_pivot}_value")],
        columns=f"{column_to_pivot}_value",
    ).reset_index(drop=False)
    columns_ordered = [col for col in df.columns if not col.startswith(f'{new_column_prefix}_')]
    for i in range(len(r)):
        columns_ordered += [f'{new_column_prefix}_{i}']
    return df[columns_ordered]


if __name__ == '__main__':
    DATA_DIR = Path("/cosma7/data/dp004/dc-cues1/FORGE/")
    train_df = pd.read_csv(DATA_DIR / "redshift_train.csv",)
    test_df = pd.read_csv(DATA_DIR / "redshift_test.csv",)
    val_df = pd.read_csv(DATA_DIR / "redshift_val.csv",)

    train_df = pivot_k(train_df)
    test_df = pivot_k(test_df)
    val_df = pivot_k(val_df)

    train_df.to_csv(DATA_DIR / "redshift_train_multi.csv", index=False)
    test_df.to_csv(DATA_DIR / "redshift_test_multi.csv", index=False)
    val_df.to_csv(DATA_DIR / "redshift_val_multi.csv", index=False)
