#!/usr/bin/env python

import pandas as pd

data_path = "./data/regression"


def check_df(df):
    cols_with_q = df.columns[df.isin(["?"]).any()]
    print(cols_with_q)


df = pd.read_csv(f"{data_path}/dataset.csv")
print(df.head())

check_df(df)

numeric_cols = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].mean())

na_cols = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "Fence",
    "PoolQC",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "MiscFeature",
]
for col in na_cols:
    df[col] = df[col].replace("?", "NA")

none_cols = ["MasVnrType"]
for col in none_cols:
    df[col] = df[col].replace("?", "None")

mix_cols = ["Electrical"]
for col in mix_cols:
    df[col] = df[col].replace("?", "Mix")

check_df(df)
df.to_csv(f"{data_path}/cleaned_dataset.csv")
