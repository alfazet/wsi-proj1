#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

data_path = "./data/regression"
dataset_name = sys.argv[1] 

def encode_categorical(df):
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def check_df(df):
    cols_with_q = df.columns[df.isin(["?"]).any()]
    print(cols_with_q)

def remove_question_marks(df):
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

df = pd.read_csv(f"{data_path}/{dataset_name}")
check_df(df)
remove_question_marks(df)
df = encode_categorical(df)
check_df(df)
df.to_csv(f"{data_path}/cleaned_{dataset_name}")
