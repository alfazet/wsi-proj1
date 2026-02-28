#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

data_path = "./data/classification"
dataset_name = sys.argv[1]

def encode_categorical(df):
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

df = pd.read_csv(f"{data_path}/{dataset_name}")
df = encode_categorical(df)
df.to_csv(f"{data_path}/cleaned_{dataset_name}")
