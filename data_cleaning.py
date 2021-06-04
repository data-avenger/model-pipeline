#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define constants
SELECT_COLS = ['Market/Regular ',  'actual_eta', 'Org_lat_lon', 'Des_lat_lon', 'ontime',
'delay','trip_start_date', 'trip_end_date', 'TRANSPORTATION_DISTANCE_IN_KM', 
'supplierNameCode', 'Material Shipped']
DATE_COLS = ['trip_start_date', 'trip_end_date']
FLOAT_COLS = ['origin_lat','origin_lon','des_lat','des_lon','days_taken']
CAT_COLS = ['market_regular','supplier_name_code','material_shipped']
REPLACE_REGEX_DICT = {r'(?!^)([A-Z]+)':r'_\1','/| ': '','__':'_','T_R':'TR','/':''}
DROP_COLS = ['ontime', 'delay','trip_start_date','trip_end_date','actual_eta','org_lat_lon','des_lat_lon']

# ## Understand/View 
df = pd.read_csv('logistics-data.csv')

# ## Rename/Select
# The column names are a bit of a mess. Better make them consistent with `snake_case` style (it's the Python standard, get it)!
df = df[SELECT_COLS]
for col_name, rename_col in REPLACE_REGEX_DICT.items():
    df.columns = df.columns.str.replace(col_name, rename_col)
df.columns = df.columns.str.lower()


# ## Create/Drop Columns
df[['origin_lat', 'origin_lon']] = df['org_lat_lon'].str.split(',',expand=True) 
df[['des_lat', 'des_lon']] = df['des_lat_lon'].str.split(',',expand=True) 
df[DATE_COLS] = df[DATE_COLS].astype('datetime64')
days_taken = (df['trip_end_date']-df['trip_start_date'])/ np.timedelta64(1, 'D')
df.insert(0,'days_taken',days_taken)
df = df.drop(df[df['days_taken']<0].index, axis=0)
df = df.drop(DROP_COLS, axis=1)


# ## Convert Data Types
df[FLOAT_COLS] = df[FLOAT_COLS].astype("float")
df[CAT_COLS] = df[CAT_COLS].astype("category")

# ## Handle Missing Values
missing_values_df = df.loc[:, df.isnull().sum()>0]

# Fill missing values for y data (do not scale as we want do not want to unscale for predictions)
y = df["days_taken"]
num_imputer = SimpleImputer(missing_values= np.nan, strategy='median')
y_pre = num_imputer.fit_transform(y.to_numpy().reshape(len(y), 1)) 


# Preprocess features X
df.drop("days_taken",axis=1, inplace=True)
numeric_features =  df.select_dtypes('number').columns
numeric_transformer = Pipeline(
steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_features = df.select_dtypes(exclude='number').columns
categorical_transformer = Pipeline(
steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
]
)
preprocess = ColumnTransformer(
transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
]
)
X_pre = preprocess.fit_transform(df)
dataset = np.concatenate((y_pre, X_pre.toarray()), axis=1)

pd.DataFrame(dataset).to_csv('clean-data.csv', index=False)