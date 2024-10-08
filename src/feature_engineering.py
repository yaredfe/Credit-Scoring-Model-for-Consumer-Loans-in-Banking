import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore
from xverse.transformer import WOE
import numpy as np

# 1. Aggregate Features
def create_aggregate_features(df,customer_id_col,transaction_col):
    aggregate_features = df.groupby(customer_id_col).agg(
        total_transaction_amount=(transaction_col, 'sum'),
        average_transaction_amount=(transaction_col, 'mean'),
        transaction_count=(transaction_col, 'count'),
        transaction_stddev=(transaction_col, 'std')
    ).reset_index()
    df=df.merge(aggregate_features,on=customer_id_col,how="left")
    return df
# 2. Extract Transaction Date Features
def extract_date_features(df, date_col):
    df[date_col]= pd.to_datetime(df[date_col])
    df['transaction_hour'] = df[date_col].dt.hour
    df['transaction_day'] = df[date_col].dt.day
    df['transaction_month'] = df[date_col].dt.month
    df['transaction_year'] = df[date_col].dt.year
    return df

def strip(df):
    categorical_columns=df.select_dtypes(include="object").columns
    for column in categorical_columns:
        if df[column].str.contains(r'_\d+').any():
            df[column] = df[column].str.extract(r'_(\d+)').astype(float)
        else:
            pass

    return df

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns, dtype=int)

# 4. Handle Missing Values
def handle_missing_values(df, column, strategy='mean'):
    if strategy == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    elif strategy == 'mode':
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df

# 5. Normalize/Standardize Numerical Features
# b. Standardization
def standardize(df, column):
    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[column])
    return df
# a. Normalization
def normalize(df, column):
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[column])
    return df

# 6. Weight of Evidence (WoE) Transformation
def apply_woe(df, columns, target):
    print("Applying WOE transformation...")
    print("DataFrame shape:", df.shape)
    print("Columns passed:", columns)

    # Check for missing columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")

    # Check if target column exists
    if target not in df.columns:
        raise ValueError(f"The target column '{target}' is missing from the DataFrame.")

    woe_transformer = WOE()
    woe_dfs = []

    for col in columns:
        print(f"Processing column: {col}")  # Add this line for debugging
        woe_transformer.fit(df[[col]], df[target])
        woe_df = woe_transformer.transform(df[[col]])
        woe_dfs.append(woe_df)

    return pd.concat(woe_dfs, axis=1)

# 7. Recency, Frequency, Monetary, and Score (RFMS) Calculation
def calculate_rfms(df, customer_id_col, date_col, transaction_col):
    df[date_col] = pd.to_datetime(df[date_col])

   # Group by customer_id to get the date of each customer's last transaction
    df['last_transaction_date'] = df.groupby(customer_id_col)[date_col].transform('max')

    # Calculate recency: Difference in days between the last transaction date and the current date (or dataset max)

    current_date = df[date_col].max()  # Use the most recent date in the dataset

    df['recency'] = (current_date - df['last_transaction_date']).dt.days
    # Frequency: Total transaction count
    df['frequency'] = df.groupby(customer_id_col)[transaction_col].transform('count')

    # Monetary: Total transaction amount
    df['monetary'] = df.groupby(customer_id_col)[transaction_col].transform('sum')

    # Calculate RFMS Score
    df['rfms_score'] = df['recency'] * 0.2 + df['frequency'] * 0.4 + df['monetary'] * 0.4

    return df

# 8. Assign Default Risk Label based on RFMS Score
def assign_default_risk(df, score_col, threshold=0.5):
    df['default_risk'] = df[score_col].apply(lambda x: 'good' if x > threshold else 'bad')
    return df