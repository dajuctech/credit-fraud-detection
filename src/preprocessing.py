# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """
    Cleans, scales, and balances the credit card fraud dataset.

    Args:
        df (DataFrame): Raw input dataset.

    Returns:
        X_resampled, y_resampled (tuple): Preprocessed and balanced features and target.
    """

    # Remove duplicates
    df = df.drop_duplicates()

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()  # Or handle imputation here

    # Drop the 'Time' column
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    # Scale the 'Amount' column
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Split features and labels
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

if __name__ == "__main__":
    df = pd.read_csv("data/creditcard.csv")
    X_res, y_res = preprocess_data(df)
    print(f"✅ Resampled data shape: {X_res.shape}, {y_res.shape}")
    print(f"Class distribution after resampling:\n{pd.Series(y_res).value_counts()}")
