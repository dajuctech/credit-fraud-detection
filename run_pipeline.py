import pandas as pd

from src.eda import run_eda
from src.preprocessing import clean_and_scale, apply_smote
from src.feature_selection import select_features

# Load data
df = pd.read_csv('data/creditcard.csv')

# Run EDA
run_eda(df)

# Preprocessing
X, y = clean_and_scale(df)
X_resampled, y_resampled = apply_smote(X, y)

# Feature Selection
select_features(X_resampled, y_resampled)
