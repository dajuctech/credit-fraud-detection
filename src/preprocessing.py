import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def clean_and_scale(df):
    df = df.drop_duplicates()
    df = df.drop(['Time'], axis=1)
    
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

if __name__ == "__main__":
    clean_and_scale()
