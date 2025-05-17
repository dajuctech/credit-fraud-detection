import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from preprocessing import preprocess_data

def train_models(data_path='data/creditcard.csv', save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load and preprocess data
    df = pd.read_csv(data_path)
    X_resampled, y_resampled = preprocess_data(df)

    # Split the resampled dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Define models to train
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000)
        #"random_forest": RandomForestClassifier(n_estimators=100),
        #"svm": SVC(probability=True),
        #"knn": KNeighborsClassifier(),
        #"xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Train and save models
    for name, model in models.items():
        print(f"🚀 Training {name}...")
        model.fit(X_train, y_train)
        model_path = os.path.join(save_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"✅ Saved {name} to {model_path}")

    # Save test sets for evaluation
    joblib.dump(X_test, os.path.join(save_dir, "X_test.joblib"))
    joblib.dump(y_test, os.path.join(save_dir, "y_test.joblib"))
    print("🎉 All models trained and test sets saved successfully.")

if __name__ == "__main__":
    train_models()
