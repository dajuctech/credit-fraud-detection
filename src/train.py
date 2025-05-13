import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def train_models(X, y, save_dir="models"):
    """
    Trains multiple classification models and saves them to disk.

    Args:
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
        save_dir (str): Directory to save trained models.

    Returns:
        dict: Evaluation results of each model.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        #"RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        #"NaiveBayes": GaussianNB(),
        #"SVM": SVC(probability=True),
        #"KNN": KNeighborsClassifier(),
        #"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    for name, model in models.items():
        print(f"\n📌 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print(f"✅ {name} Evaluation:")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))

        # Save model
        model_path = os.path.join(save_dir, f"{name}.joblib")
        joblib.dump(model, model_path)

        results[name] = {
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds, output_dict=True)
        }

    return results

# Example run for testing
if __name__ == "__main__":
    from preprocessing import clean_and_scale, apply_smote

    # Load dataset
    df = pd.read_csv("data/creditcard.csv")

    # Preprocessing
    X, y = clean_and_scale(df)
    X_res, y_res = apply_smote(X, y)

    # Train models and save
    results = train_models(X_res, y_res)
