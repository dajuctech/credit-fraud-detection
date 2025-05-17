import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(save_dir='models'):
    X_test = joblib.load(os.path.join(save_dir, "X_test.joblib"))
    y_test = joblib.load(os.path.join(save_dir, "y_test.joblib"))
    return X_test, y_test

def evaluate_models(save_dir='models'):
    X_test, y_test = load_data(save_dir)
    results = {}

    for model_file in os.listdir(save_dir):
        if model_file.endswith(".joblib") and "test" not in model_file:
            model_path = os.path.join(save_dir, model_file)
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            results[model_file] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }

    return results

if __name__ == "__main__":
    evaluation_results = evaluate_models()
    for model, metrics in evaluation_results.items():
        print(f"\n📊 {model}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
