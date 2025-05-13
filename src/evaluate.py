import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_models(model_dir="models"):
    """
    Loads all joblib models from a directory.
    """
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith(".joblib"):
            name = file.replace(".joblib", "")
            models[name] = joblib.load(os.path.join(model_dir, file))
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluates a dictionary of models on test data.
    Returns a DataFrame of metrics.
    """
    metrics = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }

    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics["Model"].append(name)
        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["Precision"].append(precision_score(y_test, y_pred))
        metrics["Recall"].append(recall_score(y_test, y_pred))
        metrics["F1 Score"].append(f1_score(y_test, y_pred))

    return pd.DataFrame(metrics)

def plot_metrics(df):
    """
    Plots a bar chart for all model performance metrics.
    """
    df.set_index("Model").plot(kind="bar", figsize=(12, 6))
    plt.title("Model Comparison Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

# Example run
if __name__ == "__main__":
    from preprocessing import clean_and_scale, apply_smote
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv("data/creditcard.csv")
    X, y = clean_and_scale(df)
    X_res, y_res = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    models = load_models()
    metrics_df = evaluate_models(models, X_test, y_test)
    print(metrics_df)
    plot_metrics(metrics_df)
