from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

def select_features(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns)
    feat_imp.sort_values().plot(kind='barh', figsize=(10, 6), title="Feature Importances")
    plt.show()

if __name__ == "__main__":
    select_features()
