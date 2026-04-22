import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tp6")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy de la Régression Logistique : {accuracy:.4f}")

    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame(
        {
            "Feature": data.feature_names,
            "Coefficient": coefficients,
        }
    )
    feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
    feature_importance = feature_importance.sort_values(by="Abs_Coefficient", ascending=True)

    plt.figure(figsize=(10, 8))
    colors = ["red" if c < 0 else "blue" for c in feature_importance["Coefficient"]]
    plt.barh(feature_importance["Feature"][-15:], feature_importance["Coefficient"][-15:], color=colors[-15:])
    plt.xlabel("Valeur du Coefficient (β)")
    plt.title("Top 15 - Importance des variables (Régression Logistique)")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()

    output_filename = "TP6/outputs/tabular/glassbox_coefficients.png"
    plt.savefig(output_filename)
    print(f"Graphique sauvegardé dans {output_filename}")

    top_negative = feature_importance.sort_values(by="Coefficient").head(3)[["Feature", "Coefficient"]]
    top_positive = feature_importance.sort_values(by="Coefficient", ascending=False).head(3)[["Feature", "Coefficient"]]
    print("\nTop 3 vers Maligne (classe 0):")
    print(top_negative.to_string(index=False))
    print("\nTop 3 vers Bénigne (classe 1):")
    print(top_positive.to_string(index=False))


if __name__ == "__main__":
    main()
