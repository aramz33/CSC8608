import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tp6")

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import shap
import numpy as np


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

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Accuracy du Random Forest : {model.score(X_test, y_test):.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap_values_class1 = shap_values[:, :, 1]

    patient_idx = 0
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_class1[patient_idx], show=False)
    plt.tight_layout()
    output_local = "TP6/outputs/tabular/shap_waterfall.png"
    plt.savefig(output_local, bbox_inches="tight")
    plt.close()
    print(f"Waterfall plot sauvegardé dans {output_local}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_class1, X_test, show=False)
    plt.tight_layout()
    output_global = "TP6/outputs/tabular/shap_summary.png"
    plt.savefig(output_global, bbox_inches="tight")
    plt.close()
    print(f"Summary plot sauvegardé dans {output_global}")

    mean_abs_shap = np.abs(shap_values_class1.values).mean(axis=0)
    top_global_idx = np.argsort(mean_abs_shap)[::-1][:5]
    print("\nTop 5 variables globales (mean |SHAP|):")
    for idx in top_global_idx:
        print(f"- {X_test.columns[idx]}: {mean_abs_shap[idx]:.4f}")

    patient_values = shap_values_class1[patient_idx].values
    patient_features = X_test.iloc[patient_idx]
    dominant_idx = int(abs(patient_values).argmax())
    dominant_feature = X_test.columns[dominant_idx]
    dominant_value = patient_features.iloc[dominant_idx]
    dominant_shap = patient_values[dominant_idx]
    print(
        "Patient 0 - plus forte contribution : "
        f"{dominant_feature} = {dominant_value:.4f} (SHAP={dominant_shap:.4f})"
    )


if __name__ == "__main__":
    main()
