import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

from .data_generator import generate_nanoparticle_data


def train_and_evaluate():
    # ── Data ──────────────────────────────────────────────────────────────
    df = generate_nanoparticle_data(n_samples=500)
    X = df.drop("toxic", axis=1)
    y = df["toxic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Models ────────────────────────────────────────────────────────────
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":           SVC(kernel="rbf", probability=True, random_state=42),
        "KNN":           KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    os.makedirs("results", exist_ok=True)
    os.makedirs("model",   exist_ok=True)

    # ── Train / Evaluate ──────────────────────────────────────────────────
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        cv  = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model":    model,
            "accuracy": acc,
            "cv_mean":  cv.mean(),
            "cv_std":   cv.std(),
            "y_pred":   y_pred,
            "y_proba":  y_proba,
        }

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  CV Accuracy   : {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Toxic','Toxic'])}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_accuracy_comparison(results)
    _plot_confusion_matrices(results, y_test)
    _plot_roc_curves(results, y_test)
    _plot_feature_importance(results["Random Forest"]["model"], X.columns)

    # ── Save best model ───────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    joblib.dump({"model": best_model, "scaler": scaler, "features": list(X.columns)},
                "model/best_model.pkl")
    print(f"\n✅  Best model: {best_name}  (accuracy={results[best_name]['accuracy']:.4f})")
    print("    Saved to model/best_model.pkl")

    return results


# ── Helper plots ──────────────────────────────────────────────────────────────

def _plot_accuracy_comparison(results):
    names  = list(results.keys())
    test_acc = [results[n]["accuracy"] for n in names]
    cv_mean  = [results[n]["cv_mean"]  for n in names]
    cv_std   = [results[n]["cv_std"]   for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - 0.2, test_acc, 0.35, label="Test Accuracy", color="#2196F3")
    bars2 = ax.bar(x + 0.2, cv_mean,  0.35, label="CV Accuracy",   color="#4CAF50",
                   yerr=cv_std, capsize=5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xticks(x);  ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    for bar in [*bars1, *bars2]:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/accuracy_comparison.png", dpi=150)
    plt.close()


def _plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Non-Toxic", "Toxic"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.4f}")
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=150)
    plt.close()


def _plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2196F3", "#F44336", "#4CAF50"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate");  ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=150)
    plt.close()


def _plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)),
           importances[indices], color="#2196F3", edgecolor="white")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Random Forest – Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()