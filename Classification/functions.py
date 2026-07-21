"""Reusable functions for the classification models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay)

RANDOM_STATE = 42
LABELS = ["Negative", "Neutral", "Positive"]


def evaluate(y_true, y_pred, model_name="model", plot=True):
    """Score a model and return a dict of metrics."""
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    for label in LABELS:
        results[f"f1_{label.lower()}"] = f1_score(
            y_true, y_pred, labels=[label], average="macro", zero_division=0)

    print(f"=== {model_name} ===")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred, labels=LABELS),
            display_labels=LABELS).plot(ax=axes[0], cmap="Blues", colorbar=False)
        axes[0].set_title(f"{model_name} — counts")
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true").round(2),
            display_labels=LABELS).plot(ax=axes[1], cmap="Blues", colorbar=False)
        axes[1].set_title("Normalised by true class")
        plt.tight_layout()
        plt.show()

    return results

def load_splits(path="../data/processed/amazon_reviews_clean.csv"):
    """Return the canonical train/val/test split. Identical for every model."""
    df = pd.read_csv(path, low_memory=False)
    df["text_full"] = (df["reviews.title"].fillna("") + ". " + df["reviews.text"]).str.strip()
    df = df.dropna(subset=["text_full", "sentiment"])

    X, y = df["text_full"], df["sentiment"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    return X_train, X_val, X_test, y_train, y_val, y_test