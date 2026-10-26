"""
Generate the figures embedded in the README (saved to screenshots/).

Reproducible from the committed data + pipeline + reports:
    python scripts/generate_figures.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.data_loader import clean_data, load_telco_data
from src.explain import explain_customer
from src.feature_engineering import separate_features_and_target
from src.pipeline import FeatureEngineer  # noqa: F401  (unpickling)

OUT = "screenshots"
os.makedirs(OUT, exist_ok=True)
sns.set_style("whitegrid")
RED, GREEN, BLUE = "#e74c3c", "#2ecc71", "#3498db"

DATA = "data/WA_Fn-UseC_-_Telco_Customer_Churn.csv"
df_raw = load_telco_data(DATA)
df = clean_data(df_raw)
X, y = separate_features_and_target(df)
pipe = joblib.load("models/churn_pipeline.pkl")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# 1. Churn rate by contract — the headline business signal
rate = df_raw.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).reindex(
    ["Month-to-month", "One year", "Two year"]
)
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(rate.index, rate.values * 100, color=[RED, "#e67e22", GREEN])
ax.bar_label(bars, fmt="%.0f%%", padding=3, fontweight="bold")
ax.set_title("Churn rate by contract type", fontweight="bold")
ax.set_ylabel("Churn rate (%)")
ax.set_ylim(0, max(rate.values * 100) * 1.2)
save(fig, "fig_churn_by_contract.png")

# 2. Model comparison
comp = pd.read_csv("reports/model_comparison.csv").set_index("Model")
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
fig, ax = plt.subplots(figsize=(9, 4.8))
comp[metrics].T.plot.bar(ax=ax, color=[BLUE, RED], width=0.8)
ax.set_title("Model comparison (held-out test set)", fontweight="bold")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_xticklabels(metrics, rotation=0)
ax.legend(title="")
save(fig, "fig_model_comparison.png")

# 3. Confusion matrix of the deployed model
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cm = confusion_matrix(y_test, pipe.predict(X_test))
fig, ax = plt.subplots(figsize=(5.2, 4.4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix — deployed model", fontweight="bold")
save(fig, "fig_confusion_matrix.png")

# 4. Permutation feature importance
imp = pd.read_csv("reports/feature_importance.csv").sort_values("Importance").tail(12)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(imp["Feature"], imp["Importance"], color=BLUE)
ax.set_title("Permutation feature importance", fontweight="bold")
ax.set_xlabel("Mean ROC-AUC drop when shuffled")
save(fig, "fig_feature_importance.png")

# 5. Per-customer explanation (linear SHAP values) — the showcase
high_risk = dict(
    gender="Male", SeniorCitizen=1, Partner="No", Dependents="No", tenure=1,
    PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
    OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
    StreamingTV="Yes", StreamingMovies="Yes", Contract="Month-to-month",
    PaperlessBilling="Yes", PaymentMethod="Electronic check",
    MonthlyCharges=95.0, TotalCharges=190.0,
)
proba = float(pipe.predict_proba(pd.DataFrame([high_risk]))[0, 1])
exp = explain_customer(pipe, pd.DataFrame([high_risk])).head(8).iloc[::-1]
colors = [RED if c > 0 else GREEN for c in exp["contribution"]]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(exp["feature"], exp["contribution"], color=colors)
ax.axvline(0, color="#333", linewidth=0.8)
ax.set_title(f"Why is this customer at risk?  (churn probability {proba:.0%})", fontweight="bold")
ax.set_xlabel("Contribution to churn log-odds  (red = increases risk, green = reduces)")
save(fig, "fig_explanation_example.png")

print("\nAll figures generated.")
