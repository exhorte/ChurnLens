"""About / Model Card: what this model is, how it was built, and where it must not be trusted."""
from __future__ import annotations

import streamlit as st

from .common import NOT_A_GUARANTEE, num, pct
from .context import AppContext

LIMITATIONS = [
    "Trained on a public **sample** dataset of one telecom provider: results say nothing about the customers of "
    "a real company, another market or another period.",
    "**Snapshot data without dates.** `tenure` and `TotalCharges` are cumulative values observed at extraction "
    "time; for churned customers they stop at departure, so part of their signal is *consequence*, not cause. "
    "No temporal validation was possible.",
    "**Associations, not causes.** Importance and explanations describe what the model learned; they do not show "
    "what drives churn or what would change it.",
    "**Probabilities are a ranking tool first.** They were checked for calibration on validation data only; "
    "calibration can drift on new populations.",
    "**Sensitive attributes** (`gender`, `SeniorCitizen`) are inputs of the demo model. No fairness audit was "
    "performed: do not use scores to make decisions about individuals without such a review.",
    "The decision threshold is a statistical default (F1 on validation data). A real campaign threshold depends "
    "on contact cost and customer value (see Retention Simulator).",
    "The retention simulator is a scenario calculator under assumptions you enter; it does not measure the "
    "causal effect of any campaign.",
    "Fixed hyperparameters (no tuning) and a single train/test split: differences smaller than the "
    "cross-validation standard deviation should not be over-interpreted.",
]


def render(ctx: AppContext) -> None:
    ws, meta = ctx.ws, ctx.ws.metadata
    st.title("About / Model Card")
    st.warning(NOT_A_GUARANTEE + " It supports analysis; it does not replace business validation.")
    if not ws.is_demo:
        st.info("**Custom model** trained in this session on your own data. It is not equivalent to the demo "
                "model, has not been reviewed, and disappears when the session ends.")

    st.subheader("Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", ws.bundle.label)
    c2.metric("Trained", ws.bundle.trained_at[:19].replace("T", " "))
    c3.metric("Run", meta["run_id"])
    st.write(f"Selection: {meta['selection']['explanation']}")
    st.write(f"Hyperparameters (fixed): `{meta['model_params']}`")
    env = meta["environment"]
    st.caption(f"Python {env['python']} · scikit-learn {env['scikit_learn']} · pandas {env['pandas']} · "
               f"numpy {env['numpy']} · config {meta['config_fingerprint']} · seeds {meta['random_state']}")

    st.subheader("Data")
    data = meta["data"]
    st.write(f"{data['rows']:,} labelled rows ({data['n_train']:,} train / {data['n_test']:,} test), positive rate "
             f"{pct(data['positive_rate'])}. Target: `{meta['target']['column']}`, positive class "
             f"`{meta['target']['positive_label']}`. Identifier column: `{data['id_column']}`.")
    if data.get("file_sha256"):
        st.caption(f"File `{data['file']}` · SHA-256 `{data['file_sha256'][:16]}…`")
    if ws.ignored_columns:
        st.write("**Columns ignored / rejected:**")
        st.dataframe([{"column": c, "reason": why} for c, why in ws.ignored_columns.items()],
                     hide_index=True, width="stretch")
    st.write("**Input contract** (columns the model needs):")
    st.dataframe([{"column": c["name"], "type": c["kind"],
                   "allowed / seen": ", ".join(c["categories"]) if c["kind"] == "categorical"
                   else f"{c['minimum']:g} to {c['maximum']:g}"} for c in meta["schema"]],
                 hide_index=True, width="stretch")
    if ws.is_demo:
        st.caption("Source: IBM Telco customer churn sample dataset (Kaggle: blastchar/telco-customer-churn). Its "
                   "licence could not be verified: it is not redistributed here. See docs/DATA_CARD.md.")

    st.subheader("Performance")
    cv, test, cal = meta["cv_scores_selected_model"], ws.test_metrics, meta["calibration_out_of_fold"]
    ci = meta.get("test_metrics_ci_95", {})
    interval = lambda m: f" (95% interval {ci[m][0]:.3f}-{ci[m][1]:.3f})" if m in ci else ""  # noqa: E731
    st.markdown(
        f"- **Validation** (5-fold CV on training split): ROC-AUC {num(cv['roc_auc_mean'])} ± {num(cv['roc_auc_std'])}, "
        f"Average precision {num(cv['average_precision_mean'])} ± {num(cv['average_precision_std'])}.\n"
        f"- **Test set (used once)** at threshold {meta['threshold']['threshold']:.2f}: ROC-AUC {num(test['roc_auc'])}"
        f"{interval('roc_auc')}, Average precision {num(test['average_precision'])}{interval('average_precision')}, "
        f"precision {num(test['precision'])}, "
        f"recall {num(test['recall'])}, F1 {num(test['f1'])}.\n"
        f"- **Calibration** (out-of-fold): mean predicted {pct(cal['mean_predicted'])} vs observed "
        f"{pct(cal['observed_rate'])}, Brier {num(cal['brier_score'])}."
    )
    st.caption(f"Threshold: {meta['threshold']['description']} Risk segments: Low < {meta['segments']['low_below']:.2f} "
               f"≤ Medium < {meta['segments']['high_from']:.2f} ≤ High (business convention).")

    st.subheader("Intended use")
    st.write("Exploring and demonstrating a churn-analysis workflow: comparing models, studying errors and "
             "thresholds, and testing retention scenarios. **Not** intended for automated decisions about people.")
    st.subheader("Limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    if meta.get("screening_warnings"):
        st.subheader("Data screening warnings")
        for item in meta["screening_warnings"]:
            st.markdown(f"- {item}")
    with st.expander("Full metadata (JSON)"):
        st.json(meta)
