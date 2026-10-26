# -*- coding: utf-8 -*-
"""Generate the product documentation PDF for Customer Churn Prediction."""
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image, Preformatted, ListFlowable, ListItem, HRFlowable,
)
from PIL import Image as PILImage

OUT = "docs/Customer_Churn_Prediction_Documentation.pdf"
NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#2e6da4")
ACCENT = colors.HexColor("#3498db")
LIGHT = colors.HexColor("#eef3f8")
GREY = colors.HexColor("#566573")
CODEBG = colors.HexColor("#f4f6f8")

styles = getSampleStyleSheet()
def S(name, **kw):
    parent = kw.pop("parent", styles["Normal"])
    styles.add(ParagraphStyle(name, parent=parent, **kw))

S("Body", fontSize=10.2, leading=15, alignment=TA_JUSTIFY, spaceAfter=7, textColor=colors.HexColor("#1c2833"))
S("H1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=6, spaceAfter=10)
S("H2", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=5)
S("Caption", fontSize=8.6, leading=11, alignment=TA_CENTER, textColor=GREY, spaceAfter=10, spaceBefore=3)
S("CodeBlock", parent=styles["Code"], fontName="Courier", fontSize=8.2, leading=11.2,
  backColor=CODEBG, borderColor=colors.HexColor("#d5dbdb"), borderWidth=0.5,
  borderPadding=7, leftIndent=2, spaceBefore=4, spaceAfter=9)
S("BulletItem", fontSize=10.2, leading=14.5, spaceAfter=3, textColor=colors.HexColor("#1c2833"))
S("TOCItem", fontSize=10.8, leading=18, textColor=colors.HexColor("#1c2833"))
S("CoverTitle", fontName="Helvetica-Bold", fontSize=30, leading=35, alignment=TA_CENTER, textColor=NAVY)
S("CoverSub", fontSize=14, leading=20, alignment=TA_CENTER, textColor=BLUE)
S("CoverMeta", fontSize=10.5, leading=16, alignment=TA_CENTER, textColor=GREY)

story = []
def H1(t): story.append(Paragraph(t, styles["H1"]))
def H2(t): story.append(Paragraph(t, styles["H2"]))
def P(t): story.append(Paragraph(t, styles["Body"]))
def code(t): story.append(Preformatted(t, styles["CodeBlock"]))
def gap(h=6): story.append(Spacer(1, h))
def rule(): story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#d5dbdb"), spaceBefore=4, spaceAfter=10))
def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(i, styles["BulletItem"]), leftIndent=12, value="•") for i in items],
        bulletType="bullet", leftIndent=10, spaceAfter=8))
def figure(path, caption, width=14*cm):
    iw, ih = PILImage.open(path).size
    story.append(Spacer(1, 4))
    story.append(Image(path, width=width, height=width * ih / iw))
    story.append(Paragraph(caption, styles["Caption"]))
def table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    ts = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1c2833")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cdd5dd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(ts))
    story.append(t); story.append(Spacer(1, 10))

# COVER
story.append(Spacer(1, 4.5*cm))
story.append(Paragraph("Customer Churn Prediction", styles["CoverTitle"]))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Product &amp; Technical Documentation", styles["CoverSub"]))
story.append(Spacer(1, 1.2*cm))
story.append(HRFlowable(width="40%", thickness=1.5, color=ACCENT, hAlign="CENTER"))
story.append(Spacer(1, 1.2*cm))
story.append(Paragraph(
    "A machine-learning system that predicts which telecom customers are likely to "
    "churn, served through an interactive web app and a REST API &mdash; with a second "
    "mode that trains a fresh model on <i>any</i> uploaded dataset.",
    ParagraphStyle("ct", parent=styles["CoverMeta"], fontSize=11, leading=16)))
story.append(Spacer(1, 2.0*cm))
story.append(Paragraph("Davide Infantino &nbsp;&middot;&nbsp; github.com/dinfalabs/customer-churn-prediction", styles["CoverMeta"]))
story.append(Paragraph("Version 2.0 &nbsp;&middot;&nbsp; " + datetime.date.today().strftime("%B %Y"), styles["CoverMeta"]))
story.append(PageBreak())

# TOC
H1("Contents"); rule()
toc = [
    ("1. Executive Summary", 0),
    ("2. Case Study: From a Broken Model to ROC-AUC 0.85", 0),
    ("3. How It Works, Step by Step", 0),
    ("3.1 Data loading and the synthetic-data guardrail", 1),
    ("3.2 Cleaning", 1),
    ("3.3 Feature engineering", 1),
    ("3.4 The modeling pipeline (fit-once / serve-once)", 1),
    ("3.5 Training and model selection", 1),
    ("3.6 Evaluation and metrics", 1),
    ("3.7 Key churn drivers", 1),
    ("3.8 Explainability (linear SHAP values)", 1),
    ("4. The Two Modes (Demo and Bring-Your-Own-Data)", 0),
    ("5. The Scoring API", 0),
    ("6. Architecture and Project Structure", 0),
    ("7. Deployment", 0),
    ("8. Quality: Testing and Continuous Integration", 0),
    ("9. Limitations and Roadmap", 0),
    ("10. Glossary", 0),
]
for label, lvl in toc:
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" * lvl + label, styles["TOCItem"]))
story.append(PageBreak())

# 1
H1("1. Executive Summary"); rule()
P("<b>Customer Churn Prediction</b> estimates the probability that a telecom customer "
  "will cancel their subscription (\"churn\"), so that retention teams can act <i>before</i> "
  "the customer leaves. It is a complete, production-style machine-learning system rather "
  "than a notebook experiment.")
P("The product offers two ways to use it:")
bullets([
    "<b>Demo mode</b> &mdash; explore a model trained on the well-known Telco Customer "
    "Churn dataset: analyse churn patterns, score individual customers, and read an "
    "explanation of every prediction.",
    "<b>Bring-your-own-data mode</b> &mdash; upload <i>any</i> CSV with a binary target; "
    "the app detects feature types, trains and compares models, lets you predict, and "
    "download the trained model. This turns the project from a demo into a reusable tool.",
])
P("On the real Telco test set the deployed model reaches a <b>ROC-AUC of 0.847</b> and a "
  "<b>recall of 0.80</b> &mdash; it correctly identifies about 80% of the customers who "
  "will actually churn. Predictions are served by both a Streamlit web application and a "
  "FastAPI service, and every prediction is accompanied by a per-customer explanation.")
gap()
table([
    ["At a glance", ""],
    ["Problem", "Binary classification: will a customer churn? (Yes / No)"],
    ["Dataset", "Telco Customer Churn — 7,043 customers, 19 features, ~26.5% churn"],
    ["Deployed model", "Logistic Regression (selected over Random Forest by cross-validation)"],
    ["Headline metrics", "ROC-AUC 0.847 · Recall 0.797 · F1 0.611 · Precision 0.496"],
    ["Interfaces", "Streamlit web app · FastAPI REST service · Docker image"],
    ["Quality", "21 automated tests · GitHub Actions CI · per-customer explanations"],
], [4.2*cm, 11.3*cm])

# 2
H1("2. Case Study: From a Broken Model to ROC-AUC 0.85"); rule()
P("This project began as an <b>audit</b> of an existing codebase, and the audit uncovered "
  "a system that looked professional but did not work. Three findings stood out:")
bullets([
    "<b>The dataset was synthetic.</b> The committed CSV was random values from a fallback "
    "function, with the churn label statistically independent of every feature. No model "
    "can learn from noise.",
    "<b>The shipped model was dead.</b> It predicted \"No Churn\" for 100% of customers "
    "(F1 = 0, ROC-AUC = 0.49 — a coin toss).",
    "<b>The app ignored half its inputs.</b> A train/serve mismatch silently set five of "
    "the user's inputs to zero at prediction time.",
])
P("The refactor replaced the synthetic data with the real dataset (behind a guardrail that "
  "rejects synthetic data), and rebuilt the modeling code around a single <i>fit-once / "
  "serve-once</i> pipeline that eliminates the train/serve mismatch by construction:")
gap()
table([
    ["Metric (real test set)", "Before", "After"],
    ["ROC-AUC", "0.49 (random)", "0.847"],
    ["Recall (churners caught)", "0.00", "0.797"],
    ["F1-score", "0.00", "0.611"],
    ["Customers flagged as churn", "0 of 7,043", "realistic"],
], [7.5*cm, 4*cm, 4*cm])
P("The lesson embedded in the product: a model is only as good as its data and the "
  "consistency between how it is trained and how it is served.")

# 3
story.append(PageBreak())
H1("3. How It Works, Step by Step"); rule()
P("This section follows a customer record from raw data to an explained prediction. The "
  "same transformation chain is used during training and serving &mdash; the central "
  "design idea of the system.")
H2("3.1 Data loading and the synthetic-data guardrail")
P("The dataset is a required input — the loader never fabricates data. If the file is "
  "missing or looks synthetic (placeholder IDs, or no relationship between contract type "
  "and churn) it fails loudly instead of silently poisoning training:")
code('def load_telco_data(path):\n'
     '    if not os.path.exists(path):\n'
     '        raise FileNotFoundError("Download the real Telco dataset ...")\n'
     '    df = pd.read_csv(path)\n'
     '    _assert_is_real_telco(df)   # rejects placeholder / signal-free data\n'
     '    return df')
H2("3.2 Cleaning")
P("Cleaning removes duplicates, trims stray whitespace from text fields, and converts "
  "TotalCharges (which ships as text with 11 blank cells) to a number. Missing values are "
  "<b>not</b> filled here — imputation is delegated to the pipeline so that training and "
  "serving share exactly one strategy.")
H2("3.3 Feature engineering")
P("Four derived features add business signal. They are computed row-by-row, so they behave "
  "identically on a 7,000-row training set and on a single live customer:")
table([
    ["Feature", "Meaning"],
    ["TotalServices", "Count of active add-on services (security, backup, streaming, ...)"],
    ["ContractRisk", "Month-to-month = 3, One year = 2, Two year = 1"],
    ["ChargeRatio", "Monthly charges divided by total charges (tenure-adjusted spend)"],
    ["TenureSegment", "Tenure bucketed into 0-1y, 1-2y, 2-3y, 3y+"],
], [4.2*cm, 11.3*cm])
H2("3.4 The modeling pipeline (fit-once / serve-once)")
P("Feature engineering, missing-value imputation, scaling of numbers, and one-hot encoding "
  "of categories all live inside a single scikit-learn Pipeline object:")
code("FeatureEngineer\n"
     "   -> ColumnTransformer(\n"
     "        numeric:     impute(median) + StandardScaler\n"
     "        categorical: impute(most_frequent) + OneHotEncoder(ignore unknown))\n"
     "   -> Classifier (Logistic Regression / Random Forest)")
P("This object is fitted once during training and saved as one file (churn_pipeline.pkl). "
  "Both the app and the API call pipeline.predict_proba(raw_dataframe). Because nothing is "
  "re-fitted at prediction time, training and serving transform inputs identically &mdash; "
  "the bug that plagued the original code cannot recur. Unknown categories are handled "
  "safely, so single-row and previously unseen values never crash the model.")
H2("3.5 Training and model selection")
P("The training script splits off a 20% test set, then selects the best model using <b>"
  "5-fold cross-validation on the training set only</b>. The test set is touched exactly "
  "once, for the final report, so the reported numbers are not optimistically biased. Both "
  "candidates use class weighting to handle the ~26.5% churn imbalance.")
table([
    ["Candidate", "Outcome"],
    ["Logistic Regression", "Chosen — higher CV ROC-AUC (0.851); linear and interpretable"],
    ["Random Forest", "Runner-up — CV ROC-AUC 0.845"],
], [5.2*cm, 10.3*cm])
H2("3.6 Evaluation and metrics")
P("The deployed model is evaluated on the held-out test set of 1,409 customers. Recall is "
  "weighted heavily: in churn, catching customers who will actually leave usually matters "
  "more than occasionally flagging a loyal one.")
table([
    ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    ["Logistic Regression (deployed)", "0.731", "0.496", "0.797", "0.611", "0.847"],
    ["Random Forest", "0.771", "0.553", "0.727", "0.628", "0.840"],
], [6.1*cm, 1.9*cm, 1.9*cm, 1.7*cm, 1.6*cm, 1.9*cm])
figure("screenshots/fig_model_comparison.png", "Figure 1 — Model comparison on the held-out test set.")
figure("screenshots/fig_confusion_matrix.png",
       "Figure 2 — Confusion matrix of the deployed model; the bottom-right cell is the churners correctly caught.",
       width=9.5*cm)
H2("3.7 Key churn drivers")
P("Permutation importance (how much the score drops when a feature is shuffled) ranks the "
  "strongest drivers: total charges and tenure, internet-service type, and contract type. "
  "Month-to-month, fibre-optic, newer customers churn far more.")
figure("screenshots/fig_churn_by_contract.png",
       "Figure 3 — Churn rate by contract type: 43% for month-to-month vs 3% for two-year.",
       width=10.5*cm)
figure("screenshots/fig_feature_importance.png",
       "Figure 4 — Permutation feature importance for the deployed model.", width=12*cm)
H2("3.8 Explainability (linear SHAP values)")
P("Every prediction is explainable per customer. Because the deployed model is linear, the "
  "decomposition is <b>exact</b>: the churn log-odds equal the model intercept plus the sum "
  "of each feature's contribution. These contributions are precisely the (linear) SHAP "
  "values, computed with no extra dependencies:")
code("logit(churn) = intercept + sum_i ( coefficient_i  x  feature_i )")
P("Positive contributions push the customer toward churn (red), negative ones toward "
  "retention (green). The same chart appears live in the app under \"Why this prediction?\".")
figure("screenshots/fig_explanation_example.png",
       "Figure 5 — Per-customer explanation for a high-risk profile (94% churn probability).",
       width=13*cm)

# 4
story.append(PageBreak())
H1("4. The Two Modes"); rule()
H2("4.1 Demo (Telco)")
P("The demo runs on the bundled, pre-trained model and offers five pages:")
table([
    ["Page", "What it shows"],
    ["Overview", "Dataset size, churn rate, average tenure"],
    ["Churn Insights", "Churn distribution and rates by contract and internet service"],
    ["Make Prediction", "A form to score one customer, with probability, advice and the explanation"],
    ["Model Details", "Live metrics, model comparison and feature importance"],
    ["Data Overview", "The raw dataset, summary statistics and feature distributions"],
], [4.2*cm, 11.3*cm])
H2("4.2 Bring-Your-Own-Data")
P("The second mode turns the project into a reusable tool. The user uploads a CSV, picks "
  "the target column, and the system does the rest:")
bullets([
    "<b>Auto-detect</b> numeric vs categorical columns; drop ID-like and constant columns.",
    "<b>Guardrails</b> stop bad inputs: minimum rows, a strictly binary target, and a "
    "minimum minority-class size — a clear error rather than a silently bad model.",
    "<b>Train &amp; compare</b> Logistic Regression and Random Forest with cross-validation.",
    "<b>Inspect</b> a comparison table, confusion matrix and feature importance.",
    "<b>Predict</b> a single record via an auto-generated form, and <b>download</b> the "
    "trained pipeline as a portable .pkl (depends only on scikit-learn).",
])
P("The generic pipeline is as strong as the bespoke one: on the Telco data it reaches a "
  "cross-validated ROC-AUC of 0.846, essentially matching the hand-built version.")

# 5
H1("5. The Scoring API"); rule()
P("A FastAPI service exposes the same pipeline over HTTP, so other systems (a CRM, a batch "
  "job) can request scores. Requests are validated with Pydantic: invalid categories or "
  "out-of-range numbers are rejected automatically with a 422 error.")
table([
    ["Method", "Path", "Purpose"],
    ["GET", "/health", "Liveness and which model is loaded"],
    ["GET", "/model", "Model metadata (metrics, version, timestamp)"],
    ["POST", "/predict", "Score one customer"],
    ["POST", "/predict/batch", "Score up to 1,000 customers in one call"],
], [2.4*cm, 4.2*cm, 8.9*cm])
P("Example request and response:")
code('POST /predict\n'
     '{ "gender":"Male", "SeniorCitizen":1, "tenure":1, "Contract":"Month-to-month",\n'
     '  "InternetService":"Fiber optic", "MonthlyCharges":95.0, ... }\n\n'
     '200 OK\n'
     '{ "churn": true, "churn_probability": 0.9388, "risk": "high" }')

# 6
story.append(PageBreak())
H1("6. Architecture and Project Structure"); rule()
P("The code is organised so the heavy logic is testable independently of the user "
  "interface. The Streamlit app and the API are thin layers over the shared pipeline.")
table([
    ["Path", "Responsibility"],
    ["src/pipeline.py", "Feature engineering + the single fit-once/serve-once pipeline"],
    ["src/data_loader.py", "Load (with guardrail), clean and validate the dataset"],
    ["src/explain.py", "Per-customer explanations (exact linear SHAP values)"],
    ["src/automl.py", "Generic 'bring your own data' trainer with guardrails"],
    ["train_model.py", "Train, select, evaluate and persist the model + metadata"],
    ["app.py", "Streamlit web app (both modes, all pages)"],
    ["service/main.py", "FastAPI scoring service"],
    ["tests/", "21 automated tests (engine, app flow, API, explanations, AutoML)"],
    ["Dockerfile, .github/", "Container image and continuous integration"],
], [4.6*cm, 10.9*cm])

# 7
H1("7. Deployment"); rule()
P("<b>Web app (Streamlit Community Cloud, free).</b> Connect the GitHub repository, choose "
  "branch main and file app.py, select Python 3.12+, and deploy. The app loads the "
  "committed model, so no training runs on the server.")
P("<b>API (Docker, any container host).</b>")
code("docker build -t churn-api .\n"
     "docker run -p 8000:8000 churn-api    # POST to http://localhost:8000/predict")

# 8
H1("8. Quality: Testing and Continuous Integration"); rule()
P("21 automated tests run on every push and pull request via GitHub Actions. They go "
  "beyond \"does it run\" and assert real behaviour:")
bullets([
    "<b>Performance gate</b> — fails if test ROC-AUC drops below 0.78 or recall below 0.45, "
    "catching a model that has degenerated to the majority class.",
    "<b>Train/serve guard</b> — a high-risk profile must score clearly higher than a "
    "low-risk one, catching the historical input-dropping bug.",
    "<b>Explanation correctness</b> — the per-feature contributions must reconstruct the "
    "predicted probability exactly.",
    "<b>API</b> — health, prediction shape, validation errors (422) and batch scoring.",
    "<b>AutoML</b> — learns signal, drops ID-like columns, and enforces every guardrail.",
])
P("Before release the whole system was stress-tested end to end: deterministic training, "
  "twelve API edge cases, all app pages and the full upload-train-predict-download flow "
  "rendered without error, and the generic engine handled all-numeric, all-categorical, "
  "missing-value and unusual-target datasets.")

# 9
H1("9. Limitations and Roadmap"); rule()
P("Honest limitations, documented in the model card:")
bullets([
    "The model is trained on a single point-in-time snapshot; there is no drift monitoring "
    "or scheduled retraining yet.",
    "Precision is around 0.50 — roughly half of flagged customers would not actually churn. "
    "Acceptable for retention outreach, but not for punitive actions.",
    "gender is a feature; before any real deployment it should be checked for disparate "
    "impact (and possibly removed).",
])
P("Planned next steps: a live hosted demo, decision-threshold tuning to a business cost, "
  "additional algorithms (XGBoost / LightGBM), a fairness check, and model registry / drift "
  "detection for true production operation.")

# 10
H1("10. Glossary"); rule()
table([
    ["Term", "Plain-English meaning"],
    ["Churn", "A customer cancelling their subscription."],
    ["ROC-AUC", "How well the model ranks churners above non-churners (0.5 = random, 1.0 = perfect)."],
    ["Recall", "Of the customers who actually churn, the share the model catches."],
    ["Precision", "Of the customers the model flags, the share that actually churn."],
    ["F1-score", "A single balance between precision and recall."],
    ["Pipeline", "One object that bundles all data transformations with the model."],
    ["One-hot encoding", "Turning a category (e.g. contract type) into numeric yes/no columns."],
    ["SHAP value", "Each feature's contribution to a single prediction."],
    ["Cross-validation", "Rotating train/validation splits to estimate performance without using the test set."],
], [4.2*cm, 11.3*cm])
gap(14)
story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#d5dbdb")))
gap(6)
P("<font color='#566573' size=9>Customer Churn Prediction &middot; Documentation v2.0 &middot; "
  "github.com/dinfalabs/customer-churn-prediction &middot; Generated " +
  datetime.date.today().strftime("%d %B %Y") + "</font>")

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8); canvas.setFillColor(GREY)
    if doc.page > 1:
        canvas.drawString(2*cm, 1.1*cm, "Customer Churn Prediction — Documentation")
        canvas.drawRightString(A4[0]-2*cm, 1.1*cm, "Page %d" % doc.page)
        canvas.setStrokeColor(colors.HexColor("#d5dbdb"))
        canvas.line(2*cm, 1.45*cm, A4[0]-2*cm, 1.45*cm)
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                        leftMargin=2*cm, rightMargin=2*cm,
                        title="Customer Churn Prediction — Documentation", author="Davide Infantino")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Wrote", OUT)
