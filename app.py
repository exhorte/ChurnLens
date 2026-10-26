"""
Customer Churn Prediction - Streamlit Application

A web application for predicting customer churn using machine learning.

Run this app with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import io
import joblib
import json
import logging
import os
import sys
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_telco_data
# FeatureEngineer must be importable so joblib can reconstruct the saved pipeline.
from src.pipeline import FeatureEngineer  # noqa: F401
from src.explain import explain_customer, supports_explanation
from src.automl import train_on_dataframe, DataValidationError

# Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-card {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the trained end-to-end pipeline and its metadata.

    Returns:
        tuple: (pipeline, metadata_dict). pipeline is None if not yet trained.
    """
    try:
        pipeline = joblib.load('models/churn_pipeline.pkl')
    except FileNotFoundError:
        st.error("❌ Trained model not found.")
        st.info("Please run `python train_model.py` first to train and save the model.")
        return None, {}

    metadata = {}
    try:
        with open('models/metadata.json') as fh:
            metadata = json.load(fh)
    except FileNotFoundError:
        pass
    return pipeline, metadata


@st.cache_data
def load_dataset():
    """Load the dataset for analysis."""
    try:
        df = load_telco_data('data/WA_Fn-UseC_-_Telco_Customer_Churn.csv')
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None


def page_overview():
    """Home/Overview page."""
    st.markdown("<h1 class='main-header'>🔮 Customer Churn Prediction</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Customer Churn Prediction System
    
    This application uses **machine learning** to predict which customers are likely to churn
    (cancel their subscription) based on their characteristics and behavior.
    
    **Key Features:**
    - 📊 **Project Overview** - Understand the dataset and business problem
    - 📈 **Churn Insights** - Explore patterns in customer churn
    - 🎯 **Make Predictions** - Predict churn for individual customers
    - 📋 **Model Details** - Learn about the model's performance
    """)
    
    # Key metrics
    st.markdown("### 📊 Dataset Overview")
    
    df = load_dataset()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            churn_count = (df['Churn'] == 'Yes').sum()
            st.metric("Churned Customers", f"{churn_count:,}")
        
        with col3:
            churn_rate = (df['Churn'] == 'Yes').sum() / len(df)
            st.metric("Churn Rate", f"{churn_rate:.2%}")
        
        with col4:
            avg_tenure = df['tenure'].mean()
            st.metric("Avg. Tenure", f"{avg_tenure:.1f} months")
    
    st.markdown("---")
    
    # Navigation instructions
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Explore Insights** - Navigate to the "Churn Insights" tab to analyze churn patterns
    2. **Make Predictions** - Go to the "Make Prediction" tab to predict churn for a customer
    3. **View Model Details** - Check the "Model Details" tab for performance metrics
    4. **Review Data** - Visit the "Data Overview" tab to explore the full dataset
    """)


def page_churn_insights():
    """Churn insights and analysis page."""
    st.markdown("<h1 class='main-header'>📈 Churn Insights</h1>", unsafe_allow_html=True)
    
    df = load_dataset()
    if df is None:
        return
    
    # Churn distribution
    st.markdown("### Churn Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        churn_counts = df['Churn'].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=['No Churn', 'Churn'],
                values=[churn_counts['No'], churn_counts['Yes']],
                marker=dict(colors=['#2ecc71', '#e74c3c']),
                hole=0.3
            )
        ])
        fig.update_layout(
            title="Churn Distribution",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        churn_pct = (df['Churn'].value_counts() / len(df) * 100).round(2)
        metrics_data = {
            'Status': ['No Churn', 'Churn'],
            'Count': [churn_counts['No'], churn_counts['Yes']],
            'Percentage': [f"{churn_pct['No']:.1f}%", f"{churn_pct['Yes']:.1f}%"]
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, width="stretch")
    
    st.markdown("---")
    
    # Churn by key factors
    st.markdown("### Churn by Key Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Contract' in df.columns:
            contract_churn = pd.crosstab(df['Contract'], df['Churn'])
            contract_churn_pct = (contract_churn['Yes'] / (contract_churn['No'] + contract_churn['Yes']) * 100).round(2)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=contract_churn_pct.index,
                    y=contract_churn_pct.values,
                    marker=dict(color='#e74c3c'),
                    text=[f"{v:.1f}%" for v in contract_churn_pct.values],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Churn Rate by Contract Type",
                xaxis_title="Contract Type",
                yaxis_title="Churn Rate (%)",
                height=400
            )
            st.plotly_chart(fig, width="stretch")
    
    with col2:
        if 'InternetService' in df.columns:
            internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
            internet_churn_pct = (internet_churn['Yes'] / (internet_churn['No'] + internet_churn['Yes']) * 100).round(2)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=internet_churn_pct.index,
                    y=internet_churn_pct.values,
                    marker=dict(color='#9b59b6'),
                    text=[f"{v:.1f}%" for v in internet_churn_pct.values],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Churn Rate by Internet Service",
                xaxis_title="Internet Service",
                yaxis_title="Churn Rate (%)",
                height=400
            )
            st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Tenure analysis
    st.markdown("### Tenure Impact on Churn")
    
    fig = go.Figure()
    
    for churn_status in ['No', 'Yes']:
        fig.add_trace(go.Histogram(
            x=df[df['Churn'] == churn_status]['tenure'],
            name='Churn' if churn_status == 'Yes' else 'No Churn',
            opacity=0.7,
            marker=dict(color='#e74c3c' if churn_status == 'Yes' else '#2ecc71')
        ))
    
    fig.update_layout(
        title="Tenure Distribution by Churn Status",
        xaxis_title="Tenure (months)",
        yaxis_title="Number of Customers",
        barmode='overlay',
        height=400
    )
    st.plotly_chart(fig, width="stretch")


def page_prediction():
    """Customer churn prediction page."""
    st.markdown("<h1 class='main-header'>🎯 Make Prediction</h1>", unsafe_allow_html=True)
    
    pipeline, metadata = load_pipeline()
    if pipeline is None:
        return

    st.markdown("### Enter Customer Information")

    # Create input form (all 19 model features are collected)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("More Services & Billing")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", 18.25, 118.75, 65.0, 0.25)
    with col2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 8684.8, 1500.0, 10.0)

    st.markdown("---")

    # Make prediction
    if st.button("🔮 Predict Churn", width="stretch", key="predict_btn"):

        # Build the customer record with the exact raw schema the pipeline expects
        customer_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
        }

        try:
            # One call: the fitted pipeline does FE + encoding + scaling + predict.
            probability = pipeline.predict_proba(pd.DataFrame([customer_data]))[0]
            prediction = int(probability[1] >= 0.5)

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")
            
            if prediction == 1:
                churn_status = "⚠️ **HIGH RISK** - Likely to Churn"
                color = "#e74c3c"
                prob_label = "Churn Probability"
                prob_value = probability[1]
            else:
                churn_status = "✅ **LOW RISK** - Unlikely to Churn"
                color = "#2ecc71"
                prob_label = "Retention Probability"
                prob_value = probability[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{churn_status}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric(prob_label, f"{prob_value:.1%}")
            
            # Probability breakdown
            st.markdown("### Prediction Confidence")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Churn', 'Churn'],
                    y=[probability[0], probability[1]],
                    marker=dict(color=['#2ecc71', '#e74c3c']),
                    text=[f"{probability[0]:.1%}", f"{probability[1]:.1%}"],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Prediction Probability",
                yaxis_title="Probability",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, width="stretch")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if prediction == 1:
                st.warning("""
                **This customer is at risk of churning. Consider:**
                - Offering special promotions or discounts
                - Contacting the customer to address concerns
                - Improving service quality or support
                - Providing personalized offers based on their usage
                """)
            else:
                st.success("""
                **This customer is satisfied. Continue to:**
                - Maintain current service quality
                - Monitor for any changes in behavior
                - Offer relevant upsell opportunities
                - Encourage loyalty programs
                """)

            # Per-customer explanation (linear SHAP values)
            if supports_explanation(pipeline):
                st.markdown("### 🔍 Why this prediction?")
                contrib = explain_customer(
                    pipeline, pd.DataFrame([customer_data])
                ).head(8).iloc[::-1]
                exp_fig = go.Figure(go.Bar(
                    x=contrib["contribution"], y=contrib["feature"], orientation="h",
                    marker=dict(color=["#e74c3c" if c > 0 else "#2ecc71"
                                       for c in contrib["contribution"]]),
                ))
                exp_fig.update_layout(
                    title="Feature contributions (red = increases churn risk, green = reduces)",
                    xaxis_title="Contribution to churn log-odds",
                    height=380, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(exp_fig, width="stretch")
                st.caption("Exact contributions from the linear model (linear SHAP values): "
                           "with the intercept they sum to the predicted churn log-odds.")

        except Exception:
            st.error("❌ Unable to generate a prediction for these inputs. "
                     "Please review the values and try again.")
            logging.exception("Prediction failed")


def page_model_details():
    """Model performance details page."""
    st.markdown("<h1 class='main-header'>📋 Model Details</h1>", unsafe_allow_html=True)

    # Live metrics from the deployed model's metadata
    _, metadata = load_pipeline()
    if metadata:
        st.markdown(f"### Deployed Model: `{metadata.get('model', 'n/a')}`")
        tm = metadata.get('test_metrics', {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{tm.get('roc_auc', float('nan')):.3f}")
        c2.metric("F1-Score", f"{tm.get('f1', float('nan')):.3f}")
        c3.metric("Recall", f"{tm.get('recall', float('nan')):.3f}")
        c4.metric("Precision", f"{tm.get('precision', float('nan')):.3f}")
        st.caption(
            f"Trained: {metadata.get('trained_at', 'n/a')} · "
            f"scikit-learn {metadata.get('sklearn_version', 'n/a')} · "
            f"{metadata.get('n_train', 0):,} train / {metadata.get('n_test', 0):,} test samples"
        )
        st.markdown("---")

    # Load comparison data
    comparison_file = 'reports/model_comparison.csv'
    if os.path.exists(comparison_file):
        comparison_df = pd.read_csv(comparison_file)
        
        st.markdown("### Model Performance Comparison")
        st.dataframe(comparison_df, hide_index=True, width="stretch")
    
    # Load feature importance
    importance_file = 'reports/feature_importance.csv'
    if os.path.exists(importance_file):
        st.markdown("---")
        st.markdown("### Top 15 Feature Importance")
        
        importance_df = pd.read_csv(importance_file)
        
        fig = go.Figure(data=[
            go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker=dict(color='#3498db')
            )
        ])
        fig.update_layout(
            title=f"Permutation Feature Importance — {metadata.get('model', 'model')}",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("**Interpretation:**")
        top_features = importance_df.head(3)['Feature'].tolist()
        st.info(f"""
        The most important features for predicting churn are:
        1. {top_features[0]}
        2. {top_features[1]}
        3. {top_features[2]}
        
        These features have the greatest influence on the model's predictions.
        """)


def page_data_overview():
    """Data overview page."""
    st.markdown("<h1 class='main-header'>📊 Data Overview</h1>", unsafe_allow_html=True)
    
    df = load_dataset()
    if df is None:
        return
    
    st.markdown("### Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), width="stretch")
    
    st.markdown("---")
    
    st.markdown("### Numerical Features Summary")
    st.dataframe(df.describe(), width="stretch")
    
    st.markdown("---")
    
    st.markdown("### Categorical Features")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        st.markdown(f"**{col}**")
        value_counts = df[col].value_counts()
        st.dataframe(value_counts, width="stretch")


def page_train_your_own():
    """Bring-your-own-data training page (AutoML-lite)."""
    st.markdown("<h1 class='main-header'>📤 Train on Your Own Data</h1>", unsafe_allow_html=True)
    st.markdown("""
    Upload **any churn dataset** (CSV) with a **binary target** column. The app
    auto-detects feature types, trains and compares two models, then lets you
    predict and download the trained model. The data stays in this session.
    """)

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("👆 Upload a CSV to begin — or switch to **Demo (Telco)** in the sidebar.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not read this file as CSV. Please check the format.")
        return
    if df.shape[1] < 2:
        st.error("The dataset needs at least 2 columns (features + a target).")
        return

    st.markdown(f"**Loaded:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(8), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("🎯 Target column (what to predict)",
                              options=list(df.columns), index=len(df.columns) - 1)
    with col2:
        exclude = st.multiselect("Columns to ignore (optional)",
                                 options=[c for c in df.columns if c != target])

    if st.button("🚀 Train models", width="stretch", type="primary"):
        with st.spinner("Training and evaluating models…"):
            try:
                st.session_state["byod"] = train_on_dataframe(df, target, exclude=exclude)
            except DataValidationError as exc:
                st.session_state.pop("byod", None)
                st.error(f"❌ {exc}")
                return
            except Exception:
                st.session_state.pop("byod", None)
                st.error("❌ Training failed unexpectedly. Try a cleaner dataset.")
                logging.exception("BYOD training failed")
                return

    result = st.session_state.get("byod")
    if result is None:
        return

    st.success(f"✅ Trained. Best model: **{result.best_model}** "
               f"(positive class = `{result.positive_label}`).")

    st.markdown("### Model comparison")
    rows = []
    for name, r in result.results.items():
        m = r["metrics"]
        rows.append({"Model": name, "CV ROC-AUC": round(r["cv_roc_auc"], 3),
                     "Test ROC-AUC": round(m["roc_auc"], 3), "Recall": round(m["recall"], 3),
                     "Precision": round(m["precision"], 3), "F1": round(m["f1"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if result.dropped:
        st.caption("Ignored columns: " + ", ".join(f"`{c}` ({why})" for c, why in result.dropped))

    best = result.results[result.best_model]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Confusion matrix")
        cm = best["confusion_matrix"]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
            colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}"))
        fig.update_layout(height=360, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("### Feature importance")
        imp = best.get("importance")
        if imp is not None and len(imp):
            top = imp.head(12).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"],
                                   orientation="h", marker=dict(color="#3498db")))
            fig.update_layout(height=360, xaxis_title="Permutation importance")
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### 🎯 Predict a single record")
    inputs = {}
    cols = st.columns(3)
    for i, (col, meta) in enumerate(result.schema.items()):
        with cols[i % 3]:
            if meta["type"] == "numeric":
                inputs[col] = st.number_input(col, value=float(meta["median"]), key=f"byod_in_{col}")
            else:
                inputs[col] = st.selectbox(col, meta["choices"], key=f"byod_in_{col}")
    if st.button("Predict", key="byod_predict"):
        try:
            proba = float(result.best_pipeline.predict_proba(pd.DataFrame([inputs]))[0, 1])
            st.metric(f"Probability of `{result.positive_label}`", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))
        except Exception:
            st.error("Could not score this record with the given inputs.")
            logging.exception("BYOD predict failed")

    st.markdown("---")
    buffer = io.BytesIO()
    joblib.dump(result.best_pipeline, buffer)
    st.download_button("⬇️ Download trained model (.pkl)", data=buffer.getvalue(),
                       file_name="churn_model.pkl", mime="application/octet-stream")
    st.caption("Reload later with `joblib.load(...)` and call `.predict_proba(your_dataframe)`.")


# Sidebar navigation
def main():
    """Main app function."""
    st.sidebar.markdown("<h1 style='text-align: center'>Navigation</h1>", unsafe_allow_html=True)

    mode = st.sidebar.radio("Mode", ["🎯 Demo (Telco)", "📤 Your own data"], key="app_mode")
    st.sidebar.markdown("---")

    if mode == "📤 Your own data":
        page_train_your_own()
    else:
        page = st.sidebar.radio(
            "Select a page:",
            ["🏠 Overview", "📈 Churn Insights", "🎯 Make Prediction",
             "📋 Model Details", "📊 Data Overview"]
        )

        if page == "🏠 Overview":
            page_overview()
        elif page == "📈 Churn Insights":
            page_churn_insights()
        elif page == "🎯 Make Prediction":
            page_prediction()
        elif page == "📋 Model Details":
            page_model_details()
        elif page == "📊 Data Overview":
            page_data_overview()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📌 Project Info
    
    **Customer Churn Prediction**
    
    A machine learning project to predict which telecom customers are likely to churn.
    
    **Technologies:**
    - Python
    - Scikit-learn
    - Streamlit
    - Plotly
    
    **Dataset:** Telco Customer Churn
    """)


if __name__ == "__main__":
    main()
