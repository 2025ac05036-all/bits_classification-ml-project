import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="BITS Classification ML Project", page_icon="🤖", layout="wide")

st.title("🤖 BITS Classification ML Project")
st.subheader("Machine Learning Model Comparison Dashboard")

st.markdown(
    """
    This application compares five supervised machine-learning classifiers
    on the **Breast Cancer Wisconsin (Diagnostic) dataset**.

    Upload the test dataset, select a model, and evaluate its performance
    using **Accuracy, AUC, Precision, Recall, F1 Score, and MCC**.
    """
)

st.markdown("---")

MODEL_PATHS = {
    "Logistic Regression": "Model/logistic_regression.pkl",
    "Decision Tree": "Model/decision_tree.pkl",
    "K-Nearest Neighbors": "Model/knearest_neighbors.pkl",
    "Gaussian Naive Bayes": "Model/gaussian_naive_bayes.pkl",
    "Random Forest": "Model/random_forest.pkl",
}

TARGET_COLUMN = "target"

FEATURE_COLUMNS = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

CLASS_NAMES = {0: "Malignant", 1: "Benign"}

@st.cache_resource
def load_models():
    loaded = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded

models = load_models()

if not models:
    st.error("No trained models found. Check that all .pkl files are inside the Model/ folder.")
    st.stop()

st.sidebar.title("📌 Project Information")

st.sidebar.markdown(
    """
    **Dataset**  
    Breast Cancer Wisconsin (Diagnostic)

    **Models implemented**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Gaussian Naive Bayes
    - Random Forest

    **Evaluation metrics**
    - Accuracy
    - AUC
    - Precision
    - Recall
    - F1 Score
    - MCC
    """
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Model Selection")

selected_model_name = st.sidebar.selectbox("Select a classification model:", list(models.keys()))
selected_model = models[selected_model_name]

st.header("1. Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Upload the test dataset containing the 30 features and target column."
)

if uploaded_file is None:
    st.info("Please upload test_data.csv to calculate evaluation metrics.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the uploaded CSV: {e}")
    st.stop()

missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]

if TARGET_COLUMN not in df.columns:
    st.error("The uploaded CSV must contain a 'target' column.")
    st.stop()

if missing_features:
    st.error("The uploaded CSV is missing required features:")
    st.write(missing_features)
    st.stop()

X_test = df[FEATURE_COLUMNS].copy()
y_test = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

if y_test.isna().any():
    st.error("The target column must contain numeric labels 0 and 1.")
    st.stop()

if not set(y_test.unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 and 1.")
    st.stop()

st.header("2. Uploaded Test Data")
c1, c2, c3 = st.columns(3)
c1.metric("Test Instances", len(df))
c2.metric("Features", len(FEATURE_COLUMNS))
c3.metric("Classes", y_test.nunique())

with st.expander("View uploaded test data"):
    st.dataframe(df, use_container_width=True)

st.header("3. Model Evaluation")

try:
    y_pred = selected_model.predict(X_test)
    if hasattr(selected_model, "predict_proba"):
        y_score = selected_model.predict_proba(X_test)[:, 1]
    elif hasattr(selected_model, "decision_function"):
        y_score = selected_model.decision_function(X_test)
    else:
        y_score = y_pred
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_score),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Performance Metrics — {selected_model_name}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

st.subheader("Confusion Matrix and Classification Report")
cm_col, report_col = st.columns(2)

with cm_col:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"]
    )
    st.dataframe(cm_df, use_container_width=True)

with report_col:
    st.write("### Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=["Malignant", "Benign"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"),
                 use_container_width=True)

st.subheader("Predictions on Test Data")
prediction_df = X_test.copy()
prediction_df["Actual Class"] = y_test.map(CLASS_NAMES)
prediction_df["Predicted Class"] = pd.Series(y_pred, index=X_test.index).map(CLASS_NAMES)
if hasattr(selected_model, "predict_proba"):
    prediction_df["Prediction Probability"] = y_score
st.dataframe(prediction_df, use_container_width=True)

st.header("4. Comparison of All Models")
comparison_results = []

for model_name, model in models.items():
    try:
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_test)
        else:
            score = pred

        comparison_results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, score),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })
    except Exception as e:
        st.warning(f"Could not evaluate {model_name}: {e}")

comparison_df = pd.DataFrame(comparison_results)

if not comparison_df.empty:
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1 Score": "{:.4f}", "MCC": "{:.4f}"
        }),
        use_container_width=True
    )
    st.write("### Metric Comparison")
    st.bar_chart(comparison_df.set_index("Model"))

st.markdown("---")
st.caption("Breast Cancer Wisconsin (Diagnostic) Dataset | Machine Learning Classification Project")
