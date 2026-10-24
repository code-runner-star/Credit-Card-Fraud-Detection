import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    .main {padding-top: 1.2rem;}
    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .hero {
        padding: 1.25rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(120deg, rgba(55, 95, 160, .18), rgba(40, 160, 145, .10));
        border: 1px solid rgba(120, 120, 120, .22);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Credit Card Fraud Detection")
st.caption("Machine-learning demo for identifying transactions that may need further review.")

# -------------------- Load saved model --------------------
@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

MODEL_PATH = "fraud_detection_model.pkl"

try:
    artifact = load_artifact(MODEL_PATH)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        threshold = float(artifact.get("threshold", 0.5))
        model_name = artifact.get("model_name", "Saved machine-learning model")
        saved_features = artifact.get("features")
    else:
        model = artifact
        threshold = 0.5
        model_name = type(model).__name__
        saved_features = None

    if model is None:
        raise ValueError("The saved artifact does not contain a model under the 'model' key.")

except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.write("Check that `fraud_detection_model.pkl` is in the same folder as `app.py` and that your local scikit-learn version matches the version used in Google Colab.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

# Dataset feature names used by the common Kaggle credit-card fraud dataset.
DEFAULT_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def get_expected_features(model, saved_features):
    if saved_features is not None:
        return list(saved_features)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Pipelines sometimes expose feature names through the preprocessing step.
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return DEFAULT_FEATURES

FEATURES = get_expected_features(model, saved_features)

def prepare_features(frame):
    """Validate and arrange input columns in the order expected by the model."""
    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        raise ValueError(
            "The uploaded file is missing required model columns: "
            + ", ".join(missing[:12])
            + (" ..." if len(missing) > 12 else "")
        )
    X = frame[FEATURES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError("Some required feature values are blank or non-numeric in: " + ", ".join(bad_cols[:12]))
    return X

def get_probabilities(X):
    """Return fraud probabilities when the estimator supports predict_proba."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        # Pipelines commonly expose classes_ on the pipeline itself.
        fraud_index = classes.index(1) if 1 in classes else (probabilities.shape[1] - 1)
        return probabilities[:, fraud_index]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        # Convert scores to a rough 0-1 display value. This is not a calibrated probability.
        return 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
    return None

def predict_frame(frame):
    X = prepare_features(frame)
    probabilities = get_probabilities(X)
    predictions = model.predict(X).astype(int)
    if probabilities is None:
        probabilities = np.full(len(X), np.nan)
        labels = np.where(predictions == 1, "Flagged for review", "Not flagged")
    else:
        predictions = (probabilities >= threshold).astype(int)
        labels = np.where(predictions == 1, "Flagged for review", "Lower predicted risk")
    result = frame.copy()
    result["Fraud_Probability"] = probabilities
    result["Prediction"] = labels
    return result, X, predictions, probabilities

# -------------------- Sidebar / navigation --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard & Upload", "Single Transaction", "Model Evaluation", "About Project"],
)
st.sidebar.divider()
st.sidebar.write("**Model:**", str(model_name))
st.sidebar.write("**Decision threshold:**", f"{threshold:.3f}")
st.sidebar.caption("A flagged result is a model estimate, not proof of fraud.")

if page == "Dashboard & Upload":
    st.markdown(
        '<div class="hero"><h3>Analyze transaction files</h3>'
        '<p>Upload a CSV to receive model predictions, review risk scores, explore patterns, and download results.</p></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload a transaction CSV file", type=["csv"], key="batch_upload")
    if uploaded is None:
        st.info("Upload a CSV containing the feature columns used during training. The original Kaggle file `creditcard.csv` is suitable if its columns match your model.")
        st.markdown("**Expected common dataset columns:** `Time`, `V1`–`V28`, and `Amount`. The `Class` column is optional and is used only for evaluation when present.")
    else:
        try:
            data = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(data):,} rows** and **{len(data.columns)} columns**.")
            with st.spinner("Analyzing transactions..."):
                results, X, predictions, probabilities = predict_frame(data)

            total = len(results)
            flagged = int((predictions == 1).sum())
            low_risk = total - flagged
            flagged_pct = 100 * flagged / total if total else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Transactions analyzed", f"{total:,}")
            m2.metric("Flagged for review", f"{flagged:,}")
            m3.metric("Not flagged", f"{low_risk:,}")
            m4.metric("Flag rate", f"{flagged_pct:.2f}%")

            tab1, tab2, tab3 = st.tabs(["Results", "Charts", "Evaluation (if labels exist)"])
            with tab1:
                st.subheader("Prediction results")
                filter_choice = st.selectbox("Filter transactions", ["All transactions", "Flagged for review", "Lower predicted risk"])
                view = results
                if filter_choice == "Flagged for review":
                    view = results[results["Prediction"] == "Flagged for review"]
                elif filter_choice == "Lower predicted risk":
                    view = results[results["Prediction"] == "Lower predicted risk"]

                search_text = st.text_input("Optional: search a value in the displayed rows")
                if search_text.strip():
                    view = view[
                        view.astype(str).apply(
                            lambda col: col.str.contains(search_text, case=False, na=False)
                        ).any(axis=1)
                    ]
                st.caption(f"Showing {len(view):,} of {len(results):,} rows.")
                st.dataframe(view.head(5000), use_container_width=True, height=420)
                if len(view) > 5000:
                    st.info("The table is limited to the first 5,000 matching rows for responsiveness. The download includes all matching rows.")
                csv_bytes = view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download displayed results as CSV",
                    data=csv_bytes,
                    file_name="fraud_predictions_filtered.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download all predictions as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

            with tab2:
                left, right = st.columns(2)
                with left:
                    st.subheader("Prediction breakdown")
                    counts = results["Prediction"].value_counts().reindex(
                        ["Lower predicted risk", "Flagged for review"], fill_value=0
                    )
                    fig, ax = plt.subplots()
                    ax.bar(counts.index, counts.values)
                    ax.set_ylabel("Number of transactions")
                    ax.tick_params(axis="x", rotation=10)
                    st.pyplot(fig)
                    plt.close(fig)
                with right:
                    st.subheader("Fraud probability distribution")
                    valid_probs = results["Fraud_Probability"].dropna()
                    if len(valid_probs):
                        fig, ax = plt.subplots()
                        ax.hist(valid_probs, bins=30)
                        ax.set_xlabel("Model fraud score / probability")
                        ax.set_ylabel("Number of transactions")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("This model does not expose probability scores.")

                if "Amount" in results.columns:
                    st.subheader("Transaction amount by prediction")
                    fig, ax = plt.subplots()
                    for label in ["Lower predicted risk", "Flagged for review"]:
                        vals = pd.to_numeric(
                            results.loc[results["Prediction"] == label, "Amount"], errors="coerce"
                        ).dropna()
                        if len(vals):
                            ax.hist(vals, bins=30, alpha=0.55, label=label)
                    ax.set_xlabel("Amount")
                    ax.set_ylabel("Transactions")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

            with tab3:
                if "Class" not in data.columns:
                    st.info("No `Class` column was found. This is normal for new, unlabeled transactions. To evaluate performance, use a held-out test set with known labels—not training data.")
                else:
                    y_true = pd.to_numeric(data["Class"], errors="coerce")
                    valid = y_true.isin([0, 1])
                    if not valid.all():
                        st.warning("Some Class values are not 0 or 1, so they were excluded from evaluation.")
                    y = y_true[valid].astype(int).to_numpy()
                    p = predictions[valid.to_numpy()]
                    tp = int(((y == 1) & (p == 1)).sum())
                    tn = int(((y == 0) & (p == 0)).sum())
                    fp = int(((y == 0) & (p == 1)).sum())
                    fn = int(((y == 1) & (p == 0)).sum())
                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall = tp / (tp + fn) if (tp + fn) else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                    accuracy = (tp + tn) / len(y) if len(y) else 0.0
                    a, b, c, d = st.columns(4)
                    a.metric("Accuracy", f"{accuracy:.3f}")
                    b.metric("Precision", f"{precision:.3f}")
                    c.metric("Recall", f"{recall:.3f}")
                    d.metric("F1 score", f"{f1:.3f}")
                    st.write("Confusion matrix (rows = actual, columns = predicted)")
                    cm = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=["Actual legitimate (0)", "Actual fraud (1)"],
                        columns=["Predicted legitimate (0)", "Predicted fraud (1)"],
                    )
                    st.dataframe(cm, use_container_width=True)
                    st.warning("These metrics are trustworthy as a generalization estimate only when this file is a separate held-out test set that was not used for training or threshold selection.")

        except Exception as exc:
            st.error("I couldn't analyze this CSV.")
            st.write("Check that the file is a CSV and contains all required model input columns.")
            st.code(f"{type(exc).__name__}: {exc}")

elif page == "Single Transaction":
    st.markdown(
        '<div class="hero"><h3>Check one transaction</h3>'
        '<p>Enter all required model features to get a prediction. This is a demonstration, not a banking decision.</p></div>',
        unsafe_allow_html=True,
    )
    st.caption("The form shows the features your saved model expects. For this dataset, V1–V28 are transformed features; do not invent values for a real financial decision.")
    with st.form("single_transaction_form"):
        values = {}
        columns = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with columns[i % 3]:
                default = 0.0
                if feature == "Amount":
                    default = 1.0
                values[feature] = st.number_input(feature, value=float(default), format="%.6f", key=f"single_{feature}")
        submitted = st.form_submit_button("Predict transaction", type="primary")
    if submitted:
        try:
            one = pd.DataFrame([values], columns=FEATURES)
            _, _, one_pred, one_prob = predict_frame(one)
            if one_pred[0] == 1:
                st.error("Flagged for review by the model")
            else:
                st.success("Not flagged by the model")
            if len(one_prob) and np.isfinite(one_prob[0]):
                st.metric("Estimated fraud probability / score", f"{one_prob[0]:.2%}")
            st.caption("This prediction is only as meaningful as the input data and model calibration. It is not proof of fraud or legitimacy.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Model Evaluation":
    st.header("Model evaluation")
    st.write("This page explains what the evaluation metrics mean. For actual scores, upload a separate labeled test CSV under **Dashboard & Upload**.")
    st.markdown(
        """
        - **Precision:** Of transactions flagged by the model, the proportion that were actually fraud.
        - **Recall:** Of all actually fraudulent transactions, the proportion detected by the model.
        - **F1 score:** A combined measure of precision and recall.
        - **Confusion matrix:** Counts correct predictions, false alarms, and missed fraud.
        - **ROC-AUC / Average precision:** Useful ranking metrics; average precision is especially informative when fraud is rare.
        """
    )
    st.warning("Do not report metrics computed on the training dataset as evidence of performance on new transactions. Use the held-out test set from your Colab evaluation.")

    st.subheader("Feature importance (when supported)")
    estimator = model
    if hasattr(model, "named_steps"):
        # Explain the final estimator in a sklearn pipeline.
        estimator = list(model.named_steps.values())[-1]
    importance = None
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        importance = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    if importance is not None and len(importance) == len(FEATURES):
        fi = pd.DataFrame({"Feature": FEATURES, "Importance": importance}).sort_values("Importance", ascending=False).head(15)
        st.bar_chart(fi.set_index("Feature"))
        st.caption("These are model-specific importance indicators, not proof of causation.")
    else:
        st.info("This model doesn't expose a directly supported feature-importance vector. Some model types require additional explainability tools.")

elif page == "About Project":
    st.header("About this project")
    st.write(
        "This educational application uses a saved machine-learning model to estimate which credit-card transactions may be suspicious. "
        "It demonstrates data upload, prediction, filtering, visualization, and evaluation."
    )
    st.subheader("Technology used")
    st.markdown("- Python\n- pandas and NumPy\n- scikit-learn\n- joblib\n- Streamlit\n- Matplotlib")
    st.subheader("Important limitations")
    
    st.markdown(
        """
        - A model prediction is not proof that a transaction is fraudulent.
        - Historical dataset performance may not match performance on future transactions.
        - The sample Kaggle dataset uses transformed `V1`–`V28` features; these are not ordinary, human-readable transaction details.
        - Do not upload real cardholder information or sensitive banking data to this demo.
        """
    )

st.divider()
st.caption("Educational prototype • Review flagged transactions rather than treating predictions as definitive decisions.")
