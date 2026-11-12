from pathlib import Path
import tempfile
import joblib
import pandas as pd
import streamlit as st
from src.preprocessing import prepare_xy
from src.modeling import train_and_evaluate, predict_one, save_pipeline

st.set_page_config(page_title="Project Sentinel", page_icon="☄️", layout="wide")
st.title("☄️ Project Sentinel")
st.subheader("Hazardous Asteroid Prediction")
st.caption("Educational classifier for historical NEO dataset labels — not an official NASA/JPL risk assessment.")

uploaded = st.file_uploader("Upload the NEO dataset CSV", type=["csv"])
model_path = Path("models/asteroid_pipeline.joblib")
if "trained_model" not in st.session_state and model_path.exists():
    try:
        saved = joblib.load(model_path)
        st.session_state.trained_model = saved["pipeline"]
        st.session_state.metadata = saved["metadata"]
    except Exception as exc:
        st.warning(f"Could not load saved model: {exc}")

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        st.write(f"Dataset shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
        st.dataframe(raw.head(), use_container_width=True)
        if st.button("Train / evaluate model", type="primary"):
            with st.spinner("Training and evaluating models. This may take a few minutes..."):
                X, y = prepare_xy(raw)
                model, details = train_and_evaluate(X, y)
                st.session_state.trained_model = model
                st.session_state.metadata = details
                model_path.parent.mkdir(exist_ok=True)
                save_pipeline(model, details, model_path)
            st.success("Training complete. Model saved locally.")
    except Exception as exc:
        st.error(f"Dataset could not be processed: {exc}")

if "metadata" in st.session_state:
    meta = st.session_state.metadata
    st.markdown("### Evaluation")
    st.write("Selected model:", meta["selected_model"])
    st.json(meta["metrics"])
    st.download_button("Download trained model", model_path.read_bytes(),
                       file_name="asteroid_pipeline.joblib",
                       mime="application/octet-stream")
    st.markdown("### Try a prediction")
    cols = st.columns(2)
    values = {}
    for i, col in enumerate(meta["feature_columns"]):
        with cols[i % 2]:
            values[col] = st.number_input(col, value=0.0, format="%.8f", key=f"input_{col}")
    if st.button("Predict asteroid label"):
        try:
            result = predict_one(st.session_state.trained_model, values, meta["feature_columns"])
            st.subheader(result["prediction"])
            if result["hazardous_class_probability_score"] is not None:
                st.metric("Hazardous-class model score", f"{result['hazardous_class_probability_score']:.2%}")
            st.info("This is a model score for the dataset label, not an impact probability.")
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("Upload a compatible CSV and train the model, or place a trained model at models/asteroid_pipeline.joblib.")
