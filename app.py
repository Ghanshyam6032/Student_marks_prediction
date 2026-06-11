import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.result-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    background-color: #1f2937;
    color: white;
    margin-top: 20px;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Prediction System")
st.markdown("Predict whether a student will PASS or FAIL using Machine Learning.")

# ---------------- LOAD FILES ----------------
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📊 Dashboard")
    st.info("Fill student details and click Predict.")
    st.success("Model Loaded Successfully")

# ---------------- INPUT FORM ----------------
st.subheader("📝 Student Information")

input_data = {}

for i in range(0, len(feature_columns), 2):

    col1, col2 = st.columns(2)

    field1 = feature_columns[i]

    with col1:
        if field1 in label_encoders:
            input_data[field1] = st.selectbox(
                field1.replace("_", " ").title(),
                label_encoders[field1].classes_,
                key=field1
            )
        else:
            input_data[field1] = st.number_input(
                field1.replace("_", " ").title(),
                value=50.0,
                key=field1
            )

    if i + 1 < len(feature_columns):

        field2 = feature_columns[i + 1]

        with col2:
            if field2 in label_encoders:
                input_data[field2] = st.selectbox(
                    field2.replace("_", " ").title(),
                    label_encoders[field2].classes_,
                    key=field2
                )
            else:
                input_data[field2] = st.number_input(
                    field2.replace("_", " ").title(),
                    value=50.0,
                    key=field2
                )

# ---------------- PREDICT BUTTON ----------------
if st.button("🚀 Predict Performance"):

    try:

        input_df = pd.DataFrame([input_data])

        # Label Encoding
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])

        # Scale numerical columns if present
        scale_cols = [
            col for col in [
                "last_exam_score",
                "assignment_scores_avg",
                "concept_understanding_score"
            ]
            if col in input_df.columns
        ]

        if scale_cols:
            input_df[scale_cols] = scaler.transform(
                input_df[scale_cols]
            )

        # Ensure correct order
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Prediction
        prediction = model.predict(input_df)

        st.divider()

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(input_df)[0]

            fail_prob = probs[0] * 100
            pass_prob = probs[1] * 100

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Pass Probability",
                    f"{pass_prob:.2f}%"
                )

            with c2:
                st.metric(
                    "Fail Probability",
                    f"{fail_prob:.2f}%"
                )

            st.progress(int(pass_prob))

        # Result
        if prediction[0] == 1:

            st.markdown(f"""
            <div class="result-card">
                <h1>🎉 PASS</h1>
                <h3>Student is likely to pass.</h3>
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

        else:

            st.markdown("""
            <div class="result-card">
                <h1>❌ FAIL</h1>
                <h3>Student may need improvement.</h3>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Student Performance Prediction using Machine Learning")