import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Success AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS - UNIQUE COLOR THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #FFF5EB 0%,
        #F1EDFF 50%,
        #E8FAF2 100%
    );
    color: #51436F;
}

.block-container {
    max-width: 1100px;
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}

h1, h2, h3, h4 {
    color: #66518E !important;
}

p, label {
    color: #62577A !important;
}

.hero {
    background: linear-gradient(
        110deg,
        #FADCCB,
        #E4DBFF,
        #D8F4E6
    );
    padding: 22px;
    border-radius: 22px;
    text-align: center;
    border: 1px solid white;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 36px;
    color: #65518A !important;
    margin-bottom: 5px;
}

.hero p {
    color: #695C80 !important;
}

.card {
    background: rgba(255,255,255,0.90);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #E9DDF5;
    margin-bottom: 12px;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8DDF5;
    padding: 16px;
    border-radius: 16px;
}

div.stButton > button {
    background: linear-gradient(
        90deg,
        #9275C8,
        #70BFA4
    );
    color: white;
    border: none;
    border-radius: 12px;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #795DB0,
        #52A98B
    );
    color: white;
    border: none;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: white;
    border-radius: 10px;
}

.result-box {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #E4D9F2;
    text-align: center;
}

.footer {
    text-align: center;
    color: #817494;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🎓 Student Success AI</h1>
    <p>Smart Academic Performance Prediction System</p>
    <p>✨ Predict • Analyze • Improve • Succeed</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("student_Pass_Fail_classification.pkl")


if not os.path.exists("student_Pass_Fail_classification.pkl"):
    st.error("Model file not found. Please upload the PKL file.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# MODEL INFORMATION
# ==========================================

st.markdown("### 🌷 Student Performance Center")

tab1, tab2, tab3 = st.tabs([
    "🎯 Single Prediction",
    "📂 Batch Prediction",
    "📊 History & Reports"
])


# ==========================================
# TAB 1 - SINGLE PREDICTION
# ==========================================

with tab1:

    left, right = st.columns([1, 1], gap="large")

    with left:

        st.markdown("#### 📝 Student Details")

        study_hours = st.number_input(
            "📚 Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )

        attendance = st.number_input(
            "📅 Attendance Percentage",
            min_value=0,
            max_value=100,
            value=78,
            step=1
        )

        previous_marks = st.number_input(
            "📝 Previous Marks",
            min_value=0,
            max_value=100,
            value=64,
            step=1
        )

        threshold = st.slider(
            "🎚️ Pass Decision Threshold",
            min_value=10,
            max_value=90,
            value=50,
            step=5,
            help="Pass probability must reach this threshold."
        )

        predict = st.button(
            "✨ Analyze Student Performance",
            use_container_width=True
        )

    with right:

        st.markdown("#### 💜 Performance Guide")

        st.markdown("""
        <div class="card">
            <h4>📚 Study Hours</h4>
            <p>Daily study time of the student.</p>
            <hr>
            <h4>📅 Attendance</h4>
            <p>Student's attendance percentage.</p>
            <hr>
            <h4>📝 Previous Marks</h4>
            <p>Marks obtained in previous examinations.</p>
            <hr>
            <h4>🤖 AI Analysis</h4>
            <p>Logistic Regression predicts the academic result.</p>
        </div>
        """, unsafe_allow_html=True)

    # ======================================
    # PREDICTION RESULT
    # ======================================

    if predict:

        student = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Marks": [previous_marks]
        })

        probabilities = model.predict_proba(student)[0]

        classes = list(model.classes_)

        pass_probability = (
            probabilities[classes.index(1)] * 100
            if 1 in classes else 0
        )

        fail_probability = (
            probabilities[classes.index(0)] * 100
            if 0 in classes else 0
        )

        # Custom threshold
        prediction = int(
            pass_probability >= threshold
        )

        result = "Pass" if prediction == 1 else "Fail"

        confidence = max(
            pass_probability,
            fail_probability
        )

        st.session_state.history.append({
            "Study Hours": study_hours,
            "Attendance": attendance,
            "Previous Marks": previous_marks,
            "Prediction": result,
            "Pass Probability": round(pass_probability, 2),
            "Fail Probability": round(fail_probability, 2)
        })

        st.markdown("---")

        st.markdown("### 🎉 Analysis Result")

        if prediction == 1:
            st.success("🌟 Predicted Result: PASS")
        else:
            st.warning("📖 Predicted Result: FAIL")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🟢 Pass Probability",
            f"{pass_probability:.2f}%"
        )

        c2.metric(
            "🔴 Fail Probability",
            f"{fail_probability:.2f}%"
        )

        c3.metric(
            "🎯 Confidence Indicator",
            f"{confidence:.2f}%"
        )

        st.markdown("#### 📊 Probability Analysis")

        st.write("Pass Probability")
        st.progress(int(round(pass_probability)))

        st.write("Fail Probability")
        st.progress(int(round(fail_probability)))

        st.caption(
            f"Decision threshold: {threshold}% | "
            f"Prediction: {result}"
        )

        # SMART SUGGESTIONS

        st.markdown("### 💡 Smart Improvement Suggestions")

        tips = []

        if study_hours < 5:
            tips.append(
                "📚 Increase daily study time gradually."
            )

        if attendance < 75:
            tips.append(
                "📅 Improve attendance and attend classes regularly."
            )

        if previous_marks < 50:
            tips.append(
                "📝 Focus on revision and practice tests."
            )

        if study_hours >= 5 and attendance >= 75 and previous_marks >= 50:
            tips.append(
                "🌟 Maintain your current study routine."
            )

        for tip in tips:
            st.info(tip)

        st.caption(
            "This is an ML-based estimate, not a guaranteed "
            "academic result. Confidence is the model's "
            "highest class probability."
        )


# ==========================================
# TAB 2 - BATCH PREDICTION
# ==========================================

with tab2:

    st.markdown("### 📂 Multiple Student Prediction")

    st.write(
        "Upload a CSV file containing multiple students' details."
    )

    sample = pd.DataFrame({
        "Study_Hours": [6, 3, 8],
        "Attendance": [80, 55, 90],
        "Previous_Marks": [70, 35, 85]
    })

    st.download_button(
        "⬇️ Download Sample CSV",
        sample.to_csv(index=False).encode("utf-8"),
        "student_sample.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload Student CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_data = pd.read_csv(uploaded_file)

            required = [
                "Study_Hours",
                "Attendance",
                "Previous_Marks"
            ]

            if not all(
                col in batch_data.columns
                for col in required
            ):
                st.error(
                    "CSV must contain Study_Hours, "
                    "Attendance and Previous_Marks columns."
                )

            else:

                batch_data = batch_data.copy()

                for col in required:
                    batch_data[col] = pd.to_numeric(
                        batch_data[col],
                        errors="coerce"
                    )

                valid = (
                    batch_data[required].notna().all(axis=1)
                    & batch_data["Study_Hours"].between(0, 24)
                    & batch_data["Attendance"].between(0, 100)
                    & batch_data["Previous_Marks"].between(0, 100)
                )

                valid_data = batch_data.loc[valid].copy()

                if len(valid_data) == 0:

                    st.error(
                        "CSV mein valid student records nahi mile."
                    )

                else:

                    X_batch = valid_data[required]

                    probabilities = model.predict_proba(X_batch)

                    classes = list(model.classes_)

                    pass_prob = (
                        probabilities[:, classes.index(1)] * 100
                        if 1 in classes
                        else np.zeros(len(valid_data))
                    )

                    fail_prob = (
                        probabilities[:, classes.index(0)] * 100
                        if 0 in classes
                        else np.zeros(len(valid_data))
                    )

                    batch_data.loc[valid, "Pass Probability (%)"] = np.round(
                        pass_prob, 2
                    )

                    batch_data.loc[valid, "Fail Probability (%)"] = np.round(
                        fail_prob, 2
                    )

                    batch_data.loc[valid, "Prediction"] = np.where(
                        pass_prob >= 50,
                        "Pass",
                        "Fail"
                    )

                    st.success(
                        f"{len(valid_data)} students analyzed successfully!"
                    )

                    st.dataframe(
                        batch_data,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.metric(
                        "Predicted Pass Students",
                        int((pass_prob >= 50).sum())
                    )

                    st.metric(
                        "Predicted Fail Students",
                        int((pass_prob < 50).sum())
                    )

                    st.download_button(
                        "⬇️ Download Complete Results",
                        batch_data.to_csv(index=False).encode("utf-8"),
                        "student_batch_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

                    if (~valid).any():
                        st.warning(
                            f"{int((~valid).sum())} invalid rows "
                            "were not predicted."
                        )

        except Exception as e:
            st.error(f"Unable to process CSV: {e}")


# ==========================================
# TAB 3 - HISTORY AND REPORTS
# ==========================================

with tab3:

    st.markdown("### 📊 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        total = len(history_df)

        passed = (
            history_df["Prediction"] == "Pass"
        ).sum()

        failed = (
            history_df["Prediction"] == "Fail"
        ).sum()

        a, b, c = st.columns(3)

        a.metric("Total Predictions", total)
        b.metric("Pass Predictions", int(passed))
        c.metric("Fail Predictions", int(failed))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download History Report",
            history_df.to_csv(index=False).encode("utf-8"),
            "student_history.csv",
            "text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.history = []
            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Make your first prediction!"
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <h4>💜 Student Success AI</h4>
    Built with Python | Streamlit | Logistic Regression
    <br>
    Making Education Smarter with Machine Learning ✨
</div>
""", unsafe_allow_html=True)
