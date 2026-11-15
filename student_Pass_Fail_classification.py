import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F3EEFF,
        #E9F6FF,
        #E9FFF6
    );
    color: #344477;
}

.block-container {
    max-width: 850px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

h1, h2, h3 {
    color: #40527D !important;
}

p, label {
    color: #526184 !important;
}

.hero {
    background: linear-gradient(
        110deg,
        #DDD5FF,
        #D9EEFF,
        #D8F8E9
    );
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 18px;
    border: 1px solid white;
}

.hero h1 {
    color: #40527D !important;
    font-size: 34px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 14px;
}

div.stButton > button {
    background: linear-gradient(
        90deg,
        #8878D9,
        #64A9DF
    );
    color: white;
    border: none;
    border-radius: 12px;
    height: 45px;
    font-weight: bold;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCDFF5;
    border-radius: 15px;
    padding: 15px;
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #DCDFF5;
    text-align: center;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #7180A5;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Success Predictor</h1>
    <p>Discover student performance using Machine Learning</p>
    <p>✨ Learn Better • Perform Better • Succeed Better</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("student_Pass_Fail_classification.pkl")


if not os.path.exists("student_Pass_Fail_classification.pkl"):
    st.error("Model file nahi mili. PKL file GitHub par upload karo.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()


# ---------------- INPUT SECTION ----------------
st.subheader("✨ Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "📚 Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.5
    )

    attendance = st.number_input(
        "📅 Attendance (%)",
        min_value=0,
        max_value=100,
        value=78,
        step=1
    )

with col2:
    previous_marks = st.number_input(
        "📝 Previous Marks (%)",
        min_value=0,
        max_value=100,
        value=64,
        step=1
    )

st.caption("Enter student information to predict the academic result.")


# ---------------- PREDICTION ----------------
if st.button("✨ Predict Student Result", use_container_width=True):

    student_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks]
    })

    prediction = model.predict(student_data)[0]

    probabilities = model.predict_proba(student_data)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    pass_probability = class_probabilities.get(1, 0) * 100
    fail_probability = class_probabilities.get(0, 0) * 100

    result = "Pass" if prediction == 1 else "Fail"

    # Save history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Study Hours": study_hours,
        "Attendance": attendance,
        "Previous Marks": previous_marks,
        "Prediction": result,
        "Pass Probability (%)": round(pass_probability, 2),
        "Fail Probability (%)": round(fail_probability, 2)
    })

    # ---------------- RESULT ----------------
    st.markdown("---")
    st.subheader("💜 Prediction Result")

    if prediction == 1:
        st.success("🎉 Congratulations! Predicted Result: PASS")
    else:
        st.warning("📖 Predicted Result: FAIL")

    r1, r2 = st.columns(2)

    with r1:
        st.metric(
            "🟢 Pass Probability",
            f"{pass_probability:.2f}%"
        )

    with r2:
        st.metric(
            "🔴 Fail Probability",
            f"{fail_probability:.2f}%"
        )

    # Probability bars
    st.write("### 📊 Performance Probability")

    st.write("Pass")
    st.progress(int(round(pass_probability)))

    st.write("Fail")
    st.progress(int(round(fail_probability)))

    # ---------------- SUGGESTIONS ----------------
    st.subheader("💡 Student Improvement Tips")

    tips = []

    if study_hours < 5:
        tips.append("📚 Try to increase your daily study hours.")

    if attendance < 75:
        tips.append("📅 Improve your class attendance.")

    if previous_marks < 50:
        tips.append("📝 Focus on improving your previous marks.")

    if not tips:
        tips.append("🌟 Keep maintaining your good study routine!")

    for tip in tips:
        st.info(tip)

    st.caption(
        "Note: This prediction is based on a trained ML model. "
        "It is an estimate, not a guaranteed academic result."
    )


# ---------------- PREDICTION HISTORY ----------------
if "history" in st.session_state and st.session_state.history:

    st.markdown("---")
    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv,
        file_name="student_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<div class="footer">
    <h4>💜 Student Success Predictor</h4>
    Built with Python | Streamlit | Logistic Regression
    <br>
    Making Learning Smarter with Machine Learning ✨
</div>
""", unsafe_allow_html=True)
