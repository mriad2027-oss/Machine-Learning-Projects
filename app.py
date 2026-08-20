import streamlit as st
import numpy as np
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CardioAI | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# LOAD MODEL + SCALER  (fixed: no error handling before —
# a missing .pkl file used to crash with a raw traceback)
# =========================================================

try:
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(
        "⚠ Could not load the model files (`logistic_model.pkl` / `scaler.pkl`). "
        "Make sure both files are in the same folder as this script.\n\n"
        f"Details: {e}"
    )
    st.stop()


# =========================================================
# HTML RENDER HELPER  (THE MAIN FIX)
#
# The bug in the screenshots: raw <div> tags were showing up as
# visible text instead of being rendered. Cause: every HTML string
# was written indented (matching the Python code's own indentation,
# e.g. inside `with col1:` blocks) and had blank lines between inner
# tags. Standard Markdown treats any line indented 4+ spaces as a
# literal code block, AND a blank line inside a raw-HTML block ends
# that block early -- so Markdown was "escaping" out of HTML mode
# partway through each card and printing the rest as plain text.
#
# Fix: strip leading whitespace and remove blank lines from every
# HTML string right before handing it to st.markdown, so it always
# renders as one continuous HTML block regardless of how it's
# indented in the Python source (which stays nicely indented/
# readable, since the helper cleans it up automatically).
# =========================================================

def html_block(content: str) -> None:
    lines = [line.strip() for line in content.strip("\n").split("\n")]
    lines = [line for line in lines if line != ""]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# =========================================================
# HELPER: build a repeating ECG waveform path (server-side,
# so the SVG markup doesn't need to be hand-duplicated)
# =========================================================

def build_ecg_path(units: int = 8, unit_width: int = 100) -> str:
    """One 'unit' is a small P-wave bump, a sharp QRS spike, and a
    T-wave bump, all sitting on a flat baseline at y=50. Repeating it
    `units` times gives a continuous heartbeat trace."""
    d = "M0,50"
    for i in range(units):
        x = i * unit_width
        d += (
            f" L{x+10},50"
            f" L{x+20},50 L{x+23},35 L{x+26},50"
            f" L{x+33},50"
            f" L{x+36},6 L{x+40},94 L{x+44},50"
            f" L{x+50},50"
            f" L{x+55},40 L{x+60},58 L{x+65},50"
            f" L{x+unit_width},50"
        )
    return d


ECG_UNITS = 8
ECG_UNIT_WIDTH = 100
ECG_TOTAL_WIDTH = ECG_UNITS * ECG_UNIT_WIDTH
ECG_PATH_D = build_ecg_path(ECG_UNITS, ECG_UNIT_WIDTH)
FLATLINE_PATH_D = f"M0,50 L{ECG_TOTAL_WIDTH},50"


# =========================================================
# CSS
# (left as a plain st.markdown call: a <style> block is an HTML
# "type 1" block per the Markdown spec, which is NOT terminated by
# blank lines, so it was never affected by the bug above)
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37, 99, 235, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14, 165, 233, 0.08),
            transparent 30%
        ),
        #07111f;
}


/* Main width */

.block-container {
    max-width: 1200px;
    padding-top: 45px;
    padding-bottom: 50px;
}


/* =========================================================
   HERO
   ========================================================= */

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 50px;

    background: rgba(59, 130, 246, 0.10);
    border: 1px solid rgba(59, 130, 246, 0.25);

    color: #60a5fa;

    font-size: 11px;
    font-weight: 700;

    letter-spacing: 1.2px;

    margin-bottom: 18px;
}

.hero-title {
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -3px;
    line-height: 1;
    color: #f8fafc;
}

.hero-title span {
    color: #3b82f6;
}

.hero-subtitle {
    margin-top: 18px;

    max-width: 680px;

    color: #94a3b8;

    font-size: 17px;
    line-height: 1.6;
}


/* =========================================================
   CARDS
   ========================================================= */

.card {
    background: rgba(15, 23, 42, 0.72);

    border: 1px solid rgba(148, 163, 184, 0.12);

    border-radius: 18px;

    padding: 22px;

    min-height: 150px;

    box-shadow:
        0 15px 40px rgba(0, 0, 0, 0.18);
}

.card-icon {
    font-size: 27px;
    margin-bottom: 12px;
}

.card-title {
    color: #f8fafc;

    font-size: 16px;
    font-weight: 700;

    margin-bottom: 8px;
}

.card-text {
    color: #94a3b8;

    font-size: 13px;

    line-height: 1.6;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-title {
    margin-top: 42px;
    margin-bottom: 7px;

    color: #f8fafc;

    font-size: 28px;
    font-weight: 800;
}

.section-description {
    color: #94a3b8;

    font-size: 14px;

    margin-bottom: 24px;
}


/* =========================================================
   FORM CONTAINER
   ========================================================= */

.form-container {
    background: rgba(15, 23, 42, 0.72);

    border: 1px solid rgba(148, 163, 184, 0.12);

    border-radius: 22px;

    padding: 25px;

    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.20);
}


/* =========================================================
   STREAMLIT INPUTS
   ========================================================= */

.stNumberInput label,
.stSelectbox label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

.stNumberInput input {
    background-color: #0b1729 !important;

    color: #f8fafc !important;

    border-radius: 10px !important;
}

div[data-baseweb="select"] {
    background-color: #0b1729 !important;

    border-radius: 10px !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {

    width: 100%;

    min-height: 50px;

    border: none;

    border-radius: 12px;

    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );

    color: white;

    font-size: 15px;

    font-weight: 700;

    transition: 0.2s;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 12px 30px rgba(37, 99, 235, 0.35);
}


/* =========================================================
   RESULT
   ========================================================= */

.result-card {

    margin-top: 30px;

    padding: 30px;

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.96),
            rgba(12, 24, 43, 0.90)
        );

    border: 1px solid rgba(59, 130, 246, 0.18);

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.25);
}

.result-header {

    color: #64748b;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.5px;

    text-transform: uppercase;
}

.result-value {

    font-size: 30px;

    font-weight: 800;

    margin-top: 10px;

    margin-bottom: 22px;
}

.high-risk {
    color: #fb7185;
}

.low-risk {
    color: #34d399;
}

.probability-label {

    color: #cbd5e1;

    font-size: 14px;

    font-weight: 600;
}

.probability {

    color: #60a5fa;

    font-size: 48px;

    font-weight: 800;

    margin-top: 3px;
}


/* =========================================================
   HEART MONITOR (new)
   ========================================================= */

.monitor-card {
    margin-top: 22px;
    padding: 22px 24px;
    border-radius: 20px;
    background: #060d18;
    border: 1px solid rgba(148, 163, 184, 0.12);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
}

.monitor-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}

.monitor-title {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.monitor-status {
    font-size: 13px;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 50px;
}

.monitor-status.ok {
    color: #34d399;
    background: rgba(52, 211, 153, 0.10);
    border: 1px solid rgba(52, 211, 153, 0.25);
}

.monitor-status.alert {
    color: #fb7185;
    background: rgba(251, 113, 133, 0.10);
    border: 1px solid rgba(251, 113, 133, 0.25);
}

/* Heart icon */

.heart-icon {
    font-size: 26px;
    display: inline-block;
}

.heart-icon.beating {
    animation: heartBeat 1.05s ease-in-out infinite;
}

.heart-icon.stopped {
    opacity: 0.35;
    filter: grayscale(1);
}

@keyframes heartBeat {
    0%   { transform: scale(1); }
    20%  { transform: scale(1.22); }
    35%  { transform: scale(1); }
    50%  { transform: scale(1.15); }
    65%  { transform: scale(1); }
    100% { transform: scale(1); }
}

/* ECG trace */

.ecg-wrapper {
    width: 100%;
    height: 90px;
    overflow: hidden;
    border-radius: 14px;
    background: #0b1729;
    border: 1px solid rgba(148, 163, 184, 0.15);
    position: relative;
}

.ecg-track {
    display: flex;
    width: 200%;
    height: 100%;
}

.ecg-track.scrolling {
    animation: ecgScroll 2.4s linear infinite;
}

.ecg-svg {
    width: 50%;
    height: 100%;
    flex-shrink: 0;
}

@keyframes ecgScroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}

.flatline-flicker {
    animation: flicker 1.6s ease-in-out infinite;
}

@keyframes flicker {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.75; }
}

.monitor-caption {
    margin-top: 10px;
    font-size: 12px;
    color: #64748b;
}


/* =========================================================
   DISCLAIMER
   ========================================================= */

.disclaimer {

    margin-top: 25px;

    padding: 18px 20px;

    border-radius: 14px;

    background: rgba(245, 158, 11, 0.06);

    border: 1px solid rgba(245, 158, 11, 0.16);
}

.disclaimer-title {

    color: #fbbf24;

    font-weight: 700;

    margin-bottom: 6px;
}

.disclaimer-text {

    color: #94a3b8;

    font-size: 12px;

    line-height: 1.6;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    margin-top: 60px;

    padding-top: 25px;

    border-top:
        1px solid rgba(148, 163, 184, 0.08);

    text-align: center;

    color: #64748b;

    font-size: 12px;
}

.footer strong {
    color: #94a3b8;
}


/* Hide Streamlit menu/footer */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

html_block("""
<div class="badge">
    ● AI-POWERED HEALTH ANALYTICS
</div>

<div class="hero-title">
    Cardio<span>AI</span>
</div>

<div class="hero-subtitle">
    Heart Disease Risk Prediction powered by Machine Learning
</div>
""")


st.write("")


# =========================================================
# TOP CARDS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    html_block("""
    <div class="card">
        <div class="card-icon">🧠</div>
        <div class="card-title">
            Machine Learning
        </div>
        <div class="card-text">
            Logistic Regression analyzes clinical features
            to classify heart disease risk.
        </div>
    </div>
    """)


with col2:

    html_block("""
    <div class="card">
        <div class="card-icon">📊</div>
        <div class="card-title">
            Clinical Features
        </div>
        <div class="card-text">
            The model uses 13 patient health indicators
            to generate its prediction.
        </div>
    </div>
    """)


with col3:

    html_block("""
    <div class="card">
        <div class="card-icon">⚡</div>
        <div class="card-title">
            Instant Analysis
        </div>
        <div class="card-text">
            Enter patient information and receive
            the prediction and probability instantly.
        </div>
    </div>
    """)


# =========================================================
# PATIENT SECTION
# =========================================================

html_block("""
<div class="section-title">
    Patient Information
</div>

<div class="section-description">
    Enter the patient's clinical information below.
</div>
""")


# =========================================================
# INPUT FORM
# =========================================================

with st.form("prediction_form"):

    html_block('<div class="form-container">')

    col1, col2, col3 = st.columns(3)


    # =====================================================
    # COLUMN 1
    # =====================================================

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=55
        )


        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
                "Female" if x == 0 else "Male"
        )


        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[x]
        )


        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=130
        )


        chol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=240
        )


    # =====================================================
    # COLUMN 2
    # =====================================================

    with col2:

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )


        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )


        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )


        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )


        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )


    # =====================================================
    # COLUMN 3
    # =====================================================

    with col3:

        slope = st.selectbox(
            "ST Segment Slope",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )


        ca = st.selectbox(
            "Major Vessels (CA)",
            [0, 1, 2, 3, 4]
        )


        thal = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Unknown",
                1: "Normal",
                2: "Fixed Defect",
                3: "Reversible Defect"
            }[x]
        )


        st.write("")


        predict_button = st.form_submit_button(
            "🔍  Analyze Heart Disease Risk"
        )


    html_block('</div>')


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # EXACT SAME FEATURE ORDER AS TRAINING
    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])


    # =====================================================
    # SCALE
    # =====================================================

    input_scaled = scaler.transform(input_data)


    # =====================================================
    # PREDICT
    #
    # IMPORTANT FIX: this model was trained on data where the target
    # label is FLIPPED relative to the usual convention. Confirmed
    # empirically -- the classic high-risk markers for this dataset
    # (blocked vessels/ca, exercise-induced angina/exang, ST
    # depression/oldpeak, thalassemia defect/thal) all push toward
    # class 0, not class 1. So here: class 0 = disease present,
    # class 1 = no disease. Everything below reads raw_proba[0] as
    # the "risk" probability instead of raw_proba[1].
    # =====================================================

    raw_prediction = model.predict(input_scaled)[0]
    raw_proba = model.predict_proba(input_scaled)[0]  # [P(class 0), P(class 1)]

    is_high_risk = (raw_prediction == 0)
    probability = raw_proba[0]        # probability of elevated risk
    probability_percent = probability * 100


    # =====================================================
    # RESULT
    # =====================================================

    if is_high_risk:

        result_title = "⚠ Elevated Heart Disease Risk"

        result_class = "high-risk"

    else:

        result_title = "✓ Lower Heart Disease Risk"

        result_class = "low-risk"


    html_block(f"""
    <div class="result-card">
        <div class="result-header">
            CARDIOAI ANALYSIS
        </div>
        <div class="result-value {result_class}">
            {result_title}
        </div>
        <div class="probability-label">
            Estimated Probability of Class 1
        </div>
        <div class="probability">
            {probability_percent:.1f}%
        </div>
    </div>
    """)


    # =====================================================
    # HEART MONITOR ANIMATION
    # Healthy prediction  -> scrolling ECG heartbeat, pulsing heart
    # Elevated-risk pred. -> flatline, heart stops
    # =====================================================

    if is_high_risk:
        heart_span = '<span class="heart-icon stopped">💔</span>'
        status_badge = '<span class="monitor-status alert">NO RHYTHM DETECTED</span>'
        track_class = "ecg-track flatline-flicker"
        path_d = FLATLINE_PATH_D
        stroke_color = "#fb7185"
        caption = "Flatline — the model flagged this profile as elevated risk."
    else:
        heart_span = '<span class="heart-icon beating">❤️</span>'
        status_badge = '<span class="monitor-status ok">STEADY RHYTHM</span>'
        track_class = "ecg-track scrolling"
        path_d = ECG_PATH_D
        stroke_color = "#34d399"
        caption = "Steady rhythm — the model flagged this profile as lower risk."

    html_block(f"""
    <div class="monitor-card">
        <div class="monitor-header">
            <div class="monitor-title">
                {heart_span} Heart Monitor
            </div>
            {status_badge}
        </div>
        <div class="ecg-wrapper">
            <div class="{track_class}">
                <svg class="ecg-svg" viewBox="0 0 {ECG_TOTAL_WIDTH} 100" preserveAspectRatio="none">
                    <path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
                <svg class="ecg-svg" viewBox="0 0 {ECG_TOTAL_WIDTH} 100" preserveAspectRatio="none">
                    <path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
            </div>
        </div>
        <div class="monitor-caption">{caption}</div>
    </div>
    """)


    # =====================================================
    # PROBABILITY BREAKDOWN
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Class 0 Probability (elevated risk)",
            f"{probability * 100:.1f}%"
        )


    with col2:

        st.metric(
            "Class 1 Probability (lower risk)",
            f"{(1 - probability) * 100:.1f}%"
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    html_block("""
    <div class="disclaimer">
        <div class="disclaimer-title">
            ⚠ Important Notice
        </div>
        <div class="disclaimer-text">
            CardioAI is an educational machine-learning project.
            This prediction is not a medical diagnosis and should
            not replace professional medical advice or evaluation
            by a qualified healthcare professional.
        </div>
    </div>
    """)


# =========================================================
# ABOUT
# =========================================================

html_block("""
<div class="section-title">
    About CardioAI
</div>

<div class="section-description">
    A machine-learning project demonstrating binary classification
    using clinical health features.
</div>
""")


col1, col2, col3 = st.columns(3)


with col1:

    html_block("""
    <div class="card">
        <div class="card-icon">🤖</div>
        <div class="card-title">
            Logistic Regression
        </div>
        <div class="card-text">
            The prediction engine uses Logistic Regression
            for binary classification.
        </div>
    </div>
    """)


with col2:

    html_block("""
    <div class="card">
        <div class="card-icon">⚙️</div>
        <div class="card-title">
            StandardScaler
        </div>
        <div class="card-text">
            Patient features are transformed using the same
            scaler used during model training.
        </div>
    </div>
    """)


with col3:

    html_block("""
    <div class="card">
        <div class="card-icon">📈</div>
        <div class="card-title">
            Probability
        </div>
        <div class="card-text">
            The application displays both the predicted class
            and the probability of class 1.
        </div>
    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

html_block("""
<div class="footer">
    <strong>CardioAI</strong>
    <br><br>
    Heart Disease Prediction
    • Logistic Regression
    • Machine Learning
    • Streamlit
</div>
""")