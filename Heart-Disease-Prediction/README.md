# CardioAI — Heart Disease Risk Prediction

CardioAI is a Streamlit web app that estimates a patient's risk of heart
disease from 13 clinical indicators, using a pre-trained Logistic
Regression model. It includes an animated heart monitor (a scrolling
ECG for lower-risk results, a flatline for elevated-risk results) that
visualizes the prediction as it comes in.

⚠️ **This is an educational project, not a medical tool.** Predictions
are not a diagnosis and should never replace professional medical advice.

---

## Features

- Clean, dark-themed patient intake form (13 clinical fields)
- Instant risk prediction with probability breakdown
- Animated heart-rate monitor that reflects the result (steady rhythm vs. flatline)
- Clear in-app disclaimer

## Tech Stack

- **Streamlit** — web app framework / UI
- **scikit-learn** — Logistic Regression model + StandardScaler
- **joblib** — loading the pre-trained model and scaler
- **NumPy** — input array handling

## Project Structure

```
.
├── app.py       # Main Streamlit application
├── logistic_model.pkl    # Pre-trained Logistic Regression model
├── scaler.pkl            # StandardScaler fitted to match the model
└── README.md
```

The model and scaler are required, pre-trained artifacts — the app
loads them directly with `joblib.load()` at startup and does not train
or fit anything itself.

## Setup

1. **Clone / download the project** and make sure `cardioai_app.py`,
   `logistic_model.pkl`, and `scaler.pkl` are all in the same folder.

2. **Install dependencies:**

   ```bash
   pip install streamlit scikit-learn numpy joblib
   ```

3. **Run the app:**

   ```bash
   streamlit run cardioai_app.py
   ```

4. Open the URL Streamlit prints in your terminal (usually
   `http://localhost:8501`).

## How to Use

1. Fill in the patient's clinical information (age, sex, chest pain
   type, resting blood pressure, cholesterol, and so on — 13 fields
   total).
2. Click **Analyze Heart Disease Risk**.
3. The app shows:
   - A risk verdict (Lower Risk / Elevated Risk) with a probability
   - An animated heart monitor reflecting the result
   - A probability breakdown for both classes

## Model Details

- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Input features (13, in this exact order):**
  `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
- **Preprocessing:** inputs are transformed with the bundled
  `StandardScaler` (`scaler.pkl`) before being passed to the model —
  the same scaler used when the model was trained.

### ⚠️ Important: label convention

This model's training data uses a **flipped target label** relative to
the usual convention:

- **Class `0`** → disease **present** (elevated risk)
- **Class `1`** → **no** disease (lower risk)

This was confirmed empirically by checking the model's coefficients:
established high-risk markers for this kind of data (`ca` — major
vessels blocked, `exang` — exercise-induced angina, `oldpeak` — ST
depression, `thal` — thalassemia defect) all push the model **toward**
class `0`, not class `1`. The app's prediction logic accounts for this
directly — `is_high_risk = (raw_prediction == 0)` — so the UI always
shows the correct verdict. **If this model is ever retrained or
swapped out, re-verify which class means what before trusting the
app's output.**

## Disclaimer

CardioAI is an educational machine-learning project. Its predictions
are not a medical diagnosis and should not replace evaluation by a
qualified healthcare professional.
