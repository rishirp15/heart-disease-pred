import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import scipy.stats as stats  # <-- ONLY FOR CI

# --- App & Database ---
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'predictions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '1234567890qwertyuiop'

db = SQLAlchemy(app)

# --- Load Model & Scaler ---
model = None
scaler = None
model_path = os.path.join(basedir, 'model.sav')
scaler_path = os.path.join(basedir, 'scaler.sav')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Model & Scaler loaded.")
except Exception as e:
    print(f"Error: {e}")
    model = scaler = None

# --- DB Model ---
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False, default='Anonymous')
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    chest_pain_type = db.Column(db.Integer, nullable=False)
    max_heart_rate_achieved = db.Column(db.Integer, nullable=False)
    exercise_induced_angina = db.Column(db.Integer, nullable=False)
    st_depression = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.Integer, nullable=False)
    num_major_vessels = db.Column(db.Integer, nullable=False)
    thalassemia = db.Column(db.Integer, nullable=False)
    risk_score = db.Column(db.Integer, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if not model or not scaler:
            flash('Model not loaded.', 'danger')
            return render_template('predict.html', result=None)

        try:
            # === 1. Input ===
            patient_name = request.form.get('patient_name') or 'Anonymous'
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            chest_pain_type = int(request.form['chest_pain_type'])
            max_heart_rate_achieved = int(request.form['max_heart_rate_achieved'])
            exercise_induced_angina = int(request.form['exercise_induced_angina'])
            st_depression = float(request.form['st_depression'])
            st_slope = int(request.form['st_slope'])
            num_major_vessels = int(request.form['num_major_vessels'])
            thalassemia = int(request.form['thalassemia'])

            # === 2. One-Hot Features ===
            features = [
                age,
                max_heart_rate_achieved,
                st_depression,
                num_major_vessels,
                1 if chest_pain_type == 1 else 0,  # CP Atypical
                1 if chest_pain_type == 2 else 0,  # CP Non-anginal
                1 if chest_pain_type == 0 else 0,  # CP Typical
                1 if st_slope == 1 else 0,         # Slope Flat
                1 if st_slope == 0 else 0,         # Slope Upslope
                1 if thalassemia == 2 else 0,      # Thal Normal
                1 if thalassemia == 3 else 0,      # Thal Rev
                1 if sex == 1 else 0,              # Male
                1 if exercise_induced_angina == 1 else 0  # Exang Yes
            ]

            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)[0][1]
            risk_score = int(round(proba * 100))

            # === 3. Risk Level ===
            if risk_score > 75:
                risk_level = 'Very High Risk'
                risk_class = 'level-very-high'
            elif risk_score > 50:
                risk_level = 'High Risk'
                risk_class = 'level-high'
            elif risk_score > 25:
                risk_level = 'Moderate Risk'
                risk_class = 'level-moderate'
            else:
                risk_level = 'Low Risk'
                risk_class = 'level-low'

            # === 4. CONFIDENCE INTERVAL (95%) ===
            se = (proba * (1 - proba) / 100) ** 0.5
            ci_low = max(0, int((proba - 1.96 * se) * 100))
            ci_high = min(100, int((proba + 1.96 * se) * 100))
            ci_str = f"{ci_low}% – {ci_high}%" if ci_low < ci_high else "N/A"

            # === 5. TOP 3 RISK DRIVERS (from coef_) ===
            coef = model.coef_[0]
            feat_names = [
                'Age', 'Max HR', 'ST Depression', 'Vessels',
                'CP Atypical', 'CP Non-anginal', 'CP Typical',
                'Slope Flat', 'Slope Upslope', 'Thal Normal', 'Thal Rev',
                'Male', 'Exang Yes'
            ]
            top_idx = np.argsort(np.abs(coef))[-3:][::-1]
            drivers = [(feat_names[i], round(coef[i], 2)) for i in top_idx]

            # === 6. Save to DB ===
            pred = Prediction(
                patient_name=patient_name, age=age, sex=sex, chest_pain_type=chest_pain_type,
                max_heart_rate_achieved=max_heart_rate_achieved, exercise_induced_angina=exercise_induced_angina,
                st_depression=st_depression, st_slope=st_slope, num_major_vessels=num_major_vessels,
                thalassemia=thalassemia, risk_score=risk_score, risk_level=risk_level
            )
            db.session.add(pred)
            db.session.commit()

            # === 7. Result ===
            result = {
                'name': patient_name, 'score': risk_score, 'level': risk_level, 'class': risk_class,
                'age': age, 'sex_str': 'Male' if sex == 1 else 'Female',
                'max_heart_rate_achieved': max_heart_rate_achieved, 'st_depression': st_depression,
                'num_major_vessels': num_major_vessels,
                'exercise_induced_angina_str': 'Yes' if exercise_induced_angina == 1 else 'No',
                'chest_pain_type_str': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}.get(chest_pain_type, 'N/A'),
                'st_slope_str': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}.get(st_slope, 'N/A'),
                'thalassemia_str': {1: 'Fixed Defect', 2: 'Normal', 3: 'Reversible Defect'}.get(thalassemia, 'N/A'),
                'ci': ci_str,           # ← ADDED
                'drivers': drivers      # ← ADDED
            }
            return render_template('predict.html', result=result)

        except Exception as e:
            flash(f'Error: {e}', 'danger')
            return render_template('predict.html', result=None)

    return render_template('predict.html', result=None)

@app.route('/history')
def history():
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/about')
def about():
    return render_template('about.html')

# --- Run ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)