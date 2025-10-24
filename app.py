import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# --- App & Database Configuration ---
app = Flask(__name__)

# *** THIS IS THE KEY CHANGE ***
# Get the database URL from the Render environment variable
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
# (We remove the old 'sqlite:///' line completely)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_fallback_secret_key_if_not_set')

db = SQLAlchemy(app)

# --- Model Loading ---
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'model.sav')
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Database Model ---
# (This model is unchanged)
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

# *** REMOVED ***
# We REMOVE the 'with app.app_context(): db.create_all()' from this file.
# The build.sh script now handles this.

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if not model:
            flash('Model is not loaded. Cannot make predictions.', 'danger')
            return render_template('predict.html', result=None)
        try:
            # (All your prediction logic is unchanged)
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
            
            f_age = age
            f_max_heart_rate = max_heart_rate_achieved
            f_st_depression = st_depression
            f_num_major_vessels = num_major_vessels
            f_cp_atypical = 1 if chest_pain_type == 1 else 0
            f_cp_non_anginal = 1 if chest_pain_type == 2 else 0
            f_cp_typical = 1 if chest_pain_type == 0 else 0
            f_slope_flat = 1 if st_slope == 1 else 0
            f_slope_upsloping = 1 if st_slope == 0 else 0
            f_thal_normal = 1 if thalassemia == 2 else 0
            f_thal_reversible = 1 if thalassemia == 3 else 0
            f_sex_male = 1 if sex == 1 else 0
            f_exang_yes = 1 if exercise_induced_angina == 1 else 0
            features = [
                f_age, f_max_heart_rate, f_st_depression, f_num_major_vessels,
                f_cp_atypical, f_cp_non_anginal, f_cp_typical, f_slope_flat,
                f_slope_upsloping, f_thal_normal, f_thal_reversible,
                f_sex_male, f_exang_yes
            ]
            final_features = np.array(features).reshape(1, -1)
            prediction_proba = model.predict_proba(final_features)[0][1]
            risk_score = int(round(prediction_proba * 100))

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
                
            new_prediction = Prediction(
                patient_name=patient_name, age=age, sex=sex, chest_pain_type=chest_pain_type,
                max_heart_rate_achieved=max_heart_rate_achieved, exercise_induced_angina=exercise_induced_angina,
                st_depression=st_depression, st_slope=st_slope, num_major_vessels=num_major_vessels,
                thalassemia=thalassemia, risk_score=risk_score, risk_level=risk_level
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            result = {
                'name': patient_name, 'score': risk_score, 'level': risk_level, 'class': risk_class,
                'age': age, 'sex_str': 'Male' if sex == 1 else 'Female',
                'max_heart_rate_achieved': max_heart_rate_achieved, 'st_depression': st_depression,
                'num_major_vessels': num_major_vessels,
                'exercise_induced_angina_str': 'Yes' if exercise_induced_angina == 1 else 'No',
                'chest_pain_type_str': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}.get(chest_pain_type, 'N/A'),
                'st_slope_str': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}.get(st_slope, 'N/A'),
                'thalassemia_str': {1: 'Fixed Defect', 2: 'Normal', 3: 'Reversible Defect'}.get(thalassemia, 'N/A')
            }
            return render_template('predict.html', result=result)
        except Exception as e:
            print(f"Error during prediction: {e}")
            flash(f'An error occurred: {e}', 'danger')
            return render_template('predict.html', result=None)
    return render_template('predict.html', result=None)

@app.route('/history')
def history():
    try:
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        flash('Could not retrieve prediction history.', 'danger')
        return render_template('history.html', predictions=[])

@app.route('/about')
def about():
    return render_template('about.html')

# --- Run the App ---
if __name__ == '__main__':
    # We turn debug OFF for production
    app.run(debug=False)