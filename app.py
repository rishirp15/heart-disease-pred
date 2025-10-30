import os
# import joblib  ### <-- CHANGED (Removed joblib)
import pickle  ### <-- CHANGED (Added pickle)
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# --- App & Database Configuration ---

# Get the absolute path of the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
# Configure the SQLite database, path will be in the project directory
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'predictions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '1234567890qwertyuiop'  # Change this to a random string

db = SQLAlchemy(app)

# --- Model & Scaler Loading --- ### <-- CHANGED SECTION
model = None
scaler = None ### <-- CHANGED

model_path = os.path.join(basedir, 'model.sav')
scaler_path = os.path.join(basedir, 'scaler.sav') ### <-- CHANGED

try:
    # Use pickle to load the model you saved with pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f) ### <-- CHANGED
    print("Model loaded successfully.")
    
    # Use pickle to load the scaler
    with open(scaler_path, 'rb') as f: ### <-- CHANGED
        scaler = pickle.load(f) ### <-- CHANGED
    print("Scaler loaded successfully.") ### <-- CHANGED
    
except FileNotFoundError as e: ### <-- CHANGED
    print(f"Error: Model or scaler file not found. {e}")
    model = None
    scaler = None
except Exception as e:
    print(f"Error loading model or scaler: {e}") ### <-- CHANGED
    model = None
    scaler = None


# --- Database Model ---
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False, default='Anonymous')
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Store the 9 raw input features
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False) # 1=male, 0=female
    chest_pain_type = db.Column(db.Integer, nullable=False)
    max_heart_rate_achieved = db.Column(db.Integer, nullable=False)
    exercise_induced_angina = db.Column(db.Integer, nullable=False)
    st_depression = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.Integer, nullable=False)
    num_major_vessels = db.Column(db.Integer, nullable=False)
    thalassemia = db.Column(db.Integer, nullable=False)
    
    # Prediction result
    risk_score = db.Column(db.Integer, nullable=False) # Probability (0-100)
    risk_level = db.Column(db.String(50), nullable=False) # e.g., "High Risk"

    def __repr__(self):
        return f'<Prediction {self.id} - {self.patient_name}>'

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handles the prediction form submission."""
    
    if request.method == 'POST':
        # Check if model AND scaler are loaded
        if not model or not scaler: ### <-- CHANGED
            flash('Model or Scaler is not loaded. Cannot make predictions.', 'danger') ### <-- CHANGED
            return render_template('predict.html', result=None)
            
        try:
            # 1. Get 9 RAW inputs from form
            patient_name = request.form.get('patient_name') or 'Anonymous'
            age = int(request.form['age'])
            sex = int(request.form['sex']) # 0 or 1
            chest_pain_type = int(request.form['chest_pain_type']) # 0, 1, 2, or 3
            max_heart_rate_achieved = int(request.form['max_heart_rate_achieved'])
            exercise_induced_angina = int(request.form['exercise_induced_angina']) # 0 or 1
            st_depression = float(request.form['st_depression']) # Oldpeak
            st_slope = int(request.form['st_slope']) # 0, 1, or 2
            num_major_vessels = int(request.form['num_major_vessels']) # 0, 1, 2, or 3
            thalassemia = int(request.form['thalassemia']) # 1, 2, or 3

            # 2. Manually One-Hot Encode into 13 features
            # ... (Encoding logic remains the same) ...
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

            # 3. Create feature array
            features = [
                f_age, f_max_heart_rate, f_st_depression, f_num_major_vessels,
                f_cp_atypical, f_cp_non_anginal, f_cp_typical, f_slope_flat,
                f_slope_upsloping, f_thal_normal, f_thal_reversible,
                f_sex_male, f_exang_yes
            ]
            
            final_features_unscaled = np.array(features).reshape(1, -1) ### <-- CHANGED
            
            if final_features_unscaled.shape[1] != 13: ### <-- CHANGED
                flash(f'Feature engineering error. Expected 13 features, got {final_features_unscaled.shape[1]}', 'danger')
                return render_template('predict.html', result=None)

            # 4. *** APPLY THE SCALER *** ### <-- NEW STEP
            final_features_scaled = scaler.transform(final_features_unscaled)

            # 5. Make prediction (using the SCALED features)
            prediction_proba = model.predict_proba(final_features_scaled)[0][1] ### <-- CHANGED
            risk_score = int(round(prediction_proba * 100))

            # 6. Determine risk level
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
            
            # 7. Save prediction
            new_prediction = Prediction(
                patient_name=patient_name,
                age=age,
                sex=sex,
                chest_pain_type=chest_pain_type,
                max_heart_rate_achieved=max_heart_rate_achieved,
                exercise_induced_angina=exercise_induced_angina,
                st_depression=st_depression,
                st_slope=st_slope,
                num_major_vessels=num_major_vessels,
                thalassemia=thalassemia,
                risk_score=risk_score,
                risk_level=risk_level
            )
            
            db.session.add(new_prediction)
            db.session.commit()
            
            # 8. Prepare result dictionary for display
            result = {
                'name': patient_name,
                'score': risk_score,
                'level': risk_level,
                'class': risk_class, 
                'age': age,
                'sex_str': 'Male' if sex == 1 else 'Female',
                'max_heart_rate_achieved': max_heart_rate_achieved,
                'st_depression': st_depression,
                'num_major_vessels': num_major_vessels,
                'exercise_induced_angina_str': 'Yes' if exercise_induced_angina == 1 else 'No',
                'chest_pain_type_str': {
                    0: 'Typical Angina',
                    1: 'Atypical Angina',
                    2: 'Non-anginal Pain',
                    3: 'Asymptomatic'
                }.get(chest_pain_type, 'N/A'),
                
                'st_slope_str': {
                    0: 'Upsloping',
                    1: 'Flat',
                    2: 'Downsloping'
                }.get(st_slope, 'N/A'),
                
                'thalassemia_str': {
                    1: 'Fixed Defect',
                    2: 'Normal',
                    3: 'Reversible Defect'
                }.get(thalassemia, 'N/A')
            }
            
            return render_template('predict.html', result=result)

        except Exception as e:
            print(f"Error during prediction: {e}")
            flash(f'An error occurred: {e}', 'danger')
            return render_template('predict.html', result=None)

    # For a GET request, just show the form
    return render_template('predict.html', result=None)


@app.route('/history')
def history():
    """Displays all past predictions."""
    try:
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        print(f"Error fetching history: {e}")
        flash('Could not retrieve prediction history.', 'danger')
        return render_template('history.html', predictions=[])

@app.route('/about')
def about():
    """Renders the about page with term definitions."""
    return render_template('about.html')


# --- Run the App ---

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)
