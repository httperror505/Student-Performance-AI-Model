from flask import Flask, request, jsonify, render_template
import pickle  # or joblib for loading your trained model
import os

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

# Load your trained model (ensure it's saved and accessible)
model = pickle.load(open("dataset/capstone_ai_1.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Parse incoming JSON data
    # Ensure the keys match your model's expected input
    input_features = [
        data['Age'], data['Gender'],
        data['ParentalEducation'], data['StudyTimeWeeklyCategory'], 
        data['Absences'], data['Tutoring'], 
        data['ParentalSupport'], data['ExtracurricularCategory'], data['Year Level'], data['Subject'], data['Previous Grades']
    ]
    prediction = model.predict([input_features])  # Predict using your model
    grade_classes = {
        0: 'A (GPA >= 3.0)',
        1: 'B (2.2 <= GPA < 3.0)',
        2: 'C (1.4 <= GPA < 2.2)',
        3: 'D (0.8 <= GPA < 1.4)',
        4: 'F (GPA < 0.8)'
    }
    print("Pred:", prediction)
    predicted_grade = grade_classes[prediction[0]]
    return jsonify({"Predicted Grade": predicted_grade})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_plot', methods=['POST'])
def upload_plot():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
