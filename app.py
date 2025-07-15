from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Load CSV
        df = pd.read_csv(filepath)

        if 'popularity' not in df.columns:
            return "The CSV must contain a 'popularity' column for target labels."

        # Split features and target
        X = df.drop("popularity", axis=1)
        y = df["popularity"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))

        # Train ANN model
        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                              max_iter=300, class_weight=class_weight_dict, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return render_template('result.html', accuracy=accuracy, report=report)
    except Exception as e:
        return f"Error processing file: {e}"

if __name__ == '__main__':
    app.run(debug=True)
