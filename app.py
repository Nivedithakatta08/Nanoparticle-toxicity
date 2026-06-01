from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model (Random Forest, etc.)
model = joblib.load("nanoparticle_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect input values from form
    prop1 = float(request.form["prop1"])
    prop2 = float(request.form["prop2"])
    prop3 = float(request.form["prop3"])
    prop4 = float(request.form["prop4"])
    
    # Convert to numpy array for model
    features = np.array([[prop1, prop2, prop3, prop4]])
    prediction = model.predict(features)[0]
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
