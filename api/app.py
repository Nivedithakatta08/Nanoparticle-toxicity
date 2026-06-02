from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/nanoparticle_model.pkl")  # adjust path if needed

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        prop1 = float(request.form["prop1"])
        prop2 = float(request.form["prop2"])
        prop3 = float(request.form["prop3"])
        prop4 = float(request.form["prop4"])

        features = np.array([[prop1, prop2, prop3, prop4]])
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# Do not call app.run(); Vercel handles the server lifecycle
