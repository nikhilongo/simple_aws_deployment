from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("linear_regression_package.pkl", "rb") as f:
    model = pickle.load(f)

with open("cgpa_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    cgpa = float(request.form["cgpa"])

    # Create dataframe for prediction
    data = pd.DataFrame([[cgpa]], columns=["cgpa"])

    # Scale only CGPA
    data["cgpa"] = scaler.transform(data[["cgpa"]])

    # Predict
    prediction = model.predict(data)[0]
    prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
