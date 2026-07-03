from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "car_price_model.pkl")
ENCODERS_PATH = os.path.join("model", "encoders.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    meta = pickle.load(f)

le_fuel = meta["fuel"]
le_seller = meta["seller"]
le_trans = meta["transmission"]
CURRENT_YEAR = meta["current_year"]
FEATURE_COLS = meta["feature_cols"]


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        fuel_options=list(le_fuel.classes_),
        seller_options=list(le_seller.classes_),
        trans_options=list(le_trans.classes_),
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        fuel_type = request.form["fuel_type"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]
        owner = int(request.form["owner"])

        car_age = CURRENT_YEAR - year

        fuel_enc = le_fuel.transform([fuel_type])[0]
        seller_enc = le_seller.transform([seller_type])[0]
        trans_enc = le_trans.transform([transmission])[0]

        features = np.array([[
            present_price, kms_driven, car_age,
            fuel_enc, seller_enc, trans_enc, owner
        ]])

        pred = model.predict(features)[0]
        pred = round(float(pred), 2)

        return render_template(
            "index.html",
            fuel_options=list(le_fuel.classes_),
            seller_options=list(le_seller.classes_),
            trans_options=list(le_trans.classes_),
            prediction=pred,
            form_data=request.form,
        )
    except Exception as e:
        return render_template(
            "index.html",
            fuel_options=list(le_fuel.classes_),
            seller_options=list(le_seller.classes_),
            trans_options=list(le_trans.classes_),
            prediction=None,
            error=str(e),
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
