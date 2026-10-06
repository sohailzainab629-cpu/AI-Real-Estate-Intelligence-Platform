import joblib
import pandas as pd


MODEL_PATH = "models/price_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_price(data):
    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)

    return float(prediction[0])