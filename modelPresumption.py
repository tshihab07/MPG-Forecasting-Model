import joblib
import pandas as pd

# load trained model
model = joblib.load(r'reports\forecastingModel.pkl')

# Example input data for prediction
vehicle = pd.DataFrame([{
    'cylinders': 4,
    'horsepower': 95,
    'weight': 2400,
    'car_age': 12,
    'origin_japan': 1,
    'origin_usa': 0
}])

# predict MPG
predicted_mpg = model.predict(vehicle)[0]

# output the prediction
print(f"Predicted MPG: {predicted_mpg:.2f}")