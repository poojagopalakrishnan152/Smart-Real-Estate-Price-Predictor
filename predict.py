# predict.py (Root Launcher)
import sys
import os
from pathlib import Path

subfolder = Path(__file__).resolve().parent / "Smart_Real_Estate_Price_Predictor"
if subfolder.exists():
    os.chdir(subfolder)
    sys.path.insert(0, str(subfolder))

from predict import predict_property_price

if __name__ == "__main__":
    res = predict_property_price(
        location="Whitefield",
        total_sqft=1200,
        area_type="Super built-up Area",
        availability="Ready To Move",
        society_presence=1
    )
    print("Sample Prediction:")
    for k, v in res.items():
        print(f"  {k}: {v}")
