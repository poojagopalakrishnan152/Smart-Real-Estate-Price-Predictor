# predict.py
# Prediction Engine for Smart Real Estate Price Predictor – Bengaluru
# Validates user inputs, transforms features, invokes the trained ML model,
# and formats Indian currency prices safely.
# Strictly NO bedroom, bathroom, BHK, or balcony inputs.

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "real_estate_model.pkl"
LOCATION_FILE = MODEL_DIR / "location_encoder.pkl"
FEATURES_FILE = MODEL_DIR / "model_features.pkl"

# Global cached artifacts
_MODEL = None
_LOCATION_DATA = None
_FEATURE_METADATA = None


def load_artifacts():
    """
    Safely loads serialized model and metadata with friendly error handling.
    Caches loaded objects in memory for fast prediction responses.
    """
    global _MODEL, _LOCATION_DATA, _FEATURE_METADATA

    if _MODEL is not None and _LOCATION_DATA is not None and _FEATURE_METADATA is not None:
        return _MODEL, _LOCATION_DATA, _FEATURE_METADATA

    if not MODEL_FILE.exists() or not LOCATION_FILE.exists() or not FEATURES_FILE.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Please run 'python train_model.py' first to train and generate the model."
        )

    _MODEL = joblib.load(MODEL_FILE)
    _LOCATION_DATA = joblib.load(LOCATION_FILE)
    _FEATURE_METADATA = joblib.load(FEATURES_FILE)

    return _MODEL, _LOCATION_DATA, _FEATURE_METADATA


def format_indian_currency(price_lakhs):
    """
    Formats raw price in Lakhs into standard Indian currency representation.
    Examples:
      - 25.50 -> ₹ 25.50 Lakhs
      - 85.00 -> ₹ 85.00 Lakhs
      - 125.40 -> ₹ 1.25 Crore
      - 240.00 -> ₹ 2.40 Crore
    """
    try:
        val = float(price_lakhs)
    except (ValueError, TypeError):
        return "₹ 0.00"

    # Clamp negative or unrealistically small values
    if val <= 0:
        val = 10.0

    if val >= 100.0:
        crores = val / 100.0
        return f"₹ {crores:.2f} Crore"
    else:
        return f"₹ {val:.2f} Lakhs"


def get_supported_locations():
    """Returns sorted list of validated Bengaluru locations from dataset."""
    _, loc_data, _ = load_artifacts()
    return sorted(loc_data.get("supported_locations", []))


def get_supported_area_types():
    """Returns valid property area types."""
    _, _, meta = load_artifacts()
    return meta.get("supported_area_types", [
        "Super built-up Area",
        "Built-up Area",
        "Plot Area",
        "Carpet Area"
    ])


def get_supported_availabilities():
    """Returns valid property availability statuses."""
    _, _, meta = load_artifacts()
    return meta.get("supported_availabilities", [
        "Ready To Move",
        "Under Construction",
        "Immediate Possession"
    ])


def get_model_metrics():
    """Returns accuracy and evaluation metrics of trained regression models."""
    _, _, meta = load_artifacts()
    return meta.get("metrics", {}), meta.get("all_model_metrics", {}), meta.get("best_model_name", "Regression Model")


def get_dataset_summary():
    """Returns summary dataset analytics for frontend dashboard."""
    _, _, meta = load_artifacts()
    return meta.get("dataset_summary", {})


def predict_property_price(location, total_sqft, area_type, availability, society_presence=0):
    """
    Generates predicted property price based solely on location and surrounding features.

    Parameters:
      - location (str): Bengaluru locality name
      - total_sqft (float/int/str): Property total area in sq.ft.
      - area_type (str): Property area type category
      - availability (str): Possession status
      - society_presence (int/bool/str): 1 if in registered society/gated community, else 0

    Returns:
      dict containing formatted price, numeric values, and validated property details.
    """
    model, loc_data, meta = load_artifacts()
    supported_locations = loc_data.get("supported_locations", [])
    loc_freq_map = loc_data.get("location_frequency_map", {})
    default_freq = loc_data.get("default_frequency", 25)

    # 1. Location Validation (Strict handling of unknown locations)
    if not location or not str(location).strip():
        raise ValueError("Please select a valid Bengaluru location.")

    clean_loc = str(location).strip()

    # Exact or case-insensitive matching against supported locations
    matched_loc = None
    for loc in supported_locations:
        if clean_loc.lower() == loc.lower():
            matched_loc = loc
            break

    if not matched_loc:
        raise ValueError("Location not available in the trained Bengaluru dataset. Please select a supported location.")

    # 2. Total Sqft Area Validation
    try:
        sqft_val = float(total_sqft)
    except (ValueError, TypeError):
        raise ValueError("Please enter a valid property area in square feet.")

    if sqft_val <= 0:
        raise ValueError("Property area must be greater than 0 square feet.")

    if sqft_val < 100 or sqft_val > 50000:
        raise ValueError("Property area must be between 100 and 50,000 square feet.")

    # 3. Area Type Validation & Standardization
    valid_area_types = meta.get("supported_area_types", [])
    matched_area_type = None
    clean_area_str = str(area_type).strip() if area_type else ""
    for at in valid_area_types:
        if clean_area_str.lower() == at.lower():
            matched_area_type = at
            break
    if not matched_area_type:
        matched_area_type = valid_area_types[0] if valid_area_types else "Super built-up Area"

    # 4. Availability Validation
    valid_availabilities = meta.get("supported_availabilities", [])
    matched_availability = None
    clean_avail_str = str(availability).strip() if availability else ""
    for av in valid_availabilities:
        if clean_avail_str.lower() == av.lower():
            matched_availability = av
            break
    if not matched_availability:
        matched_availability = "Ready To Move"

    # 5. Society Presence normalization
    if str(society_presence).lower() in ["1", "true", "yes", "on", "registered"]:
        society_val = 1
    else:
        society_val = 0

    # 6. Derive location frequency feature
    loc_freq = loc_freq_map.get(matched_loc, default_freq)

    # 7. Construct Feature DataFrame matching exact training schema
    # Features: ['location_clean', 'total_sqft_numeric', 'area_type', 'availability_category', 'society_presence', 'location_frequency']
    input_data = pd.DataFrame([{
        "location_clean": matched_loc,
        "total_sqft_numeric": sqft_val,
        "area_type": matched_area_type,
        "availability_category": matched_availability,
        "society_presence": society_val,
        "location_frequency": loc_freq
    }])

    # 8. Predict Price (Model outputs price in Lakhs)
    raw_pred = model.predict(input_data)[0]

    # 9. Safety Clamping: Ensure price is never negative or absurdly zero
    # Calculate approximate lower bound based on minimum Bengaluru land/construction cost
    min_reasonable_price = max(10.0, round(sqft_val * 0.0015, 2))
    predicted_lakhs = max(float(raw_pred), min_reasonable_price)
    predicted_lakhs = round(predicted_lakhs, 2)

    predicted_inr = int(round(predicted_lakhs * 100000))
    formatted_str = format_indian_currency(predicted_lakhs)
    price_per_sqft = round((predicted_lakhs * 100000.0) / sqft_val, 2)

    return {
        "success": True,
        "predicted_price_lakhs": predicted_lakhs,
        "predicted_price_inr": predicted_inr,
        "formatted_price": formatted_str,
        "location": matched_loc,
        "total_sqft": sqft_val,
        "area_type": matched_area_type,
        "availability": matched_availability,
        "society_presence": society_val,
        "society_display": "Registered Society / Gated Community" if society_val == 1 else "Standalone / Independent Property",
        "location_frequency": loc_freq,
        "price_per_sqft": price_per_sqft,
        "model_name": meta.get("best_model_name", "Trained ML Model")
    }


if __name__ == "__main__":
    print("Testing prediction module...")
    try:
        sample = predict_property_price(
            location="Whitefield",
            total_sqft=1200,
            area_type="Super built-up Area",
            availability="Ready To Move",
            society_presence=1
        )
        print("Sample Prediction Result:")
        for k, v in sample.items():
            print(f"  {k}: {str(v).encode('ascii', 'replace').decode('ascii')}")
    except Exception as e:
        print(f"Prediction test failed: {e}")
