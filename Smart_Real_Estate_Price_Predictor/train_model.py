# train_model.py
# Machine Learning Training Module for Smart Real Estate Price Predictor – Bengaluru
# Trains regression models to predict property price based on location and surroundings.
# Strictly excludes all bedroom, bathroom, BHK, and balcony information.

import sys
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import preprocessing logic
from preprocessing import preprocess_data, CLEANED_DATA_PATH, RAW_DATA_PATH

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "real_estate_model.pkl"
LOCATION_FILE = MODEL_DIR / "location_encoder.pkl"
FEATURES_FILE = MODEL_DIR / "model_features.pkl"


def train_and_evaluate_models(cleaned_csv_path=None):
    """
    Trains multiple regression models, compares their performance,
    selects the best model, and saves all required model artifacts.
    """
    print("=" * 65)
    print("SMART REAL ESTATE PREDICTOR - MODEL TRAINING & EVALUATION")
    print("=" * 65)

    # 1. Load cleaned data; if not generated yet, run preprocessing
    data_path = Path(cleaned_csv_path) if cleaned_csv_path else CLEANED_DATA_PATH
    if not data_path.exists():
        print(f"Cleaned dataset not found at {data_path}. Running preprocessing first...")
        df = preprocess_data(RAW_DATA_PATH, save_cleaned=True, verbose=False)
    else:
        print(f"Loading cleaned dataset from: {data_path}")
        df = pd.read_csv(data_path)

    print(f"Total property records for modeling: {len(df)}")

    # 2. Define Features & Target (STRICTLY NO BEDROOM/BATHROOM/BHK/BALCONY)
    feature_cols = [
        "location_clean",
        "total_sqft_numeric",
        "area_type",
        "availability_category",
        "society_presence",
        "location_frequency"
    ]
    target_col = "price"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 3. Safe Train-Test Split (80% Train, 20% Test) to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples")

    # 4. Build Preprocessing ColumnTransformer
    categorical_features = ["location_clean", "area_type", "availability_category"]
    numerical_features = ["total_sqft_numeric", "society_presence", "location_frequency"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features
            ),
            (
                "num",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    # 5. Define Candidate Regression Models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            n_jobs=1
        )
    }

    results = {}
    best_model_name = None
    best_r2 = -float("inf")
    best_pipeline = None

    print("\nEvaluating Candidate Regression Models...")
    print("-" * 65)

    for name, model in candidate_models.items():
        print(f"Training {name}...", end=" ", flush=True)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        results[name] = {
            "r2": float(r2),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "pipeline": pipeline
        }

        print("Done!")
        print(f"  R2 Score : {r2 * 100:.2f}%")
        print(f"  MAE      : Rs. {mae:.2f} Lakhs")
        print(f"  RMSE     : Rs. {rmse:.2f} Lakhs")
        print("-" * 65)

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print("\n" + "=" * 65)
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"Validation R2 Score : {best_r2 * 100:.2f}%")
    print(f"Mean Absolute Error : Rs. {results[best_model_name]['mae']:.2f} Lakhs")
    print(f"Root Mean Sq Error  : Rs. {results[best_model_name]['rmse']:.2f} Lakhs")
    print("=" * 65)

    # 6. Prepare Metadata and Encoding Info for Predictor and Web UI
    # Build list of supported real locations from dataset (excluding 'Other' for primary dropdown)
    raw_supported_locations = sorted([
        loc for loc in df["location_clean"].unique() if loc != "Other"
    ])

    # Location frequency lookup table
    location_freq_map = df.set_index("location_clean")["location_frequency"].to_dict()
    default_freq = int(df["location_frequency"].median())

    location_metadata = {
        "supported_locations": raw_supported_locations,
        "location_frequency_map": location_freq_map,
        "default_frequency": default_freq
    }

    # Supported categories
    supported_area_types = sorted(df["area_type"].unique().tolist())
    supported_availabilities = sorted(df["availability_category"].unique().tolist())

    # Dataset summary statistics for dashboard
    dataset_summary = {
        "total_properties": int(len(df)),
        "unique_locations": int(len(raw_supported_locations)),
        "avg_price_lakhs": round(float(df["price"].mean()), 2),
        "avg_price_sqft": round(float(df["price_per_sqft"].mean()), 2),
        "min_sqft": float(df["total_sqft_numeric"].min()),
        "max_sqft": float(df["total_sqft_numeric"].max()),
        "area_types": supported_area_types,
        "availabilities": supported_availabilities
    }

    model_metadata = {
        "best_model_name": best_model_name,
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": feature_cols,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "supported_locations": raw_supported_locations,
        "supported_area_types": supported_area_types,
        "supported_availabilities": supported_availabilities,
        "metrics": {
            "r2_score": round(best_r2 * 100, 2),
            "mae": round(results[best_model_name]["mae"], 2),
            "mse": round(results[best_model_name]["mse"], 2),
            "rmse": round(results[best_model_name]["rmse"], 2)
        },
        "all_model_metrics": {
            k: {
                "r2_score": round(v["r2"] * 100, 2),
                "mae": round(v["mae"], 2),
                "rmse": round(v["rmse"], 2)
            } for k, v in results.items()
        },
        "dataset_summary": dataset_summary
    }

    # 7. Save Model Artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving best model pipeline to: {MODEL_FILE}")
    joblib.dump(best_pipeline, MODEL_FILE)

    print(f"Saving location encoder data to: {LOCATION_FILE}")
    joblib.dump(location_metadata, LOCATION_FILE)

    print(f"Saving feature metadata to: {FEATURES_FILE}")
    joblib.dump(model_metadata, FEATURES_FILE)

    print("\n" + "=" * 65)
    print("MODEL TRAINING & SERIALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return model_metadata


if __name__ == "__main__":
    train_and_evaluate_models()
