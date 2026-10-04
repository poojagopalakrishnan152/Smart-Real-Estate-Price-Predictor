# preprocessing.py
# Data Preprocessing Module for Smart Real Estate Price Predictor – Bengaluru
# Loads raw dataset, cleans location, total_sqft, removes outliers and duplicates,
# and strictly removes all bedroom/bathroom/BHK/balcony features.

import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Define base paths using pathlib for robust cross-platform compatibility
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_PATH = DATASET_DIR / "Bengaluru_House_Data.csv"
CLEANED_DATA_PATH = DATASET_DIR / "cleaned_bengaluru_data.csv"

# STRICT RULE: Features related to bedrooms, bathrooms, BHK, or balconies that must be completely dropped
FORBIDDEN_FEATURES = [
    "bath",
    "bathroom",
    "bathrooms",
    "bhk",
    "BHK",
    "bedroom",
    "bedrooms",
    "balcony",
    "size",           # Contains values like '2 BHK', '4 Bedroom'
    "room count",
    "washroom",
    "washrooms"
]


def clean_sqft_value(val):
    """
    Converts diverse total_sqft strings to a clean float.
    Handles:
      - Normal floats: '1200' -> 1200.0
      - Ranges: '1200 - 1400' -> 1300.0 (average)
      - Units: '34.46Sq. Meter', '1000Sq. Yards', 'Acres', 'Perch', 'Guntha'
    Returns np.nan if conversion is not possible.
    """
    if pd.isnull(val):
        return np.nan

    val_str = str(val).strip()

    # Handle range values e.g., '2100 - 2850'
    tokens = val_str.split("-")
    if len(tokens) == 2:
        try:
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2.0
        except Exception:
            return np.nan

    # Try direct float conversion
    try:
        return float(val_str)
    except Exception:
        # Match number and unit
        m = re.match(r"^([\d\.]+)\s*([a-zA-Z\.\s]+)$", val_str)
        if m:
            try:
                num = float(m.group(1))
                unit = m.group(2).lower().replace(" ", "").replace(".", "")
                if "sqmeter" in unit:
                    return num * 10.7639
                elif "sqyard" in unit:
                    return num * 9.0
                elif "acre" in unit:
                    return num * 43560.0
                elif "perch" in unit:
                    return num * 272.25
                elif "guntha" in unit:
                    return num * 1089.0
                elif "ground" in unit:
                    return num * 2400.0
                elif "cent" in unit:
                    return num * 435.6
            except Exception:
                return np.nan
        return np.nan


def categorize_availability(val):
    """
    Groups raw availability entries into 3 standard business categories:
    - 'Ready To Move'
    - 'Under Construction'
    - 'Immediate Possession'
    """
    if pd.isnull(val):
        return "Ready To Move"
    s = str(val).strip()
    if s == "Ready To Move":
        return "Ready To Move"
    elif "Immediate" in s:
        return "Immediate Possession"
    else:
        return "Under Construction"


def clean_area_type(val):
    """
    Normalizes area_type spacing and casing.
    e.g. 'Super built-up  Area' -> 'Super built-up Area'
    """
    if pd.isnull(val):
        return "Super built-up Area"
    cleaned = re.sub(r"\s+", " ", str(val).strip())
    return cleaned


def preprocess_data(raw_csv_path=None, save_cleaned=True, verbose=True):
    """
    Executes the complete data preprocessing pipeline.
    Returns cleaned pandas DataFrame ready for ML modeling.
    """
    target_path = Path(raw_csv_path) if raw_csv_path else RAW_DATA_PATH

    # 1. Check whether dataset exists
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {target_path}. Please place Bengaluru_House_Data.csv in the dataset/ directory."
        )

    if verbose:
        print("=" * 60)
        print("SMART REAL ESTATE PREDICTOR - DATA PREPROCESSING")
        print("=" * 60)
        print(f"Loading dataset from: {target_path}")

    df = pd.read_csv(target_path)

    # 2. Display basic dataset information
    if verbose:
        print(f"\nInitial Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns present:", df.columns.tolist())
        print("\nMissing values per column before cleaning:")
        print(df.isnull().sum().to_string())

    # 3. STRICT ENFORCEMENT: Completely drop forbidden bedroom/bathroom/BHK/balcony features
    dropped_cols = []
    for col in df.columns:
        col_lower = col.strip().lower()
        if col in FORBIDDEN_FEATURES or col_lower in [f.lower() for f in FORBIDDEN_FEATURES]:
            dropped_cols.append(col)

    if dropped_cols:
        df = df.drop(columns=dropped_cols)
        if verbose:
            print(f"\n[STRICT RULE] Dropped forbidden bedroom/bathroom features: {dropped_cols}")

    # 4. Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"Removed {initial_rows - len(df)} duplicate rows. Remaining: {len(df)}")

    # 5. Clean location column
    # Location is required for location-based prediction
    df = df.dropna(subset=["location"])
    df["location"] = df["location"].astype(str).str.strip()
    df = df[~df["location"].isin(["", "nan", "None", "NaN"])]

    # 6. Clean total_sqft and convert to numeric
    df["total_sqft_numeric"] = df["total_sqft"].apply(clean_sqft_value)
    df = df.dropna(subset=["total_sqft_numeric", "price"])

    # 7. Clean and standardize area_type
    df["area_type"] = df["area_type"].apply(clean_area_type)

    # 8. Clean and categorize availability
    df["availability_category"] = df["availability"].apply(categorize_availability)

    # 9. Derive society presence feature (1 if present, 0 if null/empty)
    if "society" in df.columns:
        df["society_presence"] = df["society"].apply(
            lambda x: 1 if pd.notnull(x) and str(x).strip() not in ["", "nan", "None"] else 0
        )
    else:
        df["society_presence"] = 0

    # 10. Outlier Removal: Filter realistic total_sqft (e.g. 250 sqft to 15,000 sqft)
    before_outlier = len(df)
    df = df[(df["total_sqft_numeric"] >= 250) & (df["total_sqft_numeric"] <= 15000)]

    # 11. Calculate temporary price_per_sqft for outlier removal ONLY (price is in Lakhs, 1 Lakh = 100,000 Rs)
    # WARNING: price_per_sqft is NEVER used as an input feature for training (prevents target leakage)
    df["price_per_sqft"] = (df["price"] * 100000.0) / df["total_sqft_numeric"]

    # Remove extreme price per sqft anomalies (e.g. < 1,000 Rs/sqft or > 35,000 Rs/sqft)
    df = df[(df["price_per_sqft"] >= 1000) & (df["price_per_sqft"] <= 35000)]
    if verbose:
        print(f"Removed {before_outlier - len(df)} unrealistic/outlier rows.")

    # 12. Standardize locations: keep locations with at least 10 entries to avoid sparse noisy categories
    loc_freq = df["location"].value_counts()
    frequent_locations = set(loc_freq[loc_freq >= 10].index.tolist())
    df["location_clean"] = df["location"].apply(lambda x: x if x in frequent_locations else "Other")

    # 13. Derive location_frequency (property density/popularity of locality)
    loc_clean_freq = df["location_clean"].value_counts()
    df["location_frequency"] = df["location_clean"].map(loc_clean_freq)

    # Reorder clean columns
    clean_columns = [
        "location",
        "location_clean",
        "location_frequency",
        "total_sqft_numeric",
        "area_type",
        "availability_category",
        "society_presence",
        "price_per_sqft",
        "price"
    ]
    df_clean = df[clean_columns].copy()

    if verbose:
        print("\nCleaned Dataset Summary:")
        print(f"- Total Valid Records: {len(df_clean)}")
        print(f"- Unique Clean Locations (>=10 entries): {len(frequent_locations)}")
        print(f"- Average Property Price: Rs. {df_clean['price'].mean():.2f} Lakhs")
        print(f"- Average Price per Sq.Ft.: Rs. {df_clean['price_per_sqft'].mean():.2f}")
        print(f"- Total Sq.Ft. Range: {df_clean['total_sqft_numeric'].min()} - {df_clean['total_sqft_numeric'].max()} sq.ft.")
        print(f"- Area Types: {df_clean['area_type'].unique().tolist()}")
        print(f"- Availability Options: {df_clean['availability_category'].unique().tolist()}")

    # 14. Save cleaned dataset
    if save_cleaned:
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(CLEANED_DATA_PATH, index=False)
        if verbose:
            print(f"\nCleaned dataset saved successfully to: {CLEANED_DATA_PATH}")

    if verbose:
        print("=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)

    return df_clean


if __name__ == "__main__":
    preprocess_data()
