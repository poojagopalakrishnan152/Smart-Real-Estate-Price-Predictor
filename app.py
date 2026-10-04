# app.py (Root Launcher)
# Automatically runs the Smart Real Estate Price Predictor Flask app

import sys
import os
from pathlib import Path

# Add project subfolder to path and switch working directory
subfolder = Path(__file__).resolve().parent / "Smart_Real_Estate_Price_Predictor"
if subfolder.exists():
    os.chdir(subfolder)
    sys.path.insert(0, str(subfolder))

from app import app

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("STARTING SMART REAL ESTATE PRICE PREDICTOR FLASK SERVER")
    print("Open http://127.0.0.1:5000 in your web browser")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
