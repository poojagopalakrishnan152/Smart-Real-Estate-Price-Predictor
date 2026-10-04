# preprocessing.py (Root Launcher)
import sys
import os
from pathlib import Path

subfolder = Path(__file__).resolve().parent / "Smart_Real_Estate_Price_Predictor"
if subfolder.exists():
    os.chdir(subfolder)
    sys.path.insert(0, str(subfolder))

from preprocessing import preprocess_data

if __name__ == "__main__":
    preprocess_data()
