# train_model.py (Root Launcher)
import sys
import os
from pathlib import Path

subfolder = Path(__file__).resolve().parent / "Smart_Real_Estate_Price_Predictor"
if subfolder.exists():
    os.chdir(subfolder)
    sys.path.insert(0, str(subfolder))

from train_model import train_and_evaluate_models

if __name__ == "__main__":
    train_and_evaluate_models()
