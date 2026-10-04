# app.py
# Flask Web Application for Smart Real Estate Price Predictor – Bengaluru
# Provides web routes, REST API, SQLite history, and interactive analytics dashboard.
# Strictly excludes all bedroom, bathroom, BHK, and balcony features.

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import prediction engine and database handlers
from predict import (
    predict_property_price,
    get_supported_locations,
    get_supported_area_types,
    get_supported_availabilities,
    get_model_metrics,
    get_dataset_summary,
    load_artifacts
)
from database import (
    init_db,
    save_prediction_record,
    fetch_prediction_history,
    clear_all_predictions
)

app = Flask(__name__)
app.secret_key = "smart-bengaluru-real-estate-predictor-secret-key"

# Ensure database is initialized at startup
init_db()


def check_model_readiness():
    """Verifies that model artifacts exist before servicing requests."""
    try:
        load_artifacts()
        return True, ""
    except Exception as e:
        return False, str(e)


# =====================================================================
# WEB ROUTES
# =====================================================================

@app.route("/")
def index():
    """Landing page featuring project highlights, KPI metrics, and analytics charts."""
    model_ready, _ = check_model_readiness()

    if model_ready:
        metrics, all_models, best_model = get_model_metrics()
        dataset_summary = get_dataset_summary()
        supported_locations = get_supported_locations()
    else:
        metrics = {"r2_score": 0.0, "mae": 0.0, "rmse": 0.0}
        all_models = {}
        best_model = "Not Trained Yet"
        dataset_summary = {
            "total_properties": 0,
            "unique_locations": 0,
            "avg_price_lakhs": 0.0,
            "avg_price_sqft": 0.0
        }
        supported_locations = []

    return render_template(
        "index.html",
        model_ready=model_ready,
        metrics=metrics,
        all_models=all_models,
        best_model=best_model,
        summary=dataset_summary,
        locations=supported_locations[:20]  # Featured locations
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form page (GET) and form submission handler (POST)."""
    model_ready, err = check_model_readiness()
    if not model_ready:
        flash("Machine learning model not found! Please run 'python train_model.py' to generate the model.", "danger")
        return render_template("predict.html", model_ready=False, locations=[], area_types=[], availabilities=[])

    locations = get_supported_locations()
    area_types = get_supported_area_types()
    availabilities = get_supported_availabilities()

    if request.method == "POST":
        location = request.form.get("location", "").strip()
        total_sqft = request.form.get("total_sqft", "").strip()
        area_type = request.form.get("area_type", "").strip()
        availability = request.form.get("availability", "").strip()
        society_presence = request.form.get("society_presence", "0").strip()

        try:
            # Perform prediction using ML pipeline
            prediction = predict_property_price(
                location=location,
                total_sqft=total_sqft,
                area_type=area_type,
                availability=availability,
                society_presence=society_presence
            )

            # Store prediction in SQLite history
            save_prediction_record(
                location=prediction["location"],
                total_sqft=prediction["total_sqft"],
                area_type=prediction["area_type"],
                availability=prediction["availability"],
                society_status=prediction["society_display"],
                price_lakhs=prediction["predicted_price_lakhs"],
                formatted_price=prediction["formatted_price"]
            )

            return render_template("result.html", result=prediction)

        except ValueError as ve:
            flash(str(ve), "danger")
            return render_template(
                "predict.html",
                model_ready=True,
                locations=locations,
                area_types=area_types,
                availabilities=availabilities,
                form_data=request.form
            )
        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}", "danger")
            return render_template(
                "predict.html",
                model_ready=True,
                locations=locations,
                area_types=area_types,
                availabilities=availabilities,
                form_data=request.form
            )

    return render_template(
        "predict.html",
        model_ready=True,
        locations=locations,
        area_types=area_types,
        availabilities=availabilities
    )


@app.route("/result")
def result():
    """Result page view fallback."""
    flash("Please submit a property detail form to view an estimated price.", "info")
    return redirect(url_for("predict"))


@app.route("/history")
def history():
    """Displays historical real estate price predictions stored in SQLite."""
    records = fetch_prediction_history(limit=100)
    return render_template("history.html", records=records)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clears all stored prediction records from SQLite database."""
    clear_all_predictions()
    flash("Prediction history has been successfully cleared.", "success")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """About page detailing ML architecture, dataset, and location focus."""
    model_ready, _ = check_model_readiness()
    metrics, all_models, best_model = get_model_metrics() if model_ready else ({}, {}, "N/A")
    summary = get_dataset_summary() if model_ready else {}

    return render_template(
        "about.html",
        metrics=metrics,
        all_models=all_models,
        best_model=best_model,
        summary=summary
    )


# =====================================================================
# REST API ENDPOINTS
# =====================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for real estate price prediction.
    Accepts JSON:
      {
        "location": "Whitefield",
        "total_sqft": 1200,
        "area_type": "Super built-up Area",
        "availability": "Ready To Move",
        "society": 1
      }
    """
    model_ready, _ = check_model_readiness()
    if not model_ready:
        return jsonify({
            "success": False,
            "error": "Machine learning model is not available. Please run train_model.py first."
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "Invalid request body. Expected JSON payload."
        }), 400

    location = data.get("location")
    total_sqft = data.get("total_sqft")
    area_type = data.get("area_type")
    availability = data.get("availability")
    society_presence = data.get("society", 0)

    try:
        prediction = predict_property_price(
            location=location,
            total_sqft=total_sqft,
            area_type=area_type,
            availability=availability,
            society_presence=society_presence
        )

        # Record API prediction in database
        save_prediction_record(
            location=prediction["location"],
            total_sqft=prediction["total_sqft"],
            area_type=prediction["area_type"],
            availability=prediction["availability"],
            society_status=prediction["society_display"],
            price_lakhs=prediction["predicted_price_lakhs"],
            formatted_price=prediction["formatted_price"]
        )

        return jsonify({
            "success": True,
            "predicted_price_lakhs": prediction["predicted_price_lakhs"],
            "predicted_price_inr": prediction["predicted_price_inr"],
            "formatted_price": prediction["formatted_price"],
            "price_per_sqft": prediction["price_per_sqft"],
            "location": prediction["location"],
            "total_sqft": prediction["total_sqft"],
            "area_type": prediction["area_type"],
            "availability": prediction["availability"],
            "society_presence": prediction["society_presence"],
            "model_used": prediction["model_name"]
        }), 200

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal prediction error: {str(e)}"
        }), 500


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """
    Returns aggregated statistics derived from the clean dataset
    for rendering interactive Chart.js charts on the frontend.
    """
    cleaned_csv = Path(__file__).resolve().parent / "dataset" / "cleaned_bengaluru_data.csv"
    if not cleaned_csv.exists():
        return jsonify({"success": False, "error": "Cleaned dataset not found."}), 404

    import pandas as pd
    df = pd.read_csv(cleaned_csv)

    # 1. Top 10 locations by average price (with at least 15 properties)
    loc_stats = df[df["location_clean"] != "Other"].groupby("location_clean").agg(
        avg_price=("price", "mean"),
        avg_sqft=("total_sqft_numeric", "mean"),
        count=("price", "count"),
        avg_price_sqft=("price_per_sqft", "mean")
    ).reset_index()

    top_by_price = loc_stats[loc_stats["count"] >= 15].sort_values(
        by="avg_price", ascending=False
    ).head(10)

    # 2. Top 10 most popular locations by property count
    top_by_count = loc_stats.sort_values(
        by="count", ascending=False
    ).head(10)

    # 3. Area type distribution
    area_type_counts = df["area_type"].value_counts().to_dict()

    # 4. Availability distribution
    avail_counts = df["availability_category"].value_counts().to_dict()

    return jsonify({
        "success": True,
        "top_by_price": {
            "locations": top_by_price["location_clean"].tolist(),
            "avg_prices": [round(p, 2) for p in top_by_price["avg_price"].tolist()],
            "avg_price_sqft": [round(p, 2) for p in top_by_price["avg_price_sqft"].tolist()]
        },
        "top_by_count": {
            "locations": top_by_count["location_clean"].tolist(),
            "counts": top_by_count["count"].tolist(),
            "avg_prices": [round(p, 2) for p in top_by_count["avg_price"].tolist()]
        },
        "area_types": {
            "labels": list(area_type_counts.keys()),
            "counts": list(area_type_counts.values())
        },
        "availabilities": {
            "labels": list(avail_counts.keys()),
            "counts": list(avail_counts.values())
        }
    }), 200


# =====================================================================
# ERROR HANDLERS
# =====================================================================

@app.errorhandler(404)
def page_not_found(e):
    """Friendly 404 page handler."""
    return render_template("index.html", error_message="Page not found. Redirected to home."), 404


@app.errorhandler(500)
def server_error(e):
    """Friendly 500 page handler."""
    return render_template("index.html", error_message="An internal server error occurred. Please try again."), 500


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("STARTING SMART REAL ESTATE PRICE PREDICTOR FLASK SERVER")
    print("Open http://127.0.0.1:5000 in your web browser")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
