from flask import Flask, render_template, request
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Load model once when Flask starts
pipeline = PredictPipeline()


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=pipeline.model_name,
        threshold=pipeline.threshold
    )


@app.route("/main")
def main():
    return render_template("home.html")


@app.route("/predict-form", methods=["POST"])
def predict_form():

    try:
        # Get all form fields
        transaction = {}

        for key, value in request.form.items():
            transaction[key] = float(value)

        # Run prediction
        result = pipeline.predict_one(transaction)

        # Show result on the same page
        return render_template(
            "index.html",
            model_name=pipeline.model_name,
            threshold=pipeline.threshold,
            result=result
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            model_name=pipeline.model_name,
            threshold=pipeline.threshold
        ), 400


if __name__ == "__main__":
    app.run(debug=True)