from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Student Performance Prediction API",
    version="1.0"
)

# ================= LOAD FILES =================

try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ================= REQUEST MODEL =================

class PredictionRequest(BaseModel):
    grade_level: str
    ai_tools_used: str
    ai_usage_purpose: str

    last_exam_score: float = Field(..., ge=0, le=100)
    assignment_scores_avg: float = Field(..., ge=0, le=100)

    # User friendly scale
    concept_understanding_score: float = Field(..., ge=1, le=10)

class PredictionResponse(BaseModel):
    prediction: int
    probability_pass: float
    probability_fail: float

# ================= ROOT =================

@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API Running"
    }

# ================= PREDICT =================

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    try:

        input_df = pd.DataFrame([data.model_dump()])

        # Encode categorical columns
        for col, encoder in label_encoders.items():

            if col in input_df.columns:

                value = str(input_df[col].iloc[0])

                if value not in encoder.classes_:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value '{value}' for '{col}'"
                    )

                input_df[col] = encoder.transform(input_df[col])

        # Scale only existing numeric columns
        scale_cols = [
            col for col in [
                "last_exam_score",
                "assignment_scores_avg",
                "concept_understanding_score"
            ]
            if col in input_df.columns
        ]

        if scale_cols:
            input_df[scale_cols] = scaler.transform(
                input_df[scale_cols]
            )

        # Match training column order
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = int(model.predict(input_df)[0])

        # Safe probability handling
        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(input_df)[0]

            probability_fail = float(probs[0])
            probability_pass = float(probs[1])

        else:

            probability_pass = 1.0 if prediction == 1 else 0.0
            probability_fail = 1.0 - probability_pass

        return PredictionResponse(
            prediction=prediction,
            probability_pass=probability_pass,
            probability_fail=probability_fail
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Error: {str(e)}"
        )

# ================= RUN =================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )