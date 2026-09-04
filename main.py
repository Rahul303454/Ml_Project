from fastapi import FastAPI
import joblib
import pandas as pd
import os

from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood"
]


# Load model
model = joblib.load(
    os.path.join(
        BASE_DIR,
        "Model_Pipeline.pkl"
    )
)


class Features(BaseModel):

    latitude: float = Field(..., ge=-90, le=90)

    longitude: float = Field(..., ge=-180, le=180)

    price: float = Field(..., ge=0)

    minimum_nights: int = Field(..., ge=1, le=365)

    number_of_reviews: int = Field(..., ge=0)

    reviews_per_month: float = Field(..., ge=0)

    calculated_host_listings_count: int = Field(..., ge=0)

    availability_365: int = Field(..., ge=0, le=365)

    neighbourhood_group: str = Field(..., min_length=1)

    neighbourhood: str = Field(..., min_length=1)


# Serve Frontend
@app.get("/")
def read_root():

    return FileResponse(
        os.path.join(
            BASE_DIR,
            "index.html"
        )
    )


@app.post("/predict")
def predict(features: Features):

    row = pd.DataFrame(
        [features.model_dump()],
        columns=COLUMNS
    )

    prediction = model.predict(row)

    probability = model.predict_proba(row)

    return {
        "Predicted_room_type": prediction[0],
        "Prediction_probability": probability.tolist()[0]
    }