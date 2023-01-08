import os
import joblib
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List
import logging
import sys
import json
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from src.api.schemas import UserCreditData, PredictionInput
from src.models.predict_model_single import single_inference
from config.path_config import ROOT_DIR

MODEL_PATH = os.path.join(ROOT_DIR, "models", "LGBMClassifier_fourth_exp_tuned.joblib")
PARAMS_PATH = os.path.join(
    ROOT_DIR, "src", "experiment_config", "fourth_exp_tuned.yaml"
)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)

logger.addHandler(screen_handler)
logger.setLevel(logging.INFO)

# Define application
app = FastAPI(
    title="Credit Scoring API",
    description="Will Someone Experimence Serious Deliquency in the next 2 Years",
    version="0.1",
)





def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/")
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict_single", tags=["Prediction"])
@construct_response
def _predict_single(request: Request, user_data: PredictionInput) -> Dict:
    print("checkpoint")
    model_output_temp = list()
    request_data = user_data.dict()["model_input"]
    print("user data : ", type(request_data))
    if isinstance(request_data, list):
        for data in request_data:
            customer_id = data["customer_id"]
            del data["customer_id"]
            output = single_inference(
                **data, model_path=MODEL_PATH, params_path=PARAMS_PATH
            )
            output["data"]["customer_id"] = customer_id
            model_output_temp.append(output)

    else:
        customer_id = request_data["customer_id"]
        del request_data["customer_id"]
        output = single_inference(
            **request_data, model_path=MODEL_PATH, params_path=PARAMS_PATH
        )
        output["data"]["customer_id"] = customer_id
        model_output_temp.append(output)
    logger.info("Successfully Running Inference")
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": model_output_temp,
    }
    return response


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=5000, log_level="info")
