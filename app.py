import sys
import os
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.logger.logging import logging
from LoanPrediction.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from LoanPrediction.utils.common import load_object

from LoanPrediction.utils.model.estimator import LoanModel


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise LoanException(e, sys.exc_info())


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        # print(df)
        preprocesor = load_object("final_models/preprocessing.pkl")
        final_model = load_object("final_models/model.pkl")
        model = LoanModel(preprocessor=preprocesor, model=final_model)
        print(df.iloc[0])
        y_pred = model.predict(df)
        print(y_pred)
        df["predicted_column"] = y_pred
        print(df["predicted_column"])
        # df['predicted_column'].replace(-1, 0)
        # return df.to_json()
        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes="table table-striped")
        # print(table_html)
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )

    except Exception as e:
        raise LoanException(e, sys.exc_info())


if __name__ == "__main__":
    app_run(app, host="127.0.0.1", port=8000)
