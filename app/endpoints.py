import io
import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import logging

from src.inference import make_prediction
from src.preprocess import preprocess_data
from app.config import model

# logger and router
logger = logging.getLogger("app.endpoints")
router = APIRouter()
templates = Jinja2Templates(directory="templates")  

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Received request for the index page")
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    logger.info(f"Upload endpoint called for file: {file.filename}")

    try:
        # read only the first 500 rows from the uploaded CSV
        df = pd.read_csv(file.file, nrows=500)
        logger.info(f"Read first 500 rows of the file: {df.shape[0]} rows")

        # preprocess data and make predictions
        df_processed = preprocess_data(df)
        predictions = make_prediction(df_processed, model)
        df["prediction"] = predictions
        df["row"] = df.index + 1  
        records = df[["row", "prediction"]].to_dict(orient="records")
        logger.info("Prediction and table conversion successful")

        return templates.TemplateResponse("results.html", {"request": request, "records": records})

    except Exception as e:
        logger.error(f"Error in upload_file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/predict")
async def predict(request: Request):
    logger.info("Predict endpoint called")
    try:
        # parse JSON input and preprocess
        data = await request.json()
        df = pd.DataFrame(data)
        df = preprocess_data(df)
        
        # run inference
        predictions = make_prediction(df, model)
        logger.info("Prediction completed successfully")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
