from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.endpoints import router as api_router
from app.logger import setup_logging
import logging

# initialise logging
setup_logging()
logger = logging.getLogger(__name__)

# create the FastAPI app
app = FastAPI(
    title="Edge Inference for IoV API",
    description="A FastAPI service for predicting attack instances using an XGBoost model.",
    version="1.0.0"
)

#  Mount static folder for CSS
app.mount("/static", StaticFiles(directory="style"), name="static")

# Include API endpoints 
app.include_router(api_router)

# exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again later."}
    )

#  run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
