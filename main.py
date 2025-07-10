from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os , sys
from sensor.logger import logging
from typing import cast

from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import load_object
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping, SensorModel
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
import os
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

from fastapi.responses import JSONResponse
from  fastapi import FastAPI
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, File, UploadFile, Response
import pandas as pd
import numpy as np


app = FastAPI()



origins = ["*"]
#Cross-Origin Resource Sharing (CORS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/",tags=["authentication"])
async def  index():
    return RedirectResponse(url="/docs")





@app.get("/train")
async def train():
    try:

        training_pipeline = TrainPipeline()

        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        
        training_pipeline.run_pipeline()
        return Response("Training successfully completed!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
        




@app.get("/predict")
async def predict():
    try:
        # Step 1: Load CSV file
        file_path = "final_prediction_input.csv"
        df = pd.read_csv(file_path)
        df.replace("na", np.nan, inplace=True)
        df.fillna(0, inplace=True)  # or use dropna()


        # Step 2: Load model
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return JSONResponse(content={"error": "Model is not available"}, status_code=404)

        best_model_path = model_resolver.get_best_model_path()
        model = cast(SensorModel, load_object(best_model_path))

        # Step 3: Predict
        y_pred = model.predict(df)
        y_pred = y_pred.astype(int)
        df['predicted_column'] = y_pred
        df['predicted_column'] = df['predicted_column'].replace(TargetValueMapping().reverse_mapping())


        # Step 4: Return top 5 prediction rows
        return JSONResponse(content=df.head().to_dict(orient="records"))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    





def main():
    try:
            
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)



if __name__ == "__main__":

    # file_path="/Users/myhome/Downloads/sensorlive/aps_failure_training_set1.csv"
    # database_name="ineuron"
    # collection_name ="sensor"
    # dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)
    app_run(app ,host=APP_HOST,port=APP_PORT)









    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)