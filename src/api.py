import pandas as pd
import util as utils
from pydantic import BaseModel
import numpy as np
import data_pipeline as data_pipeline
from data_preprocess import feature_engineering, transform_standardize, transform_ohe

from fastapi import FastAPI, File, UploadFile
import uvicorn

# Load config data
config_data = utils.load_config()

class Model:
    def __init__(self):
        """Initialize preprocessor, model"""
        self.preprocessor = utils.pickle_load(config_data['preprocessor_path'])
        self.model = utils.pickle_load(config_data['best_model_path'])

    def preprocess(self, X):
        """Function to preprocess data"""
        X = X.copy()
        
        X_fe = feature_engineering(X)
        
        X_clean_num = transform_standardize(data = X[config_data['numerical_columns']],
                                        standardizer = self.preprocessor['standardizer'])
        
        X_clean_cat = transform_ohe(data = X[config_data['cat_columns']],
                                        ohe = self.preprocessor['ohe'])
        X_clean = pd.concat([X_clean_num, X_clean_cat], axis = 1)
        
        return X_clean
    
    def predict(self, X):
        """Function to predict the data"""
        # Preprocess data
        X_clean = self.preprocess(X)

        # Predict data
        y_pred = self.model.predict(X_clean)

        # Predict dictionary
        y_pred_dict = {'label': [int(i) for i in y_pred]}
        return y_pred_dict
    
class api_data(BaseModel):
    Company : object 
    TypeName : object 
    OpSys : object 
    Cpu : object 
    Inches : float
    Ram : object 
    Memory : object 
    Gpu : object 
    Weight : object 
    ScreenResolution : object 
    HDD : int
    SSD : int
    Hybrid : int
    Flash_Storage : int

# FASTAPI
app = FastAPI()

@app.get('/')
def home():
    return "Hello, FastAPI up!"

@app.post('/predict')

# def create_upload_file(file: UploadFile = File(...)):
#     # Hanlde the file only if it is a csv
#     if file.filename.endswith('.csv'):
        
#         # Read file
#         with open(file.filename, 'wb') as f:
#             f.write(file.file.read())
            
#         data = pd.read_csv(file.filename)

#         # Modeling file
#         model = Model()
        
#         y_pred = model.predict(data)

#         return y_pred
    
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)  # type: ignore
    data.columns = config_data["predictors"]
    
    # Data validation, convert dtype
    data = pd.concat(
        [
            data[config_data["predictors"][:4]].astype(str), 
            data[config_data["predictors"][4]].astype(np.float64), 
            data[config_data["predictors"][5:10]].astype(str), 
            data[config_data["predictors"][10:]].astype(np.int64) 
        ],
        axis = 1
    )

    # Check company range data
    try:
        data_pipeline.check_data(data, config_data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Modeling file
    model = Model()
    
    # Predict data
    y_pred = model.predict(data)

    return {"res" : y_pred, "error_msg": ""}

if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)