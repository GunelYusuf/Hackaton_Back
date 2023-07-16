import uvicorn
from fastapi import FastAPI, UploadFile, File
from mobile_notes import MobileNotes
from credit_card import CreditCard
from PIL import Image
from io import BytesIO
import numpy as np
import pickle
import h5py
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Create the app object
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model for MobileNotes
pipeline = pickle.load(open("pipeline.pkl", "rb"))

# Load the pre-trained model for CreditCard
xgb = pickle.load(open("xgb_model.pkl", "rb"))

# Load the pre-trained model for image classification
model = keras.models.load_model("best_model.h5")

# Define the prediction function for image classification
def predict_image(image):
    image = image.resize((227, 227))
    image = image.convert("RGB")
    image = np.array(image)
    image = image.reshape((1, 227, 227, 3))
    prediction = np.argmax(model.predict(image))
    return prediction

# Index route
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Route with a single parameter
@app.get('/{name}')
def get_name(name: str):
    return {'hi': f'{name}'}

# Endpoint for MobileNotes prediction
@app.post('/predict_mobilenotes')
def predict_mobile_notes(data: MobileNotes):
    data = data.dict()
    # Extract data from the request
    features = [
        data['battery_power'], data['blue'], data['clock_speed'], data['dual_sim'],
        data['fc'], data['four_g'], data['int_memory'], data['m_dep'],
        data['mobile_wt'], data['n_cores'], data['pc'], data['px_height'],
        data['px_width'], data['ram'], data['sc_h'], data['sc_w'],
        data['talk_time'], data['three_g'], data['touch_screen'], data['wifi']
    ]
    prediction = pipeline.predict([features])[0]
    if round(prediction) == 0:
        prediction_text = "Low Cost"
    elif round(prediction) == 1:
        prediction_text = "Medium Cost"
    elif round(prediction) == 2:
        prediction_text = "High Cost"
    else:
        prediction_text = "Very High Cost"
    return {'prediction': prediction_text}

# Endpoint for CreditCard prediction
@app.post('/predict_creditcard')
async def predict_credit_card(data: CreditCard):
    # Extract data from the request
    features = [
        data.Customer_Age, data.Total_Relationship_Count,
        data.Months_Inactive_12_mon, data.Contacts_Count_12_mon,
        data.Credit_Limit, data.Total_Revolving_Bal, data.Avg_Open_To_Buy,
        data.Total_Amt_Chng_Q4_Q1, data.Avg_Utilization_Ratio,
        data.Blue, data.Gold, data.Platinum, data.Silver
    ]
    prediction = xgb.predict([features])[0]
    if round(prediction) == 0:
        prediction_text = "The Customer is not eligible for a Credit Card"
    else:
        prediction_text = "The Customer is eligible for a Credit Card"
    return {'prediction': prediction_text}

# Endpoint for image prediction
@app.post('/predict_image')
async def upload(file: UploadFile = File(...)):
    try:
        # Open and process the uploaded image
        im = Image.open(BytesIO(await file.read()))
        prediction = predict_image(im)
        return {"prediction": prediction}
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)