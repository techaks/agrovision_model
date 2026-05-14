from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image
import io
import json

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Rebuild SAME model architecture
# -----------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax")
])

# -----------------------------------
# Load only weights from .h5
# -----------------------------------

model.load_weights("plant_disease_model.h5")

print("Weights Loaded Successfully 🔥")

# -----------------------------------
# Class names
# -----------------------------------

with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("Classes Loaded:", class_names)

# -----------------------------------
# Home route
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "AgroVision AI Backend Running Successfully 🚀"
    }

# -----------------------------------
# Predict route
# -----------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        return {
            "disease": predicted_class,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }