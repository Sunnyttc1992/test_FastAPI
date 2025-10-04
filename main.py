import io
import pickle

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps


with open('mnist_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/predict')
async def predict_image(image: UploadFile = File(...)):
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert('L')
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid image file') from exc

    pil_image = ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), resample=Image.Resampling.LANCZOS)

    image_array = np.array(pil_image, dtype=np.float32).reshape(1, -1)
    prediction = model.predict(image_array)
    return {'prediction': prediction[0]}
