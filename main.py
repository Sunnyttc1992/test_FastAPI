import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'mnist_rf_model.pkl'

model: Any | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event('startup')
def load_model() -> None:
    global model
    if model is not None:
        return

    if not MODEL_PATH.exists():
        raise RuntimeError('mnist_rf_model.pkl not found; run train_model.py first.')

    with MODEL_PATH.open('rb') as f:
        model = pickle.load(f)


@app.get('/')
def serve_index() -> FileResponse:
    index_path = BASE_DIR / 'index.html'
    if not index_path.exists():
        raise HTTPException(status_code=404, detail='index.html not found')
    return FileResponse(index_path)


@app.get('/health')
def healthcheck() -> dict[str, Any]:
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/predict')
async def predict_image(image: UploadFile = File(...)) -> dict[str, int | str]:
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert('L')
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid image file') from exc

    if model is None:
        raise HTTPException(status_code=503, detail='Model is not loaded yet')

    pil_image = ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), resample=Image.Resampling.LANCZOS)

    image_array = np.asarray(pil_image, dtype=np.float32).reshape(1, -1)
    prediction = model.predict(image_array)

    try:
        predicted_digit = int(prediction[0])
    except (TypeError, ValueError):
        predicted_digit = prediction[0]

    return {'prediction': predicted_digit}
