import io
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / 'mnist_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def serve_index():
    index_path = BASE_DIR / 'index.html'
    if not index_path.exists():
        raise HTTPException(status_code=404, detail='index.html not found')
    return FileResponse(index_path)

@app.post('/predict')
async def predict_image(image: UploadFile = File(...)):
    try: