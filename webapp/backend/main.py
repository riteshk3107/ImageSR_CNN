from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from model import load_model

app = FastAPI(title="Image Super-Resolution API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
# Path relative to where we run the server (webapp/backend)
WEIGHTS_PATH = "../../trainning/super_resolution_cnn.keras"
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        # Resolve absolute path to avoid confusion
        abs_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), WEIGHTS_PATH))
        model = load_model(abs_weights_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # We might want to exit here if model fails, but for now just log it

# @app.get("/")
# def read_root():
#     return {"message": "Image Super-Resolution API is running"}

# Mount Frontend (must be after API routes)
# We need to point to the frontend directory relative to this file
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

def preprocess_image(image_bytes: bytes):
    """
    Convert bytes to numpy array, normalize, and add batch dimension.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_np = image_np / 255.0  # Normalize to [0, 1]
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension (1, H, W, 3)
    return image_np, image.size

def postprocess_image(output_np):
    """
    Convert model output to bytes.
    """
    # output_np is (1, H, W, 3)
    output_np = np.squeeze(output_np, axis=0)
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(output_np)
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        input_tensor, original_size = preprocess_image(contents)
        
        # Predict
        prediction = model.predict(input_tensor)
        
        # Postprocess
        result_bytes = postprocess_image(prediction)
        
        return Response(content=result_bytes, media_type="image/png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)