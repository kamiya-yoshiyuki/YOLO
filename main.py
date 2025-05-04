import io
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import cv2

app = FastAPI()
model = YOLO("yolov8n.pt")
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ➡️ PIL → NumPy (YOLO入力形式、サイズ変更なし)
        img_np = np.array(image)[:, :, ::-1]  # RGB→BGR

        # 🔍 YOLO推論
        results = model(img_np)
        result_img = results[0].plot()

        # 🔁 推論結果をエンコード（Base64）
        _, buffer = cv2.imencode('.jpg', result_img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(content={"image_base64": encoded_image})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

