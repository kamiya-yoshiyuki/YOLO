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

# ✅ アスペクト比維持のletterbox（メモリ処理）
def letterbox_image(image, size=(640, 640), color=(114, 114, 114)):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = image.resize((nw, nh), Image.BILINEAR)

    new_image = Image.new('RGB', size, color)
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_image

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 📥 画像読み込み（メモリ上）
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 🔄 アスペクト比維持でリサイズ
        resized_image = letterbox_image(image)

        # ➡️ PIL → NumPy (YOLO入力形式)
        img_np = np.array(resized_image)[:, :, ::-1]  # RGB→BGR

        # 🔍 YOLO推論
        results = model(img_np)
        result_img = results[0].plot()

        # 🔁 推論結果をエンコード（Base64）
        _, buffer = cv2.imencode('.jpg', result_img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(content={"image_base64": encoded_image})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
