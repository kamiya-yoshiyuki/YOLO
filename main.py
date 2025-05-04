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

def letterbox_image(image, size=(640, 640), color=(114, 114, 114)):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = image.resize((nw, nh), Image.BILINEAR)

    new_image = Image.new('RGB', size, color)
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_image, (iw, ih)  # 元サイズも返す

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 元画像をメモリ上で読み込み
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # アスペクト比を維持したletterbox画像と元サイズを取得
        resized_image, orig_size = letterbox_image(image)

        # PIL → NumPy (BGR形式)
        img_np = np.array(resized_image)[:, :, ::-1]

        # YOLO推論
        results = model(img_np)
        result_img = results[0].plot()

        # ✅ 結果画像を元のサイズにリサイズして返す
        result_img = cv2.resize(result_img, orig_size)

        # エンコードして返却
        _, buffer = cv2.imencode('.jpg', result_img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(content={"image_base64": encoded_image})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
