import os
import uuid
import base64
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO  # YOLOv8など
from tempfile import NamedTemporaryFile

app = FastAPI()

# YOLOモデルを読み込み（キャッシュされる）
model = YOLO("yolov8n.pt")  # または fine-tuned モデルパス
import imghdr

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(await file.read())
        temp_input_path = temp_input.name

    try:
        # オリジナル画像を読み込み
        image = cv2.imread(temp_input_path)
        original_h, original_w = image.shape[:2]

        # YOLOの推論に適したサイズ（縦横比維持で最大サイズに収める）
        target_size = 640
        scale = min(target_size / original_w, target_size / original_h)
        resized_w, resized_h = int(original_w * scale), int(original_h * scale)
        resized = cv2.resize(image, (resized_w, resized_h))

        # 黒背景で padding（640x640 にしてモデルに渡す）
        padded = cv2.copyMakeBorder(
            resized,
            top=(target_size - resized_h) // 2,
            bottom=(target_size - resized_h + 1) // 2,
            left=(target_size - resized_w) // 2,
            right=(target_size - resized_w + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        cv2.imwrite(temp_input_path, padded)

        # 推論
        results = model(temp_input_path)
        result_img = results[0].plot()

        # 推論結果画像をオリジナルサイズに戻す
        result_img_resized = cv2.resize(result_img, (original_w, original_h))

        # 保存してBase64に変換
        result_path = temp_input_path.replace(suffix, "_result.jpg")
        cv2.imwrite(result_path, result_img_resized)

        with open(result_path, "rb") as f:
            image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        os.remove(temp_input_path)
        os.remove(result_path)

        return JSONResponse(content={"image_base64": encoded_image})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

