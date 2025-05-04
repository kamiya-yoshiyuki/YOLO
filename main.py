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

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # 一時ファイルに保存
    suffix = os.path.splitext(file.filename)[1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(await file.read())
        temp_input_path = temp_input.name

    try:
        # 📌 大きな画像をリサイズ（例: 最大640x640に縮小）
        image = cv2.imread(temp_input_path)
        resized = cv2.resize(image, (640, 640))  # 必要に応じて変更
        cv2.imwrite(temp_input_path, resized)

        # 🔍 推論を実行（YOLOv8など）
        results = model(temp_input_path)
        result_img = results[0].plot()  # 検出結果を画像に描画

        # 🔄 結果画像を一時ファイルに保存
        result_path = temp_input_path.replace(suffix, f"_result.jpg")
        cv2.imwrite(result_path, result_img)

        # 🔁 Base64形式に変換して返却
        with open(result_path, "rb") as f:
            image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # 🧹 クリーンアップ
        os.remove(temp_input_path)
        os.remove(result_path)

        return JSONResponse(content={"image_base64": encoded_image})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
