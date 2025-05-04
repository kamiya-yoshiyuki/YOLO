import os
import uuid
import base64
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO  # YOLOv8など
from tempfile import NamedTemporaryFile
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()

# YOLOモデルを読み込み（キャッシュされる）
model = YOLO("yolov8n.pt")  # または fine-tuned モデルパス

from ultralytics.utils.plotting import Annotator  # YOLOv8が内部で使っている描画ユーティリティ

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(await file.read())
        temp_input_path = temp_input.name

    try:
        # オリジナル画像を読み込む
        image = cv2.imread(temp_input_path)
        original_h, original_w = image.shape[:2]

        # 推論（YOLOv8は内部でリサイズ処理を行うため、ここではそのまま渡してOK）
        results = model(temp_input_path)
        result = results[0]

        # 元画像に描画
        annotator = Annotator(image)
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"
            color = colors(cls)  # ← YOLOv8標準のクラスごとの色
            annotator.box_label(xyxy, label, color=color)

        result_img = annotator.result()  # numpy配列で取得

        # Base64変換して返却
        result_path = temp_input_path.replace(suffix, "_result.jpg")
        cv2.imwrite(result_path, result_img)
        with open(result_path, "rb") as f:
            image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        os.remove(temp_input_path)
        os.remove(result_path)

        return JSONResponse(content={"image_base64": encoded_image})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


