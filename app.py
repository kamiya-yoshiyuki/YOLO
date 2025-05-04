import requests
import base64
import streamlit as st

st.title("物体検出アプリ")
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="アップロード画像", use_container_width=True)

    if st.button("推論開始"):
        with st.spinner("物体検出を実行中..."):
            response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file})
        if response.status_code == 200:
                img_base64 = response.json()["image_base64"]
                img_bytes = base64.b64decode(img_base64)
                st.image(img_bytes, caption="検出結果", use_container_width=True)
        else:
            st.error(f"エラーが発生しました: {response.text}")
