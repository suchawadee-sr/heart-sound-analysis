import streamlit as st
import gdown
import os
from tensorflow.keras.models import load_model

# 🔹 ใส่ Google Drive File ID ของคุณที่นี่
GDRIVE_FILE_ID = "1znyYiOQPDa274jtbYZGwLR91X1uqyqEJ"  # 🔄 เปลี่ยนเป็นของคุณ

# 🔹 ตรวจสอบว่ามีโมเดลหรือยัง ถ้าไม่มีให้ดาวน์โหลด
model_path = "heartbeat_model.h5"
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)

# โหลดโมเดล
st.write("✅ Loading model...")
model = load_model(model_path)
st.write("✅ Model loaded successfully!")

# Web App Interface
st.title("🔍 Heart Sound Classification Web App")
st.write("📂 อัปโหลดไฟล์เสียงหัวใจ (.wav) แล้วให้ AI วิเคราะห์ว่าเป็น **Healthy หรือ Unhealthy**")

uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์เสียง", type=["wav"])

if uploaded_file:
    file_path = f"temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("✅ File uploaded successfully!")
