import streamlit as st
import gdown
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# 🔹 ใส่ Google Drive File ID ของโมเดลที่อัปโหลด
GDRIVE_FILE_ID = "13oUZjw0OTeOoxbk5-CZHsuDonY2oquPO"  # 🔄 เปลี่ยนเป็นของคุณ

# 🔹 ตรวจสอบว่ามีโมเดลหรือยัง ถ้าไม่มีให้ดาวน์โหลด
model_path = "model_heartbeat.h5"
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)

# โหลดโมเดล
st.write("✅ Loading model...")
model = load_model(model_path)
st.write("✅ Model loaded successfully!")

# 🔹 ฟังก์ชันประมวลผลเสียง
def preprocess_audio(file_path, sr=4000, n_mels=128, max_frames=128):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] < max_frames:
            pad_width = max_frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_frames]

        return mel_spec_db.reshape(1, 128, 128, 1)

    except Exception as e:
        return None

# Web App Interface
st.title("🔍 Heart Sound Classification Web App")
st.write("📂 อัปโหลดไฟล์เสียงหัวใจ (.wav) แล้วให้ AI วิเคราะห์ว่าเป็น **Healthy หรือ Unhealthy**")

uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์เสียง", type=["wav"])

if uploaded_file:
    file_path = f"temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    preprocessed_audio = preprocess_audio(file_path)

    if preprocessed_audio is not None:
        prediction = model.predict(preprocessed_audio)
        predicted_class = np.argmax(prediction)
        classes = ["💙 Healthy", "💔 Unhealthy"]
        result = classes[predicted_class]

        st.write(f"### 🔍 ผลการวิเคราะห์: {result}")
    else:
        st.error("⚠️ Audio preprocessing failed.")
