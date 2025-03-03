import streamlit as st
import gdown
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model

# 🔹 ใส่ Google Drive File ID ของโมเดล
GDRIVE_FILE_ID = "1ABCD1234EFGH5678"  # 🔄 เปลี่ยนเป็นของคุณ

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

        return mel_spec_db.reshape(1, 128, 128, 1), y, sr

    except Exception as e:
        return None, None, None

# 🔹 UI ของ Web App
st.markdown(
    """
    <style>
    .stApp { background-color: #ffccd5; }
    .title { text-align: center; font-size: 30px; font-weight: bold; color: white; background: #ff6b81; padding: 20px; border-radius: 10px; }
    .button { background: #8b0000; color: white; font-size: 18px; padding: 10px 20px; border-radius: 5px; }
    .frame { background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">❤️ heartbeat health</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์เสียงหัวใจ (.wav)", type=["wav"])

if uploaded_file:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    preprocessed_audio, y, sr = preprocess_audio(file_path)

    if preprocessed_audio is not None:
        prediction = model.predict(preprocessed_audio)
        predicted_class = np.argmax(prediction)
        classes = ["💙 Healthy", "💔 Unhealthy"]
        result = classes[predicted_class]

        st.markdown(f'<div class="frame"><b>🔍 จังหวะที่วัดได้:</b> {sr} Hz</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="frame"><b>📊 แสดงกราฟเสียง:</b></div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform of Heart Sound")
        st.pyplot(fig)

        st.markdown(f'<div class="frame"><b>📢 ผลการวิเคราะห์:</b> {result}</div>', unsafe_allow_html=True)

    else:
        st.error("⚠️ Audio preprocessing failed.")
