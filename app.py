import streamlit as st
import gdown
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model

# ตั้งค่า Streamlit Page Config (ควรอยู่บรรทัดแรก)
st.set_page_config(page_title="💖 Heart Sound Analysis", layout="wide")

# 🎯 ใส่ Google Drive File ID ของโมเดล
GDRIVE_FILE_ID = "13oUZjw0OTeOoxbk5-CZHsuDonY2oquPO"  # เปลี่ยนเป็นของคุณ

# 🎯 ตรวจสอบว่าโมเดลมีหรือยัง ถ้าไม่มีให้โหลด
model_path = "model_heartbeat.h5"
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)

# โหลดโมเดล
st.write("✅ Loading model...")
model = load_model(model_path)
st.write("✅ Model loaded successfully!")

# 🔹 ฟังก์ชัน Band-pass Filter (20Hz - 200Hz)
def bandpass_filter(y, sr, lowcut=20.0, highcut=200.0, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered

# 🔹 UI สำหรับอัปโหลดไฟล์เสียง
st.title("💖 Heart Sound Analysis")
uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์เสียงหัวใจ (.wav)", type=["wav"])

if uploaded_file is not None:
    # 🔹 อ่านไฟล์เสียง
    y, sr = librosa.load(uploaded_file, sr=4000)

    # 🔹 กรองเสียงด้วย Band-pass Filter
    y_filtered = bandpass_filter(y, sr)

    # 🔹 แสดงกราฟเสียง (ก่อนและหลังฟิลเตอร์)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ก่อนฟิลเตอร์
    axes[0].set_title("Raw Heart Sound")
    axes[0].set_ylabel("Amplitude")
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="gray")

    # หลังฟิลเตอร์
    axes[1].set_title("Filtered Heart Sound")
    axes[1].set_ylabel("Amplitude")
    librosa.display.waveshow(y_filtered, sr=sr, ax=axes[1], color="blue")

    st.pyplot(fig)  # ✅ แสดงกราฟ

    # 🔹 **บันทึกเสียงที่ผ่านฟิลเตอร์เป็นไฟล์ .wav**
    filtered_file_path = "filtered_heart_sound.wav"
    sf.write(filtered_file_path, y_filtered, sr)  # ✅ บันทึกไฟล์

    # 🔊 **ฟังเสียงก่อนและหลังฟิลเตอร์**
    st.markdown("🔊 **ฟังเสียงก่อนและหลังฟิลเตอร์:**")
    
    # 🔹 เล่นเสียงดั้งเดิม
    st.audio(uploaded_file, format="audio/wav")

    # 🔹 เล่นเสียงที่ผ่านฟิลเตอร์จากไฟล์
    st.audio(filtered_file_path, format="audio/wav")
