import streamlit as st
import gdown
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model

# ตั้งค่าให้ Streamlit รองรับ layout กว้าง
st.set_page_config(page_title="Heartbeat Health", layout="wide")

st.markdown("""
    <style>
        .header {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            background: #CF008A;
            background: radial-gradient(circle farthest-corner at top center, #CF008A 4%, #CF799B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 ใช้ Gradient Text เป็น Title ของเว็บ
st.markdown('<div class="header">Heartbeat Health ❤️</div>', unsafe_allow_html=True)

# 🎯 ใส่ Google Drive File ID ของโมเดล
GDRIVE_FILE_ID = "13oUZjw0OTeOoxbk5-CZHsuDonY2oquPO"

# 🎯 ตรวจสอบว่าโมเดลมีหรือยัง ถ้าไม่มีให้โหลด
model_path = "model_heartbeat.h5"
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)

st.write("✅ Loading model...")
model = load_model(model_path)
st.write("✅ Model loaded successfully!")

# 🎯 ฟังก์ชัน Band-pass Filter (20Hz - 200Hz)
def bandpass_filter(y, sr, lowcut=20.0, highcut=200.0, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y)

# 🎯 ฟังก์ชันประมวลผลเสียง
def preprocess_audio(file_path, sr=4000, n_mels=128, max_frames=128):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y_filtered = bandpass_filter(y, sr)  # ใช้ Band-pass Filter
        mel_spec = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] < max_frames:
            pad_width = max_frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mel_spec_db = mel_spec_db[:, :max_frames]

        return mel_spec_db.reshape(1, 128, 128, 1), y, y_filtered, sr
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
        return None, None, None, None

# 🎯 อัปโหลดไฟล์เสียง
st.markdown('<div class="rounded-box">📂 **อัปโหลดไฟล์เสียงหัวใจ (.wav)**</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop file here", type=["wav"])

if uploaded_file is not None:
    file_path = "uploaded_heart_sound.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    preprocessed_audio, y_raw, y_filtered, sr = preprocess_audio(file_path)

    if preprocessed_audio is not None:
        col1, col2 = st.columns(2)

        # 🎯 แสดง Waveform ก่อนกรองเสียง
        with col1:
            st.markdown('<div class="rounded-box">🎵 **Waveform ก่อนกรองเสียง**</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            librosa.display.waveshow(y_raw, sr=sr, color="gray")
            plt.title("Raw Heart Sound")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.ylim(-1, 1)  # ให้กราฟมีขนาดเท่ากัน
            st.pyplot(fig)

        # 🎯 แสดง Waveform หลังกรองเสียง
        with col2:
            st.markdown('<div class="rounded-box">🎶 **Waveform หลังกรองเสียง**</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            librosa.display.waveshow(y_filtered, sr=sr, color="blue")
            plt.title("Filtered Heart Sound")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.ylim(-1, 1)  # ให้กราฟมีขนาดเท่ากัน
            st.pyplot(fig)

        # 🎯 ทำนายผล
        prediction = model.predict(preprocessed_audio)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        classes = ["❤️ Healthy", "💔 Unhealthy"]
        st.markdown(f'<div class="rounded-box">🔎 **ผลการวิเคราะห์:** {classes[predicted_class]} (ความมั่นใจ: {confidence:.2f})</div>', unsafe_allow_html=True)
