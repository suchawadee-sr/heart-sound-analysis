import streamlit as st
import gdown
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import scipy.signal as signal
import soundfile as sf
import tensorflow as tf

# 🔹 ใส่ Google Drive File ID ของโมเดล
GDRIVE_FILE_ID = "13oUZjw0OTeOoxbk5-CZHsuDonY2oquPO"  # เปลี่ยนเป็นของคุณ

# 🔹 ตรวจสอบว่ามีโมเดลหรือยัง ถ้าไม่มีให้ดาวน์โหลด
model_path = "model_heartbeat.h5"
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)

# 🔹 โหลดโมเดล AI
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()



# 🎨 **UI ของ Web App**
st.set_page_config(page_title="💖 Heart Sound Analysis", layout="wide")
st.markdown("<h1 style='text-align: center; color: #ff4d6d;'>💖 Heartbeat Health</h1>", unsafe_allow_html=True)

# 📂 อัปโหลดไฟล์เสียง
uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์เสียงหัวใจ (.wav)", type=["wav"])
# 🔹 ฟังก์ชัน Band-pass Filter (20Hz - 200Hz)
def bandpass_filter(y, sr, lowcut=20.0, highcut=200.0, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered

# 🔹 ฟังก์ชันเตรียมข้อมูลเสียงก่อนพยากรณ์
def preprocess_audio_for_prediction(file_path, sr=4000, n_mels=128, max_frames=128):
    try:
        y, sr = librosa.load(file_path, sr=sr)

        # 🔹 กรองเสียงก่อนพยากรณ์
        y_filtered = bandpass_filter(y, sr)

        # 🔹 สร้าง Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 🔹 ปรับขนาดให้เหมาะสม
        if mel_spec_db.shape[1] < max_frames:
            pad_width = max_frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_frames]

        return mel_spec_db.reshape(1, 128, 128, 1), y, y_filtered, sr

    except Exception as e:
        return None, None, None, None
        
if uploaded_file is not None:
    with st.spinner("🔍 Processing audio..."):
        file_path = f"temp.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 📌 **กรองเสียง**
        preprocessed_audio, y_raw, y_filtered, sr = preprocess_audio_for_prediction(file_path)

        if preprocessed_audio is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("🎵 **Waveform ก่อนกรองเสียง:**")
                fig, ax = plt.subplots(figsize=(5, 2))
                librosa.display.waveshow(y_raw, sr=sr, color="gray")
                plt.title("Raw Heart Sound")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                st.pyplot(fig)

            with col2:
                st.markdown("🎵 **Waveform หลังกรองเสียง:**")
                fig, ax = plt.subplots(figsize=(5, 2))
                librosa.display.waveshow(y_filtered, sr=sr, color="pink")
                plt.title("Filtered Heart Sound")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                st.pyplot(fig)

            # 📌 **ใช้โมเดลพยากรณ์**
            prediction = model.predict(preprocessed_audio)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            classes = ["💙 Healthy", "💔 Unhealthy"]
            result = classes[predicted_class]

            st.markdown(f"<h2 style='text-align: center;'>📢 ผลการวิเคราะห์: {result} (ความมั่นใจ: {confidence:.2f})</h2>", unsafe_allow_html=True)
