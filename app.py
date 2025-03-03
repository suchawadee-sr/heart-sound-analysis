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

# ตั้งค่าให้ Streamlit รองรับ layout กว้าง
st.set_page_config(page_title="Heartbeat Health", layout="wide")

# 🎨 **CSS ตกแต่งให้เป็นสีแบบที่ต้องการ**
st.markdown("""
    <style>
        body {
            background-color: #FFDFDF;
        }
        .stApp {
            background-color: #FFD1D1;
        }
        .block-container {
            padding: 2rem;
            background-color: #FFC0CB;
            border-radius: 15px;
        }
        .stButton>button {
            background-color: #7B3F3F !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 10px 20px;
        }
        .stTextInput>div>div>input {
            background-color: #7B3F3F !important;
            color: white !important;
            border-radius: 10px !important;
        }
        .stFileUploader {
            background-color: #FFF5F5 !important;
            border-radius: 10px;
        }
        .stMarkdown {
            font-size: 18px;
            font-weight: bold;
            color: #7B3F3F;
        }
        .header {
            background: linear-gradient(to bottom, #FF6B6B, #FFA07A);
            padding: 15px 20px;
            text-align: left;
            border-radius: 10px;
        }
        .header h1 {
            color: white;
            font-size: 28px;
            display: inline;
        }
        .heart-icons {
            float: right;
            font-size: 28px;
        }
        .rounded-box {
            background-color: white;
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# 🎨 **Header**
st.markdown("""
    <div class="header">
        <h1>❤️ heartbeat health</h1>
        <span class="heart-icons">🖤 🤍 ❤️</span>
    </div>
""", unsafe_allow_html=True)

# 🎯 **โหลดโมเดล**
GDRIVE_FILE_ID = "13oUZjw0OTeOoxbk5-CZHsuDonY2oquPO"
model_path = "model_heartbeat.h5"
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)

st.write("✅ Loading model...")
model = load_model(model_path)
st.write("✅ Model loaded successfully!")

# 🎯 **ฟังก์ชัน Band-pass Filter**
def bandpass_filter(y, sr, lowcut=20.0, highcut=200.0, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y)

# 🎯 **ฟังก์ชันแสดง Spectrogram**
def plot_spectrogram(y, sr, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(6, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    return fig

# 🎯 **ฟังก์ชันประมวลผลเสียง**
def preprocess_audio(file_path, sr=4000, n_mels=128, max_frames=128):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y_filtered = bandpass_filter(y, sr)
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

# 🎯 **อัปโหลดไฟล์เสียง**
st.markdown('<div class="rounded-box">📂 **อัปโหลดไฟล์เสียงหัวใจ (.wav)**</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop file here", type=["wav"])

if uploaded_file is not None:
    file_path = "uploaded_heart_sound.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    preprocessed_audio, y_raw, y_filtered, sr = preprocess_audio(file_path)

    if preprocessed_audio is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="rounded-box">🎵 **Waveform ก่อนกรองเสียง**</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            librosa.display.waveshow(y_raw, sr=sr, color="gray")
            plt.title("Raw Heart Sound")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            st.pyplot(fig)

        with col2:
            st.markdown('<div class="rounded-box">🎶 **Waveform หลังกรองเสียง**</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            librosa.display.waveshow(y_filtered, sr=sr, color="blue")
            plt.title("Filtered Heart Sound")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            st.pyplot(fig)

        st.markdown('<div class="rounded-box">🎨 **Spectrogram เปรียบเทียบก่อนและหลังการกรองเสียง:**</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("🎼 **ก่อนกรองเสียง**")
            st.pyplot(plot_spectrogram(y_raw, sr, title="Raw Spectrogram"))

        with col4:
            st.markdown("🎼 **หลังกรองเสียง**")
            st.pyplot(plot_spectrogram(y_filtered, sr, title="Filtered Spectrogram"))

        prediction = model.predict(preprocessed_audio)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        classes = ["❤️ Healthy", "💔 Unhealthy"]
        st.markdown(f'<div class="rounded-box">🔎 **ผลการวิเคราะห์:** {classes[predicted_class]} (ความมั่นใจ: {confidence:.2f})</div>', unsafe_allow_html=True)
