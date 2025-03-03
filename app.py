import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# โหลดโมเดล
model = load_model("heartbeat_model.h5")

# ฟังก์ชันประมวลผลเสียง
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
        st.error(f"Error processing {file_path}: {e}")
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

        fig, ax = plt.subplots(figsize=(6, 4))
        librosa.display.specshow(preprocessed_audio[0, :, :, 0], sr=4000, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title("🎭 Mel-Spectrogram")
        st.pyplot(fig)

    else:
        st.error("⚠️ Audio preprocessing failed.")
