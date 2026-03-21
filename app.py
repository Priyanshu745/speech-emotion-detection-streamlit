import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.models import load_model
import h5py

# --- Constants (must match training) ---
N_MFCC = 40
MAX_LEN = 174
TARGET_SR = 16000

# --- Safe model loading (Option A fix) ---
@st.cache_resource
def load_model_safely():
    try:
        model = tf.keras.models.load_model(
            "saved_model/emotion_model.keras",
            compile=False,
            safe_mode=False
        )
        le = joblib.load("saved_model/label_encoder.pkl")
        return model, le, None
    except Exception as e:
        return None, None, str(e)

model, le, load_error = load_model_safely()

if load_error:
    st.error("❌ Model failed to load")
    st.code(load_error)
    st.stop()



# --- Feature Extraction ---
def extract_features(audio_bytes):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=TARGET_SR)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Pad or truncate
    if mfccs.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :MAX_LEN]

    # Shape: (1, 174, 40, 1)
    features = mfccs.T[np.newaxis, ..., np.newaxis]

    return features, y, sr

# --- Plot waveform ---
def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

# --- Plot spectrogram ---
def plot_spectrogram(y, sr):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)

# --- UI ---
st.set_page_config(page_title="Speech Emotion Recognizer", page_icon="🎙️")
st.title("🎙️ Speech Emotion Detection")
st.write("Upload a `.wav` file to detect the emotion.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    try:
        # --- Read once (important fix) ---
        audio_bytes = uploaded_file.read()

        # --- Feature extraction ---
        features, y, sr = extract_features(audio_bytes)

        # Debug shape (optional)
        # st.write("Feature shape:", features.shape)

        # --- Visualization ---
        st.subheader("📊 Audio Visualization")
        plot_waveform(y, sr)
        plot_spectrogram(y, sr)

        # --- Prediction ---
        prediction = model.predict(features, verbose=0)

        predicted_index = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_index])[0]
        confidence = prediction[0][predicted_index]

        # --- Results ---
        st.subheader("🎯 Prediction Results")
        st.success(f"🧠 Emotion: {predicted_label.upper()} ({confidence:.2%})")

        st.markdown("#### 🔎 Class Probabilities:")
        for label, prob in zip(le.classes_, prediction[0]):
            st.write(f"- {label}: {prob:.4f}")

    except Exception as e:
        st.error(f"⚠️ Error processing audio:\n\n{e}")
