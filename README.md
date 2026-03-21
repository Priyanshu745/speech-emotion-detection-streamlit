# 🎙️ Speech Emotion Recognition Web App

This is a simple and interactive **Speech Emotion Detection** web application built with **Streamlit** and **TensorFlow**. Upload a `.wav` audio file and the app will predict the emotion conveyed in the speech using a pre-trained deep learning model.

---

## 🚀 Features

* 📁 Upload `.wav` audio files
* 📈 Visualize waveform and mel-spectrogram
* 🤖 Predict the speaker's emotion using a pre-trained CNN model
* 📊 Show class probabilities with human-readable emotion labels

---

## 🛆 Requirements

Before running the app, install the required dependencies:

```bash
pip install -r requirements.txt
```

**Main dependencies:**

* `streamlit`
* `tensorflow`
* `librosa`
* `matplotlib`
* `joblib`
* `numpy`

---

## 🧐 Model Info

* The model is a **Convolutional Neural Network (CNN)** trained on MFCC features extracted from audio clips.
* The labels are encoded using `LabelEncoder` and stored in `saved_model/label_encoder.pkl`.

---

## 📁 Project Structure

```
speech-emotion-app/
│
├── app.py                     # Streamlit app code
├── saved_model/
│   ├── emotion_model.h5       # Trained Keras model
│   └── label_encoder.pkl      # Fitted label encoder
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/speech-emotion-app.git
cd speech-emotion-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open your browser and go to:
   `http://localhost:8501`

---



## 🎯 Example Emotions

Supported emotion categories may include:

* Happy 😊
* Sad 😢
* Angry 😠
* Neutral 😐
  *(Custom depending on model training)*

---

## 📌 Notes

* Audio must be in `.wav` format.
* Sampling rate is standardized to 16 kHz during preprocessing.
* The model expects MFCC features with shape `(174, 40)`.

---

## 🧪 Future Improvements

* Live microphone recording
* Multilingual emotion support
* Real-time feedback or emotion timeline

---

## 🛡️ License

MIT License — feel free to use, modify, and distribute.

---

## 🤝 Credits

* [Streamlit](https://streamlit.io/)
* [Librosa](https://librosa.org/)
* [TensorFlow](https://www.tensorflow.org/)
