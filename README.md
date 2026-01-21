# Music_Genre_Classification
# 🎵 Music Genre Classifier (AI-Powered)

This project is a high-precision music genre classification application. It uses a deep learning model (EfficientNet) to analyze audio spectrograms and classify songs into 10 different genres.


## 📌 Project Overview
The application features a modern "Glassmorphism" UI with real-time progress tracking. When a user uploads an audio file, the system provides live updates as it performs preprocessing, feature extraction, and neural network inference.

### 🔹 Features
✅ **AI Prediction** - High-accuracy genre classification using EfficientNet.
✅ **Real-Time Progress** - Live updates via Socket.IO (Uploading ➔ Preprocessing ➔ Analysis ➔ Inference).
✅ **Modern UI** - A premium dark-themed interface with smooth animations and responsive design.
✅ **Single Server** - The Flask backend serves the frontend directly, eliminating CORS and network configuration issues.

### 🔹 Technologies
- **Frontend**: HTML5, CSS3, JavaScript (ES6+).
- **Backend**: Python, Flask, Flask-SocketIO, Flask-CORS.
- **Machine Learning**: Keras, TensorFlow 2.16.1.
- **Audio Processing**: Librosa, OpenCV.

---

## 🚀 How to Run the Application

### 1️⃣ Prerequisites
Ensure you have **Python 3.10+** installed on your system.
The following libraries are required (installed globally in the current setup):
- `flask`, `flask-socketio`, `flask-cors`
- `tensorflow==2.16.1`, `keras==3.12.0`
- `librosa`, `opencv-python-headless`, `numpy<2.0.0`, `scipy<1.14`

### 2️⃣ Start the Server
Open your terminal/command prompt and run:

```powershell
cd "backend"
python app.py
```

### 3️⃣ Access the App
Once the server starts (it will say `Running on http://127.0.0.1:5000`), open your web browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🏗 Project Structure
```text
ANN0509/
├── backend/
│   ├── app.py                  # Main Flask server & API logic
│   └── EfficientNet_Model.keras # The trained Neural Network model
├── frontend/
│   ├── index.html              # Modern UI structure
│   ├── style.css               # Premium styling & animations
│   └── script.js               # Frontend logic & Socket.IO connection
└── README.md                   # Project documentation
```

---

## � Model Training Process

The model was trained using the **GTZAN Music Genre Dataset**, following a rigorous preprocessing and transfer learning pipeline.

### 1️⃣ Preprocessing & Feature Extraction
- **Mel-Spectrogram Generation**: Audio signals were converted into Mel-Spectrograms (images of sound) using `librosa`.
- **Audio Slicing**: To maximize data, each 30-second track was sliced into **10 segments of 3 seconds each**, effectively increasing the dataset size tenfold.
- **Normalization**: Log-scaled Mel-Spectrograms were resized to $224 \times 224 \times 3$ and normalized to a 0-255 pixel range.

### 2️⃣ Architecture: EfficientNet-B0
- **Transfer Learning**: We utilized **EfficientNet-B0** pre-trained on ImageNet as the feature extractor.
- **Custom Classification Head**:
    - `GlobalAveragePooling2D` to reduce spatial dimensionality.
    - `Dense` layer with 512 units and `ReLU` activation.
    - `Dropout(0.5)` for regularization and to prevent overfitting.
    - `Softmax` output layer for 10-class probability distribution.
- **Optimization**: The model was compiled with the `Adam` optimizer (learning rate: 1e-4) and trained with `Sparse Categorical Crossentropy` loss.

### 3️⃣ Training Performance
- **Epochs**: 15 (with Learning Rate reduction on plateau).
- **Batch Size**: 32.
- **Stability**: Batch Normalization layers were frozen during training to maintain stable statistics across audio segments.

---

## 📊 Technical Details
- **Model**: EfficientNet-B0 (Transfer Learning)
- **Input Dimensions**: $224 \times 224 \times 3$
- **Inference Strategy**: Voting mechanism across multiple 3-second chunks of a song.
- **Genres**: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.
- **Backend Stack**: Python, Flask, TensorFlow, Keras 3.
- **Frontend Stack**: Vanilla HTML5, CSS3, JavaScript (ES6+).

---

## 🤝 Support
If you encounter any issues, ensure that your Python environment has the matching versions of TensorFlow and Keras, as the model was trained using **Keras 3**.
