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

## 📊 Technical Details
- **Model**: EfficientNet (Transfer Learning)
- **Input**: Mel Spectrograms (128x128)
- **Genres**: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.
- **Backend**: Python, Flask, TensorFlow, Keras 3.
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+).

---

## 🤝 Support
If you encounter any issues, ensure that your Python environment has the matching versions of TensorFlow and Keras, as the model was trained using **Keras 3**.
