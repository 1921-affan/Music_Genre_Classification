
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import numpy as np
import sys
from tensorflow.image import resize
from flask_cors import CORS
from flask_socketio import SocketIO, emit


app = Flask(__name__)
# Enable CORS for all routes and origins for development
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'running', 'message': 'Backend is active'})

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "Trained_model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = tf.keras.models.load_model(model_path)

def emit_progress(step, message):
    socketio.emit('progress', {'step': step, 'message': message})

# Preprocessing function
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    print(f"DEBUG: load_and_preprocess_data called for {file_path}")
    emit_progress('upload', 'Starting preprocessing...')
    data = []
    try:
        print("DEBUG: librosa.load starting...")
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        print(f"DEBUG: librosa.load success. SR={sample_rate}, Length={len(audio_data)}")
    except Exception as e:
        print(f"DEBUG: librosa.load FAILED: {e}")
        raise e
    emit_progress('preprocessing', 'Audio loaded successfully')
    
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    emit_progress('feature_extraction', 'Starting feature extraction...')
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

        emit_progress('feature_extraction', f'Processed chunk {i+1}/{num_chunks} (duration: {chunk_duration}s, overlap: {overlap_duration}s)')

    emit_progress('feature_extraction', 'Feature extraction completed')
    return np.array(data)

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(f"DEBUG: Receiving file...", file=sys.stderr)
        file = request.files.get('file')
        if not file:
            print("DEBUG: No file provided")
            return jsonify({'status': 'error', 'message': 'No file provided.'})
        
        file_path = f"temp_{file.filename}"
        print(f"DEBUG: Saving to {file_path}")
        file.save(file_path)

        emit_progress('upload', 'File uploaded successfully')

        print("DEBUG: Calling load_and_preprocess_data")
        X_test = load_and_preprocess_data(file_path)
        print("DEBUG: Preprocessing done")
        
        emit_progress('inference', 'Starting inference...')
        print("DEBUG: Starting prediction")
        y_pred = model.predict(X_test)
        emit_progress('inference', 'Inference completed')

        predicted_categories = np.argmax(y_pred, axis=1)
        unique_elements, counts = np.unique(predicted_categories, return_counts=True)
        max_count = np.max(counts)
        max_elements = unique_elements[counts == max_count]
        label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        result = label[max_elements[0]]

        confidence = float(counts[counts == max_count][0]) / len(predicted_categories) * 100

        os.remove(file_path)

        emit_progress('complete', 'Classification completed')

        return jsonify({
            'status': 'success',
            'predicted_class': result,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        emit_progress('error', str(e))
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True)
