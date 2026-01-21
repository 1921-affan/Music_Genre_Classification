import sys
import os
# Force python to see the user site packages where pip installed our libs
user_site = r"C:\Users\Affan yasir\AppData\Roaming\Python\Python310\site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site)
print("DEBUG: sys.path:", sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"


from flask import Flask, request, jsonify
import keras
print(f"DEBUG: Keras Version: {keras.__version__}")

# Import TensorFlow for Grad-CAM
import tensorflow as tf

# Check if we unwittingly use tf somewhere else
import librosa
import numpy as np
import sys
import cv2
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from gradcam import make_gradcam_heatmap, save_and_display_gradcam

app = Flask(__name__, static_folder='../frontend', static_url_path='')
# Enable CORS for all routes and origins for development
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration matching train_efficientnet.py
MODEL_FILENAME = "EfficientNet_Model.keras"
SAMPLE_RATE = 22050
CHUNK_DURATION = 3 # seconds
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_DURATION
INPUT_SHAPE = (224, 224)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    status = "Model Loaded" if model else "Model NOT Loaded"
    return jsonify({'status': 'running', 'message': 'Backend is active', 'model_status': status})

@app.route('/debug', methods=['GET'])
def debug_model():
    try:
        if model:
            return jsonify({'status': 'success', 'message': 'Model is already loaded.'})
        
        print("Debugging: Attempting to reload model...")
        debug_model = keras.models.load_model(model_path, compile=False)
        return jsonify({'status': 'success', 'message': 'Model reloaded successfully during debug.'})
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'traceback': traceback.format_exc()
        })

# Load the model
model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    model = None
else:
    try:
        print(f"Loading model from {model_path}...")
        model = keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

def emit_progress(step, message):
    socketio.emit('progress', {'step': step, 'message': message})

# Preprocessing function matching training logic
def load_and_preprocess_data(file_path):
    print(f"DEBUG: load_and_preprocess_data called for {file_path}")
    emit_progress('upload', 'Starting preprocessing...')
    data = []
    
    try:
        # Load audio (ensure 22050 Hz to match training)
        audio_data, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        print(f"DEBUG: Audio loaded. Length={len(audio_data)/sample_rate:.2f}s")
    except Exception as e:
        print(f"DEBUG: librosa.load FAILED: {e}")
        raise e
        
    emit_progress('preprocessing', 'Audio loaded successfully')
    
    # Slice into 3-second chunks (Non-overlapping to verify against training logic)
    # Note: For inference, we can use overlapping, but let's stick to training distribution first
    total_samples = len(audio_data)
    num_chunks = int(np.ceil(total_samples / SAMPLES_PER_CHUNK))
    
    emit_progress('feature_extraction', f'Extracting features from {num_chunks} chunks...')
    
    for i in range(num_chunks):
        start = i * SAMPLES_PER_CHUNK
        end = start + SAMPLES_PER_CHUNK
        
        chunk = audio_data[start:end]
        
        # Pad if shorter than 3s
        if len(chunk) < SAMPLES_PER_CHUNK:
            chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)), mode='constant')
            
        # -- Feature Extraction (Exact match to train_efficientnet.py) --
        
        # 1. Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # 2. Resize to 224x224 using OpenCV
        resized_spectrogram = cv2.resize(log_mel_spectrogram, INPUT_SHAPE)
        
        # 3. Stack to 3 channels
        rgb_spectrogram = np.stack([resized_spectrogram] * 3, axis=-1)
        
        # 4. Normalize to [0, 255]
        min_val = np.min(rgb_spectrogram)
        max_val = np.max(rgb_spectrogram)
        if max_val - min_val > 0:
            norm_spectrogram = (rgb_spectrogram - min_val) / (max_val - min_val)
            norm_spectrogram = norm_spectrogram * 255.0
        else:
            norm_spectrogram = np.zeros_like(rgb_spectrogram)
            
        data.append(norm_spectrogram)
        
        if (i+1) % 5 == 0:
            emit_progress('feature_extraction', f'Processed {i+1}/{num_chunks} chunks')

    emit_progress('feature_extraction', 'Feature extraction completed')
    return np.array(data)

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print("="*80)
    print("PREDICT ENDPOINT CALLED!")
    print("="*80)
    sys.stdout.flush()
    
    if model is None:
         return jsonify({'status': 'error', 'message': 'Model not loaded.'})
         
    try:
        print(f"DEBUG: Receiving file...", file=sys.stderr)
        sys.stdout.flush()
        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'No file provided.'})
        
        # Save temp file
        file_path = f"temp_{file.filename}"
        file.save(file_path)

        emit_progress('upload', 'File uploaded successfully')

        # Preprocess
        X_test = load_and_preprocess_data(file_path)
        
        emit_progress('inference', 'Starting inference...')
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Voting mechanism
        predicted_categories = np.argmax(y_pred, axis=1)
        unique_elements, counts = np.unique(predicted_categories, return_counts=True)
        max_count = np.max(counts)
        max_elements = unique_elements[counts == max_count]
        
        # Labels matching the training script
        # {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
        # 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        label_map = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        # Pick the winner
        winner_index = max_elements[0]
        result = label_map[winner_index]
        
        # Calculate confidence (percentage of chunks that agreed)
        confidence = (float(max_count) / len(predicted_categories)) * 100

        # Cleanup
        confidence = (float(max_count) / len(predicted_categories)) * 100

        # --- Grad-CAM Logic ---
        print("="*80)
        print("DEBUG: Starting Grad-CAM generation...")
        print("="*80)
        sys.stdout.flush()
        
        heatmap_base64 = None
        
        # Find index of a chunk that predicted the winner
        grad_chunk_idx = np.where(predicted_categories == winner_index)[0][0]
        grad_chunk = X_test[grad_chunk_idx]
        print(f"DEBUG: Using chunk {grad_chunk_idx} for Grad-CAM")
        sys.stdout.flush()
        
        # The model has been flattened - all EfficientNet layers are directly in the main model
        # The last conv layer is 'top_conv'
        try:
            last_conv_layer = model.get_layer('top_conv')
            print(f"DEBUG: Found last conv layer: {last_conv_layer.name}")
            sys.stdout.flush()
        except:
            # Fallback: search for it
            last_conv_layer = None
            for layer in model.layers:
                if layer.name == 'top_conv':
                    last_conv_layer = layer
                    break
            
            if not last_conv_layer:
                print("WARNING: Could not find 'top_conv' layer")
                print(f"WARNING: Last 10 layers: {[l.name for l in model.layers[-10:]]}")
                sys.stdout.flush()
                last_conv_layer = None
        
        if last_conv_layer:
            img_array = np.expand_dims(grad_chunk, axis=0)
            
            # Create grad model
            print("DEBUG: Creating grad model...")
            sys.stdout.flush()
            grad_model = keras.models.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
            print("DEBUG: Grad model created successfully")
            sys.stdout.flush()
            
            # Use GradientTape to compute gradients
            print("DEBUG: Computing gradients with GradientTape...")
            sys.stdout.flush()
            with tf.GradientTape() as tape:
                outputs = grad_model(img_array)
                # Outputs come back as a list [conv_output, predictions]
                last_conv_layer_output_val = outputs[0]
                preds = outputs[1]
                # preds might be a list with 1 element in newer Keras
                if isinstance(preds, list):
                    preds = preds[0]
                class_channel = preds[:, winner_index]
            
            print("DEBUG: Computing gradient...")
            sys.stdout.flush()
            grads = tape.gradient(class_channel, last_conv_layer_output_val)
            print(f"DEBUG: Gradients shape: {grads.shape if grads is not None else 'None'}")
            print(f"DEBUG: last_conv_layer_output_val shape: {last_conv_layer_output_val.shape}")
            sys.stdout.flush()
            
            print("DEBUG: Pooling gradients...")
            sys.stdout.flush()
            # Grads should be (1, H, W, C), we want to average over spatial dimensions (H, W)
            pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
            print(f"DEBUG: pooled_grads shape: {pooled_grads.shape}")
            sys.stdout.flush()
            
            print("DEBUG: Creating heatmap...")
            sys.stdout.flush()
            last_conv_layer_output_val = last_conv_layer_output_val[0]
            heatmap = last_conv_layer_output_val @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            print("DEBUG: Normalizing heatmap...")
            sys.stdout.flush()
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            heatmap = heatmap.numpy()
            print(f"DEBUG: Heatmap shape: {heatmap.shape}")
            sys.stdout.flush()
            
            print("DEBUG: Generating base64 image...")
            sys.stdout.flush()
            heatmap_base64 = save_and_display_gradcam(grad_chunk, heatmap)
            print(f"DEBUG: Grad-CAM SUCCESS! Base64 length: {len(heatmap_base64) if heatmap_base64 else 0}")
            sys.stdout.flush()
        
        gradcam_error = None

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

        emit_progress('complete', 'Classification completed')

        return jsonify({
            'status': 'success',
            'predicted_class': result,
            'confidence': round(confidence, 2),
            'explainability': heatmap_base64,
            'details': {
                'chunks_analyzed': len(predicted_categories),
                'votes': dict(zip([label_map[u] for u in unique_elements], counts.tolist())),
                'gradcam_error': str(gradcam_error) if 'gradcam_error' in locals() else None,
                'model_layers': [l.name for l in model.layers] if model else []
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        emit_progress('error', str(e))
        print(f"Error during prediction: {e}")
        # Cleanup
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
