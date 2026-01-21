import os
import zipfile
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import gc

# --- 1. CONFIGURATION ---
DATASET_PATH = "" # Will be set after unzip
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 15  # Strictly 15
SLICES_PER_TRACK = 10

# --- 2. ROBUST UNZIP ---
# This ensures the data is ready even if you restart the runtime
ZIP_PATH = '/content/genres_original.zip'
EXTRACT_PATH = '/content/dataset_extracted'

if not os.path.exists(EXTRACT_PATH):
    print(f"Unzipping {ZIP_PATH}...")
    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_PATH)
            print("Unzip complete.")
        except zipfile.BadZipFile:
            print("ERROR: Zip file corrupted. Please re-upload.")
            exit()
    else:
        print(f"ERROR: {ZIP_PATH} not found. Please upload it.")
        exit()
else:
    print("Dataset already unzipped.")

# Find the correct folder (handles different zip structures)
for root, dirs, files in os.walk(EXTRACT_PATH):
    if 'blues' in dirs:
        DATASET_PATH = root
        break
if not DATASET_PATH:
    DATASET_PATH = os.path.join(EXTRACT_PATH, "genres_original")

# --- 3. RAM-OPTIMIZED DATA LOADING ---
def load_data_sliced_optimized(dataset_path, slices_per_track=10):
    print("Counting files to pre-allocate memory...")
    genres = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    genre_to_id = {genre: i for i, genre in enumerate(genres)}

    total_files = sum([len([f for f in os.listdir(os.path.join(dataset_path, g)) if f.endswith('.wav')]) for g in genres])
    total_slices = total_files * slices_per_track
    print(f"Total files: {total_files}. Total slices to generate: {total_slices}")

    # Pre-allocate using uint8 (0-255) to save RAM
    X = np.zeros((total_slices, 224, 224, 3), dtype=np.uint8)
    y = np.zeros((total_slices,), dtype=np.int32)

    SAMPLES_PER_SLICE = int(SAMPLES_PER_TRACK / slices_per_track)
    current_idx = 0

    print("Processing audio...")
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        print(f"  -> Processing {genre}...")

        for filename in os.listdir(genre_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(genre_path, filename)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                    if len(signal) < SAMPLES_PER_TRACK:
                         signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), mode='constant')

                    for s in range(slices_per_track):
                        start = s * SAMPLES_PER_SLICE
                        end = start + SAMPLES_PER_SLICE
                        slice_sig = signal[start:end]

                        mel = librosa.feature.melspectrogram(y=slice_sig, sr=sr, n_mels=128)
                        log_mel = librosa.power_to_db(mel, ref=np.max)

                        resized = cv2.resize(log_mel, (224, 224))
                        rgb = np.stack([resized] * 3, axis=-1)

                        # Normalize to 0-255
                        min_v, max_v = np.min(rgb), np.max(rgb)
                        if max_v > min_v:
                            norm = (rgb - min_v) / (max_v - min_v) * 255.0
                        else:
                            norm = np.zeros_like(rgb)

                        if current_idx < total_slices:
                            X[current_idx] = norm.astype(np.uint8)
                            y[current_idx] = genre_to_id[genre]
                            current_idx += 1
                except Exception:
                    continue
        gc.collect() # RAM cleanup

    return X[:current_idx], y[:current_idx], genres

# --- 4. MODEL CREATION (FIXED) ---
def create_fixed_model(input_shape, num_classes):
    # FIX 1: INPUT SCALING
    # We use Rescaling(1.0) to just cast to float without changing the 0-255 values.
    # EfficientNet expects 0-255 inputs.
    inputs = tf.keras.Input(shape=input_shape)
    x = Rescaling(1.0)(inputs)

    # Load Base Model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)

    # FIX 2: FREEZE BATCH NORMALIZATION
    # We allow the weights to train...
    base_model.trainable = True
    # ...BUT we strictly freeze the BN layers to keep stats stable.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Head Architecture
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 5. PLOTTING FUNCTIONS ---
def plot_history(history):
    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Model Accuracy ({EPOCHS} Epochs)')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig('EfficientNet_Accuracy.png')
    plt.show()

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Model Loss ({EPOCHS} Epochs)')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig('EfficientNet_Loss.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig('EfficientNet_Confusion_Matrix.png')
    plt.show()

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    gc.collect() # Clean start
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # 1. Load Data
    X, y, genres = load_data_sliced_optimized(DATASET_PATH, slices_per_track=SLICES_PER_TRACK)
    print(f"Data Loaded. Shape: {X.shape}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Create Model
    model = create_fixed_model(INPUT_SHAPE, NUM_CLASSES)

    # 4. Train (Strictly 15 Epochs, No Early Stopping)
    print(f"Starting training for {EPOCHS} epochs...")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[reduce_lr])

    # 5. Metrics & Graphs
    print("\n--- Generating Metrics ---")
    try:
        plot_history(history)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        report = classification_report(y_test, y_pred, target_names=genres)
        print("\nClassification Report:")
        print(report)

        with open("classification_report.txt", "w") as f:
            f.write(report)

        plot_confusion_matrix(y_test, y_pred, genres)
        print("All graphs saved.")

    except Exception as e:
        print(f"Error generating metrics: {e}")
        traceback.print_exc()

    # 6. Save Model (Triple Backup)
    print("\n--- Saving Model ---")
    # 1. Keras (Modern)
    try:
        model.save("EfficientNet_Model.keras")
        print("Saved: EfficientNet_Model.keras")
    except Exception as e:
        print(f"Failed to save .keras: {e}")

    # 2. H5 (Legacy - Safe Mode)
    try:
        model.save("EfficientNet_Model.h5", include_optimizer=False)
        print("Saved: EfficientNet_Model.h5")
    except Exception as e:
        print(f"Failed to save .h5: {e}")

    # 3. Weights (Failsafe)
    try:
        model.save_weights("EfficientNet_Model.weights.h5")
        print("Saved: EfficientNet_Model.weights.h5")
    except Exception as e:
        print(f"Failed to save weights: {e}")