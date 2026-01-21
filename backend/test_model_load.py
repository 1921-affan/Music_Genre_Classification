
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import traceback

print("Keras version:", keras.__version__)

try:
    print("Attempting to load model...")
    model = keras.models.load_model("EfficientNet_Model.keras")
    print("SUCCESS: Model loaded.")
    model.summary()
except Exception:
    print("FAILURE: Model could not be loaded.")
    traceback.print_exc()
