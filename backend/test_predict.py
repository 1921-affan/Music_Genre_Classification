
import requests
import io

url = 'http://127.0.0.1:5000/predict'
# Create a dummy silent wav file in memory
# Minimal WAV header + silence
wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'

files = {'file': ('test.wav', io.BytesIO(wav_header), 'audio/wav')}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
