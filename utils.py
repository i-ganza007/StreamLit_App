import requests

# REPLACE this with your actual backend URL
BASE_URL = "https://shoe-type-classifier-summative.onrender.com"

def get_predictions():
    return requests.get(f"{BASE_URL}/predictions/").json()

def get_metrics():
    return requests.get(f"{BASE_URL}/metrics/").json()

def get_training_data():
    return requests.get(f"{BASE_URL}/training-data/").json()

def upload_image_for_prediction(file):
    files = {"file": file}
    response = requests.post(f"{BASE_URL}/predict/", files=files)
    return response.json()

def upload_zip_file(file):
    files = {"zip_file": file}
    response = requests.post(f"{BASE_URL}/upload-zip/", files=files)
    return response.json()

def trigger_retrain():
    return requests.post(f"{BASE_URL}/retrain/").json()
