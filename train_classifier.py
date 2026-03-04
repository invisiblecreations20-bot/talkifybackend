import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import soundfile as sf
from python_speech_features import mfcc

DATASET_DIR = "dataset"
MODEL_PATH = "models/gender_classifier.joblib"

def extract_features(file_path):
    y, sr = sf.read(file_path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    # ensure 16000
    if sr != 16000:
        import scipy.signal as sps
        num_samples = int(len(y) * 16000 / sr)
        y = sps.resample(y, num_samples)
        sr = 16000
    # compute MFCC (13)
    mf = mfcc(y, samplerate=sr, numcep=13, nfilt=26, winlen=0.025, winstep=0.01)
    return np.mean(mf, axis=0)

def load_data():
    X, y = [], []
    for language in os.listdir(DATASET_DIR):
        lang_path = os.path.join(DATASET_DIR, language)
        if not os.path.isdir(lang_path):
            continue
        for label, gender in enumerate(['male', 'female']):
            folder = os.path.join(lang_path, gender)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if file.lower().endswith(".wav"):
                    path = os.path.join(folder, file)
                    print(f"Loading: {path}")
                    try:
                        features = extract_features(path)
                        X.append(features)
                        y.append(gender)
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
    return np.array(X), np.array(y)

def train_model():
    print("Loading data...")
    X, y = load_data()
    print(f"Total samples loaded: {len(X)}")
    if len(X) == 0:
        print("No data found. Please prepare dataset directory with male/female WAVs.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
