# audio_utils.py (replace existing file)
import numpy as np
import soundfile as sf
import scipy.signal as sps
import noisereduce as nr
import subprocess
import os

def ffmpeg_convert_to_wav(input_file, target_sr=16000):
    output_file = os.path.splitext(input_file)[0] + "_ffmpeg.wav"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-ac", "1",
            "-ar", str(target_sr),
            output_file
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not os.path.exists(output_file):
            raise Exception("FFmpeg failed: output not created")
        return output_file
    except Exception as e:
        print("FFmpeg conversion error:", e)
        return None

def load_wav_with_soundfile(file_path):
    try:
        audio, sr = sf.read(file_path)
        return audio, sr
    except Exception as e:
        print("Error reading WAV:", e)
        return None, None

def normalize_audio(y):
    max_val = np.max(np.abs(y)) if y.size else 0.0
    if max_val > 0:
        return y / max_val
    return y

def resample_if_needed(y, sr, target_sr=16000):
    if sr != target_sr:
        num_samples = int(len(y) * target_sr / sr)
        y = sps.resample(y, num_samples)
        sr = target_sr
    return y, sr

def preprocess_audio(file_path, target_sr=16000):
    # convert
    wav_path = ffmpeg_convert_to_wav(file_path, target_sr=target_sr)
    if wav_path is None:
        raise Exception("FFmpeg conversion failed")
    # load
    y, sr = load_wav_with_soundfile(wav_path)
    if y is None:
        raise Exception("Cannot read converted WAV")
    # stereo->mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # resample if required (should already be 16k from ffmpeg)
    y, sr = resample_if_needed(y, sr, target_sr)
    # noise reduction (safe)
    try:
        y = nr.reduce_noise(y=y, sr=sr)
    except Exception as e:
        print("Noise reduction failed, using raw audio:", e)
    # normalize (numpy)
    y = normalize_audio(y)
    # save cleaned
    cleaned_path = os.path.splitext(file_path)[0] + "_cleaned.wav"
    sf.write(cleaned_path, y, sr)
    return y, sr
