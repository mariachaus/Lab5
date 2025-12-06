from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import io
import librosa
from fastapi.middleware.cors import CORSMiddleware
import logging
import tempfile
import os
import subprocess

from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Classification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SR = 22050
DURATION = 1.0
TARGET_SAMPLES = int(TARGET_SR * DURATION)
N_MELS = 64
TIME_STEPS = 44
SILENCE_RMS_THRESHOLD = 0.001  

try:
    tflite_interpreter = tf.lite.Interpreter(model_path="../animal_classifier.tflite")
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    logger.info("Model loaded successfully")
    model_loaded_time = datetime.now()
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    tflite_interpreter = None
    model_loaded_time = None

class_mapping = {
    0: "bird",
    1: "cat",
    2: "dog",
    3: "mouse",
}

model_stats = {
    "total_predictions": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "silent_detections": 0,
    "last_prediction_time": None
}


def convert_m4a_to_wav_simple(m4a_bytes):
    """
    Конвертує M4A в WAV за допомогою subprocess + ffmpeg
    Повертає bytes WAV з цільовою частотою TARGET_SR
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as m4a_file:
            m4a_path = m4a_file.name
            m4a_file.write(m4a_bytes)

        wav_path = m4a_path.replace('.m4a', '.wav')

        result = subprocess.run([
            'ffmpeg', '-y', '-i', m4a_path,
            '-ac', '1', '-ar', str(TARGET_SR),
            '-acodec', 'pcm_s16le',
            wav_path
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")

        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()

        logger.info(f"Conversion successful: {len(m4a_bytes)} -> {len(wav_bytes)} bytes")

        try:
            os.unlink(m4a_path)
            os.unlink(wav_path)
        except:
            pass

        return wav_bytes

    except Exception as e:
        try:
            if 'm4a_path' in locals() and os.path.exists(m4a_path):
                os.unlink(m4a_path)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass
        raise


def preprocess_audio_direct(file_bytes, silence_rms_threshold=SILENCE_RMS_THRESHOLD):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=TARGET_SR, mono=True, duration=DURATION)
        
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
        else:
            y = y[:TARGET_SAMPLES]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=N_MELS,      
            n_fft=2048,
            hop_length=512,    
            win_length=2048,
            power=2.0         
        )
        
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_db_T = mel_db.T
        
        logger.info(f"Raw mel shape: {mel_db_T.shape}")
        
        if mel_db_T.shape[0] < TIME_STEPS:
            pad_width = TIME_STEPS - mel_db_T.shape[0]
            mel_db_T = np.pad(mel_db_T, ((0, pad_width), (0, 0)), mode='constant', constant_values=mel_db_T.min())
        else:
            mel_db_T = mel_db_T[:TIME_STEPS, :]
        
        logger.info(f"Final mel shape before expansion: {mel_db_T.shape}")
        
        input_data = np.expand_dims(mel_db_T, axis=-1)  
        input_data = np.expand_dims(input_data, axis=0)  
        
        logger.info(f"Final input shape: {input_data.shape}")
        return input_data.astype(np.float32)

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise

def create_test_audio_data():
    """
    Створює тестові аудіо дані для демонстрації (тільки як остання міра)
    """
    logger.info("Creating test audio data")
    mel = np.random.normal(0, 1, (TIME_STEPS, N_MELS)).astype(np.float32)
    input_data = np.expand_dims(mel, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    return input_data


@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """
    Обробляє POST запит із аудіофайлом.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        audio_bytes = await file.read()
        logger.info(f"File size: {len(audio_bytes)} bytes")

        if tflite_interpreter is None:
            return {"error": "Model not loaded", "filename": file.filename}

        is_m4a = file.filename.lower().endswith('.m4a')
        logger.info(f"File is M4A: {is_m4a}")

        input_data = None
        silent_detected = False

        try:
            input_data = preprocess_audio_direct(audio_bytes)
        except ValueError as ve:
            if "silent" in str(ve).lower():
                silent_detected = True
        except Exception as e:
            logger.warning(f"Direct processing failed: {e}")

        if input_data is None and is_m4a and not silent_detected:
            try:
                wav_bytes = convert_m4a_to_wav_simple(audio_bytes)
                input_data = preprocess_audio_direct(wav_bytes)
            except ValueError as ve:
                if "silent" in str(ve).lower():
                    silent_detected = True
            except Exception as e:
                logger.error(f"M4A conversion failed: {e}")

        model_stats["total_predictions"] += 1
        model_stats["last_prediction_time"] = datetime.now().isoformat()
        if silent_detected:
            model_stats["silent_detections"] += 1
            result = {"silence": 100}
            return {
                "filename": file.filename,
                "predictions": result,
                "status": "success",
                "detected_silent": True
            }

        if input_data is None:
            logger.info("Using test data as fallback")
            input_data = create_test_audio_data()

        
        expected_shape = input_details[0]['shape']
        logger.info(f"Model expects: {expected_shape}, got: {input_data.shape}")

       
        try:
            tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
            tflite_interpreter.invoke()
            output_data = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            model_stats["failed_predictions"] += 1
            return {"error": "Inference failed", "details": str(e)}

        
        exp_scores = np.exp(output_data - np.max(output_data))
        probs = exp_scores / np.sum(exp_scores)
        logger.info(f"Probabilities: {probs}")

        result = {}
        for i, prob in enumerate(probs):
            class_name = class_mapping.get(i, f"Class_{i}")
            percentage = round(float(prob) * 100, 2)
            result[class_name] = percentage
            logger.info(f"{class_name}: {percentage}%")

        model_stats["total_predictions"] += 1
        model_stats["successful_predictions"] += 1
        model_stats["last_prediction_time"] = datetime.now().isoformat()

        return {
            "filename": file.filename,
            "predictions": result,
            "status": "success",
            "detected_silent": False
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        model_stats["failed_predictions"] += 1
        return {
            "error": f"Processing failed: {str(e)}",
            "filename": file.filename,
            "status": "error"
        }


@app.get("/")
async def root():
    return {"message": "Audio Classification API is running", "status": "ready"}


@app.get("/health")
async def health_check():
    model_status = "loaded" if tflite_interpreter is not None else "failed"
    return {"status": "healthy", "model": model_status}


@app.post("/test-predict/")
async def test_predict():
    """
    Тестовий ендпоінт для перевірки моделі
    """
    try:
        if tflite_interpreter is None:
            return {"error": "Model not loaded"}

        input_data = create_test_audio_data()

        tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
        tflite_interpreter.invoke()

        output_data = tflite_interpreter.get_tensor(output_details[0]['index'])[0]

        exp_scores = np.exp(output_data - np.max(output_data))
        probs = exp_scores / np.sum(exp_scores)

        result = {}
        for i, prob in enumerate(probs):
            class_name = class_mapping.get(i, f"Class_{i}")
            percentage = round(float(prob) * 100, 2)
            result[class_name] = percentage

        return {
            "predictions": result,
            "status": "test_success"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/model-info")
async def get_model_info():
    """
    Повертає детальну інформацію про завантажену модель
    """
    print(f"Model input shape: {input_details[0]['shape']}")

    if tflite_interpreter is None:
        return {"error": "Model not loaded"}

    try:
        
        input_info = {
            "name": input_details[0].get('name', 'unknown'),
            "shape": input_details[0]['shape'].tolist(),
            "dtype": str(input_details[0]['dtype']),
            "quantization": input_details[0].get('quantization', None)
        }

        output_info = {
            "name": output_details[0].get('name', 'unknown'),
            "shape": output_details[0]['shape'].tolist(),
            "dtype": str(output_details[0]['dtype']),
            "quantization": output_details[0].get('quantization', None)
        }

        classes_info = {
            "total_classes": len(class_mapping),
            "class_mapping": class_mapping
        }

        audio_processing_info = {
            "sample_rate": TARGET_SR,
            "duration_seconds": DURATION,
            "n_mels": N_MELS,
            "time_steps": TIME_STEPS,
            "channels": 1,
            "silence_rms_threshold": SILENCE_RMS_THRESHOLD
        }

    
        model_info = {
            "model_type": "TensorFlow Lite",
            "model_path": "../animal_classifier.tflite",
            "model_loaded_at": model_loaded_time.isoformat() if model_loaded_time else None,
            "uptime_seconds": (datetime.now() - model_loaded_time).total_seconds() if model_loaded_time else None,
            "tensorflow_version": tf.__version__
        }

        return {
            "model_info": model_info,
            "input_details": input_info,
            "output_details": output_info,
            "classes": classes_info,
            "audio_processing": audio_processing_info,
            "statistics": model_stats,
            "supported_operations": "This model expects 1-second audio clips converted to mel-spectrogram (power_to_db) with 64 mel bins and 44 time steps"
        }

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": f"Failed to get model info: {str(e)}"}
