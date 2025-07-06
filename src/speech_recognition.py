import subprocess
import numpy as np
import time
from transformers import pipeline
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class SpeechRecognizer:
    def __init__(self):
        # --- AI Model Initialisation on GPU (if available) ---
        print("Initialising Hugging Face pipeline...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda:0" else torch.float32

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print(f"✅ Pipeline initialised on {self.device}.")

        # --- Audio and VAD Parameters ---
        self.SAMPLE_RATE = 16000
        self.SAMPLE_WIDTH = 2
        self.CHANNELS = 1
        self.SILENCE_DURATION = 1.5
        
        # --- Automatic Noise Calibration ---
        print("Calibrating for ambient noise... Please be quiet for 1 second.")
        try:
            arecord_process = subprocess.Popen(
                ['arecord', '-t', 'raw', '-f', 'S16_LE', '-r', str(self.SAMPLE_RATE), '-c', '1', '-d', '1'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            noise_sample, _ = arecord_process.communicate()
            noise_array = np.frombuffer(noise_sample, dtype=np.int16)
            self.ENERGY_THRESHOLD = np.abs(noise_array).mean() * 2.0
            if self.ENERGY_THRESHOLD < 150:
                self.ENERGY_THRESHOLD = 150
            print(f"✅ Calibration complete. Ambient energy: {np.abs(noise_array).mean():.2f}, New Threshold: {self.ENERGY_THRESHOLD:.2f}")
        except Exception as e:
            print(f"Could not calibrate for noise: {e}. Using a default threshold.")
            self.ENERGY_THRESHOLD = 300


    def transcribe_speech(self):
        # This method is unchanged
        while True:
            try:
                arecord_process = subprocess.Popen(
                    ['arecord', '-t', 'raw', '-f', 'S16_LE', '-r', str(self.SAMPLE_RATE), '-c', '1'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                audio_chunk = b""
                while True:
                    chunk = arecord_process.stdout.read(1024)
                    if not chunk: break
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    if np.abs(audio_array).mean() > self.ENERGY_THRESHOLD:
                        print("Speech detected! Recording phrase...")
                        audio_chunk = chunk
                        break
                audio_data_chunks = [audio_chunk]
                silent_chunks = 0
                chunks_per_second = (self.SAMPLE_RATE * self.SAMPLE_WIDTH) / 1024
                while silent_chunks < (self.SILENCE_DURATION * chunks_per_second):
                    chunk = arecord_process.stdout.read(1024)
                    if not chunk: break
                    audio_data_chunks.append(chunk)
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    if np.abs(audio_array).mean() < self.ENERGY_THRESHOLD: silent_chunks += 1
                    else: silent_chunks = 0
                arecord_process.terminate()
                arecord_process.wait()
                print("Phrase recorded. Now transcribing...")
                frame_data = b''.join(audio_data_chunks)
                audio_array = np.frombuffer(frame_data, dtype=np.int16)
                audio_float32 = audio_array.astype(np.float32) / 32768.0
                with torch.no_grad():
                    result = self.pipe(audio_float32, generate_kwargs={"language": "english", "task": "transcribe"})
                text = result["text"].strip()
                print(f"Transcribed: {text}")
                return text
            except KeyboardInterrupt:
                if hasattr(self, 'arecord_process') and self.arecord_process.poll() is None:
                    self.arecord_process.terminate()
                print("\nShutting down speech recognition.")
                return None
            except Exception as e:
                print(f"An error occurred during speech recognition: {e}")
                if hasattr(self, 'arecord_process') and self.arecord_process.poll() is None:
                    self.arecord_process.terminate()
                return None

    def cleanup(self):
        if hasattr(self, 'arecord_process') and self.arecord_process.poll() is None:
            print("Terminating arecord process...")
            self.arecord_process.terminate()
            self.arecord_process.wait()
            print("arecord process terminated.")