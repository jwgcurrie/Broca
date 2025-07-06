import subprocess
import numpy as np
import time
from transformers import pipeline
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # Suppress the specific FutureWarning from the transformers library 

class SpeechRecognizer:
    def __init__(self):
        # --- AI Model Initialisation ---
        print("Initialising Hugging Face pipeline...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device=self.device,
        )
        print(f"✅ Pipeline initialised on {self.device}.")

        # --- Audio Parameters ---
        self.SAMPLE_RATE = 16000  # 16kHz is standard for Whisper
        self.SAMPLE_WIDTH = 2     # Bytes per sample (S16_LE = 16 bit = 2 bytes)
        self.CHANNELS = 1

        # --- Voice Activity Detection (VAD) Parameters ---
        self.SILENCE_DURATION = 1.5  # How many seconds of silence indicates the end of a phrase.
        
        # --- Automatic Noise Calibration ---
        print("Calibrating for ambient noise... Please be quiet for 1 second.")
        try:
            # Run arecord for 1 second to capture background noise
            arecord_process = subprocess.Popen(
                ['arecord', '-t', 'raw', '-f', 'S16_LE', '-r', str(self.SAMPLE_RATE), '-c', '1', '-d', '1'],
                stdout=subprocess.PIPE,
                # Redirect stderr to DEVNULL to hide ALSA messages
                stderr=subprocess.DEVNULL
            )
            noise_sample, _ = arecord_process.communicate()
            
            # Calculate the energy of the noise sample
            noise_array = np.frombuffer(noise_sample, dtype=np.int16)
            self.ENERGY_THRESHOLD = np.abs(noise_array).mean() * 2.0
            
            # Add a minimum threshold to handle very quiet environments
            if self.ENERGY_THRESHOLD < 150:
                self.ENERGY_THRESHOLD = 150
                
            print(f"✅ Calibration complete. Ambient energy: {np.abs(noise_array).mean():.2f}, New Threshold: {self.ENERGY_THRESHOLD:.2f}")

        except Exception as e:
            print(f"Could not calibrate for noise: {e}. Using a default threshold.")
            self.ENERGY_THRESHOLD = 300 # Fallback to a default value

    def transcribe_speech(self):
        while True:
            try:
                # 1. Listen for speech to begin
                print("\nListening for speech...")
                
                # Start arecord and listen until we hear sound above the threshold
                arecord_process = subprocess.Popen(
                    ['arecord', '-t', 'raw', '-f', 'S16_LE', '-r', str(self.SAMPLE_RATE), '-c', '1'],
                    stdout=subprocess.PIPE,
                    # Redirect stderr to DEVNULL to hide ALSA messages
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
                
                # 2. Record the phrase until silence is detected
                audio_data_chunks = [audio_chunk] # Start with the chunk that triggered detection
                silent_chunks = 0
                chunks_per_second = (self.SAMPLE_RATE * self.SAMPLE_WIDTH) / 1024
                
                while silent_chunks < (self.SILENCE_DURATION * chunks_per_second):
                    chunk = arecord_process.stdout.read(1024)
                    if not chunk: break
                    audio_data_chunks.append(chunk)
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    
                    if np.abs(audio_array).mean() < self.ENERGY_THRESHOLD:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0 # Reset silence counter on sound

                # Terminate the arecord process now that we have the phrase
                arecord_process.terminate()
                print("Phrase recorded. Now transcribing...")

                # 3. Transcribe the captured audio
                # Combine all the audio chunks into a single byte string
                frame_data = b''.join(audio_data_chunks)
                
                # Convert raw bytes to a NumPy array
                audio_array = np.frombuffer(frame_data, dtype=np.int16)
                
                # Convert to float32 and normalise to the range [-1, 1], which the model expects
                audio_float32 = audio_array.astype(np.float32) / 32768.0
                
                # Pass the audio data to the Hugging Face pipeline
                result = self.pipe(
                    audio_float32,
                    generate_kwargs={"language": "english", "task": "transcribe"}
                )
                text = result["text"].strip()
                
                print(f"Transcribed: {text}")
                return text

            except KeyboardInterrupt:
                # Ensure the subprocess is killed on exit
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
            self.arecord_process.wait() # Wait for the process to actually terminate
            print("arecord process terminated.")