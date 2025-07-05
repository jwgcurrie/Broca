import subprocess
import numpy as np
import time
from transformers import pipeline
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # Suppress the specific FutureWarning from the transformers library 


def main():
    """
    Captures audio using the 'arecord' command-line tool and transcribes it
    using the Hugging Face Transformers library with a local Whisper model.
    This bypasses PyAudio/sounddevice issues.
    """
    # --- AI Model Initialisation ---
    print("Initialising Hugging Face pipeline...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device=device,
    )
    print(f"✅ Pipeline initialised on {device}.")

    # --- Audio Parameters ---
    # These must match the arecord command
    SAMPLE_RATE = 16000  # 16kHz is standard for Whisper
    SAMPLE_WIDTH = 2     # Bytes per sample (S16_LE = 16 bit = 2 bytes)
    CHANNELS = 1

    # --- Voice Activity Detection (VAD) Parameters ---
    SILENCE_DURATION = 1.5  # How many seconds of silence indicates the end of a phrase.
    
    # --- Automatic Noise Calibration ---
    print("Calibrating for ambient noise... Please be quiet for 1 second.")
    try:
        # Run arecord for 1 second to capture background noise
        arecord_process = subprocess.Popen(
            ['arecord', '-t', 'raw', '-f', 'S16_LE', '-r', str(SAMPLE_RATE), '-c', '1', '-d', '1'],
            stdout=subprocess.PIPE,
            # Redirect stderr to DEVNULL to hide ALSA messages
            stderr=subprocess.DEVNULL
        )
        noise_sample, _ = arecord_process.communicate()
        
        # Calculate the energy of the noise sample
        noise_array = np.frombuffer(noise_sample, dtype=np.int16)
        ambient_energy = np.abs(noise_array).mean()
        
        # Set the energy threshold to be a multiple of the ambient noise
        ENERGY_THRESHOLD = ambient_energy * 2.0
        
        # Add a minimum threshold to handle very quiet environments
        if ENERGY_THRESHOLD < 150:
            ENERGY_THRESHOLD = 150
            
        print(f"✅ Calibration complete. Ambient energy: {ambient_energy:.2f}, New Threshold: {ENERGY_THRESHOLD:.2f}")

    except Exception as e:
        print(f"Could not calibrate for noise: {e}. Using a default threshold.")
        ENERGY_THRESHOLD = 300 # Fallback to a default value

    while True:
        try:
            # 1. Listen for speech to begin
            print("\nListening for speech...")
            
            # Start arecord and listen until we hear sound above the threshold
            arecord_process = subprocess.Popen(
                ['arecord', '-t', 'raw', '-f', 'S16_LE', '-r', str(SAMPLE_RATE), '-c', '1'],
                stdout=subprocess.PIPE,
                # Redirect stderr to DEVNULL to hide ALSA messages
                stderr=subprocess.DEVNULL
            )
            
            while True:
                audio_chunk = arecord_process.stdout.read(1024)
                if not audio_chunk: break
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                if np.abs(audio_array).mean() > ENERGY_THRESHOLD:
                    print("Speech detected! Recording phrase...")
                    break
            
            # 2. Record the phrase until silence is detected
            audio_data_chunks = [audio_chunk] # Start with the chunk that triggered detection
            silent_chunks = 0
            chunks_per_second = (SAMPLE_RATE * SAMPLE_WIDTH) / 1024
            
            while silent_chunks < (SILENCE_DURATION * chunks_per_second):
                audio_chunk = arecord_process.stdout.read(1024)
                if not audio_chunk: break
                audio_data_chunks.append(audio_chunk)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                
                if np.abs(audio_array).mean() < ENERGY_THRESHOLD:
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
            result = pipe(
                audio_float32,
                generate_kwargs={"language": "english", "task": "transcribe"}
            )
            text = result["text"].strip()
            
            print(f"Transcribed: {text}")

        except KeyboardInterrupt:
            # Ensure the subprocess is killed on exit
            if 'arecord_process' in locals() and arecord_process.poll() is None:
                arecord_process.terminate()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            if 'arecord_process' in locals() and arecord_process.poll() is None:
                arecord_process.terminate()
            break
            
    print("\nShutting down.")

if __name__ == "__main__":
    main()
