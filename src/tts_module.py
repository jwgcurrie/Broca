import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import sounddevice as sd
import numpy as np

class TTSModule:
    def __init__(self):
        print("Initialising Parler TTS model on CPU...")
        self.device = "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
        print(f"✅ Parler TTS model initialised on {self.device}.")

    def to_gpu(self):
        """Moves the model to the GPU if available."""
        if torch.cuda.is_available() and self.device == "cpu":
            print("Moving TTS model to GPU...")
            self.device = "cuda:0"
            self.model.to(self.device)
            print("✅ TTS model on GPU.")

    def to_cpu(self):
        """Moves the model to the CPU."""
        if self.device != "cpu":
            print("Moving TTS model to CPU...")
            self.device = "cpu"
            self.model.to(self.device)
            print("✅ TTS model on CPU.")

    def verbalise_speech(self, text):
        description = "A female speaker with a clear, robotic voice." 
        description_for_model = self.tokenizer(description, return_tensors="pt").to(self.device)
        text_for_model = self.tokenizer(text, return_tensors="pt").to(self.device)

        print(f"Tokenizing text (for prompt_input_ids): '{text}'")
        print(f"Tokenizing description (for input_ids): '{description}'")

        with torch.no_grad():
            speech = self.model.generate(
                input_ids=description_for_model.input_ids,
                attention_mask=description_for_model.attention_mask,
                prompt_input_ids=text_for_model.input_ids,
                prompt_attention_mask=text_for_model.attention_mask
            ).cpu().numpy()

        model_samplerate = self.model.config.sampling_rate

        playback_samplerate = 48000
        print(f"Playing speech (model sample rate: {model_samplerate} Hz, playback sample rate: {playback_samplerate} Hz)...")

        try:
            sd.play(speech.squeeze(), samplerate=playback_samplerate, blocking=True)
            print(f"Verbalised: '{text}'")
        except sd.PortAudioError as e:
            print(f"Error playing audio: {e}")
            print("Trying to list available devices and their supported sample rates...")
            self.list_audio_devices() 

    def list_audio_devices(self):
        try:
            devices = sd.query_devices()
            print("\nAvailable Audio Devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device['name']}")
                print(f"     Host API: {sd.query_hostapis(device['hostapi'])['name']}")
                print(f"     Max Input Channels: {device['max_input_channels']}")
                print(f"     Max Output Channels: {device['max_output_channels']}")
                try:
                    default_output_rate = sd.query_devices(device['name'], kind='output')['default_samplerate']
                    print(f"     Default Output Sample Rate: {default_output_rate} Hz")
                except sd.PortAudioError:
                    print(f"     Default Output Sample Rate: N/A (could not query)")
        except Exception as e:
            print(f"Could not list audio devices: {e}")

if __name__ == "__main__":
    tts = TTSModule()
    tts.to_gpu()
    tts.verbalise_speech("Hello, I am Broca, your friendly robot assistant.")