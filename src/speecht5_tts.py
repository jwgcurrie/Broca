import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import sounddevice as sd
import torchaudio
import numpy as np 

class SpeechT5TTSModule:
    """
    A robust TTS module using Microsoft SpeechT5 that resamples audio for compatibility.
    """
    def __init__(self):
        # Initialise on CPU to save GPU memory at startup
        print("Initialising SpeechT5 TTS model on CPU...")
        self.device = "cpu"
        
        # Load the processor, model, and vocoder
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Load the speaker embedding from the official dataset
        print("Loading speaker embedding from Hugging Face Hub...")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(self.device)

        print("✅ Speaker embedding loaded.")
        print(f"✅ SpeechT5 TTS model initialised on {self.device}.")

    def to_gpu(self):
        """Moves the models to the GPU if available."""
        if torch.cuda.is_available() and self.device == "cpu":
            gpu_dtype = torch.float16
            print("Moving SpeechT5 model to GPU...")
            self.device = "cuda:0"
            self.model.to(self.device, dtype=gpu_dtype)
            self.vocoder.to(self.device, dtype=gpu_dtype)
            self.speaker_embeddings = self.speaker_embeddings.to(self.device, dtype=gpu_dtype)
            print("✅ SpeechT5 model on GPU.")

    def to_cpu(self):
        """Moves the models to the CPU."""
        if self.device != "cpu":
            print("Moving SpeechT5 model to CPU...")
            self.device = "cpu"
            self.model.to(self.device)
            self.vocoder.to(self.device)
            self.speaker_embeddings = self.speaker_embeddings.to(self.device)
            print("✅ SpeechT5 model on CPU.")

    def verbalise_speech(self, text):
        """
        Synthesizes and plays audio for the given text.
        """
        print(f"Synthesizing speech with SpeechT5 for: '{text}'")
        
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            spectrogram = self.model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings=self.speaker_embeddings
            )
            audio_tensor = self.vocoder(spectrogram).cpu().to(torch.float32)

        source_sample_rate = 16000
        target_sample_rate = 48000

        # The torchaudio resampler can now work with the float32 tensor
        print(f"Resampling audio from {source_sample_rate} Hz to {target_sample_rate} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=source_sample_rate, new_freq=target_sample_rate)
        resampled_tensor = resampler(audio_tensor)
        
        audio_for_playback = resampled_tensor.squeeze().numpy()

        try:
            print(f"Playing speech (sample rate: {target_sample_rate} Hz)...")
            sd.play(audio_for_playback, samplerate=target_sample_rate, blocking=True)
            print(f"Verbalised: '{text}'")
        except Exception as e:
            print(f"Error playing audio with sounddevice: {e}")

if __name__ == "__main__":
    tts = SpeechT5TTSModule()
    tts.to_gpu()
    print("\n--- Running Standalone Test ---")
    test_sentence = "This final version should now have the correct data types for all steps."
    tts.verbalise_speech(test_sentence)
    print("--- Standalone Test Complete ---")