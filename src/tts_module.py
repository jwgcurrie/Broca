import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf 

class TTSModule:
    def __init__(self):
        print("Initialising Parler TTS model...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Using the v0.1 model, which has this specific input pattern
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
        print(f"✅ Parler TTS model initialised on {self.device}.")

    def generate_speech(self, text, output_path="speech.wav"):
        # Define the voice description (prompt for style)
        description = "A female speaker with a clear voice."

        # Tokenize the *description* (this will go to 'input_ids')
        description_for_model = self.tokenizer(description, return_tensors="pt").to(self.device)

        # Tokenize the *actual text to speak* (this will go to 'prompt_input_ids')
        text_for_model = self.tokenizer(text, return_tensors="pt").to(self.device)

        print(f"Tokenizing text (for prompt_input_ids): '{text}'")
        print(f"Tokenizing description (for input_ids): '{description}'")


        speech = self.model.generate(
            input_ids=description_for_model.input_ids,  # Pass the DESCRIPTION here
            attention_mask=description_for_model.attention_mask, # And its attention mask
            prompt_input_ids=text_for_model.input_ids, # Pass the ACTUAL TEXT here
            prompt_attention_mask=text_for_model.attention_mask # And its attention mask
        ).cpu().numpy()

        print(f"Generated speech for: '{text}' and saved to {output_path}")
        sf.write(output_path, speech.squeeze(), self.model.config.sampling_rate)

if __name__ == "__main__":
    tts = TTSModule()
    tts.generate_speech("Hello, I am Broca, your friendly robot assistant.")