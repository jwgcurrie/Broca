import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
import requests  # Added for downloading test image

class VLMHandler:
    """
    Handles the Vision Language Model (VLM) using Hugging Face transformers.
    Corrected to align with modern implementation for SmolVLM.
    """
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-256M-Instruct"):
        """
        Initialises the VLM handler. It can be moved to GPU later.
        """
        print("Initialising VLM on CPU...")
        self.device = "cpu"
        self.model_id = model_id

        # Use bfloat16 for better performance
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # The model is initialised on the CPU first
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            # Attention implementation is set dynamically when moving to a device
            _attn_implementation="eager",
        ).to(self.device)
        
        self.tokenizer = self.processor.tokenizer
        print(f"✅ VLM initialised on {self.device}.")

    def _update_attn_implementation(self):
        """Dynamically sets the attention implementation based on the device."""
        if self.device == "cuda" and torch.cuda.is_available():
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "eager"
        
        print(f"Setting attention implementation to '{attn_impl}'...")
        self.model.config._attn_implementation = attn_impl
        # We need to reload the model with the new config on the target device
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            _attn_implementation=attn_impl,
        ).to(self.device)


    def to_gpu(self):
        """Moves the model to the GPU if available."""
        if torch.cuda.is_available() and self.device == "cpu":
            print("Moving VLM to GPU...")
            self.device = "cuda"
            self._update_attn_implementation()
            print("✅ VLM on GPU.")

    def to_cpu(self):
        """Moves the model to the CPU."""
        if self.device != "cpu":
            print("Moving VLM to CPU...")
            self.device = "cpu"
            self._update_attn_implementation()
            print("✅ VLM on CPU.")

    def get_response(self, prompt: str, image: Image.Image, history: list = None):
        """
        Gets a response from the VLM. History is managed outside the function for simplicity.
        """
        if history is None:
            history = []

        # Correctly format messages for VLM with image and text
        # The user message contains a list with image and text parts
        current_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
        
        # Combine history with the current prompt
        messages = history + [current_message]

        # Prepare inputs using the processor's chat template
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True, # Added for more varied responses
            )
        
        # Use batch_decode for clean, simple decoding
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract the assistant's reply from the generated text
        response_text = generated_texts[0].strip()
        if "Assistant:" in response_text:
             response_text = response_text.split("Assistant:")[1].strip()

        # Update history
        history.append(current_message)
        history.append({"role": "assistant", "content": response_text})
        
        return response_text, history

if __name__ == "__main__":
    # URL of the test image from the example
    image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    image_path = "statue_of_liberty.jpg"
    
    try:
        # Download the image
        img_data = requests.get(image_url).content
        with open(image_path, 'wb') as handler:
            handler.write(img_data)
        print(f"Downloaded test image to {image_path}")
        
        image = Image.open(image_path)

        vlm = VLMHandler()
        vlm.to_gpu()  # Try to move to GPU if available
        
        # First question
        prompt1 = "Can you describe this image in one sentence?"
        conversation_history = []
        response1, conversation_history = vlm.get_response(prompt1, image, conversation_history)
        print(f"\nUser: {prompt1}")
        print(f"VLM Response: {response1}")

        # Follow-up question
        prompt2 = "Where is it located?"
        response2, conversation_history = vlm.get_response(prompt2, image, conversation_history)
        print(f"\nUser: {prompt2}")
        print(f"VLM Response: {response2}")
        
        vlm.to_cpu()  # Move back to CPU
        
    except Exception as e:
        print(f"An error occurred during VLM test: {e}")
    finally:
        # Clean up the downloaded image
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"\nRemoved test image at {image_path}")