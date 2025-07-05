
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMHandler:
    """
    Handles the local LLM using Hugging Face transformers.
    """
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-360M-Instruct", system_prompt="", max_history=5):
        """
        Initializes the LLM handler.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.system_prompt = system_prompt
        self.max_history = max_history

    def get_response(self, prompt, history):
        """
        Gets a response from the LLM, maintaining conversation history.
        """
        # Add the new user prompt to the history
        history.append({"role": "user", "content": prompt})

        # Construct the full message list, including the system prompt and an empty assistant turn
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "assistant", "content": ""}]

        # Create the prompt string from the chat template
        prompt_string = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the prompt string to get input IDs and attention mask
        inputs = self.tokenizer(
            prompt_string,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            # Pass both input_ids and attention_mask to the generate function
            res = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response_tokens = res[0][input_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        # Add the assistant's response to the history
        history.append({"role": "assistant", "content": response_text})

        # Trim the history to the maximum length (keeping the most recent interactions)
        if len(history) > self.max_history * 2: # Each turn has a user and assistant message
            history = history[-(self.max_history * 2):]

        return response_text, history
