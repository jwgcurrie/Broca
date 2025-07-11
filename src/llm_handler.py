import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from keyword_manager import KeywordManager # Import KeywordManager

class LLMHandler:
    """
    Handles the local LLM using Hugging Face transformers.
    """
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-360M-Instruct", system_prompt="", max_history=5, verbose=False):
        """
        Initializes the LLM handler on the CPU.
        """
        print("Initialising LLM on CPU...")
        self.device = "cpu"
        self.model_id = model_id # Store model_id for loading
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.verbose = verbose
        
        # Initialize KeywordManager
        self.keyword_manager = KeywordManager(model_id=model_id, verbose=verbose)

        if self.verbose: print(f"✅ LLM initialised on {self.device}.")

    def to_gpu(self):
        """Moves the model to the GPU if available."""
        if torch.cuda.is_available() and self.device == "cpu":
            if self.verbose: print("Moving LLM to GPU...")
            self.device = "cuda:0"
            self.model.to(self.device)
            self.keyword_manager.to_gpu() # Move keyword manager's model to GPU
            if self.verbose: print("✅ LLM on GPU.")

    def to_cpu(self):
        """Moves the model to the CPU."""
        if self.device != "cpu":
            if self.verbose: print("Moving LLM to CPU...")
            self.device = "cpu"
            self.model.to(self.device)
            self.keyword_manager.to_cpu() # Move keyword manager's model to CPU
            if self.verbose: print("✅ LLM on CPU.")

    def get_response(self, prompt, history):
        """
        Gets a response from the LLM, maintaining conversation history.
        """
        # Update keywords with user's prompt
        self.keyword_manager.update_keywords(prompt)

        history.append({"role": "user", "content": prompt})
        
        # Construct dynamic system prompt
        keyword_hint = self.keyword_manager.get_keyword_string()
        dynamic_system_prompt = f"{self.system_prompt} {keyword_hint}".strip()

        messages = [{"role": "system", "content": dynamic_system_prompt}] + history + [{"role": "assistant", "content": ""}]
        prompt_string = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_string, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            res = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.75,
                top_p=0.95,
                repetition_penalty=1.2, # Added repetition penalty
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        full_response_text = self.tokenizer.decode(res[0][input_length:], skip_special_tokens=True).strip()
        
        # Aggressive conciseness: Extract only the first complete sentence
        first_sentence_match = re.match(r"^[^.!?]*[.!?]", full_response_text)
        if first_sentence_match:
            response_text = first_sentence_match.group(0).strip()
        else:
            # If no complete sentence found, take the first part up to a reasonable length
            response_text = full_response_text.split('\n')[0].strip() # Take first line
            if len(response_text) > 100: # Arbitrary length to prevent very long first lines
                response_text = response_text[:100] + "..."

        # Update keywords with assistant's response (the truncated one)
        self.keyword_manager.update_keywords(response_text)

        history.append({"role": "assistant", "content": response_text})
        if len(history) > self.max_history * 2:
            history = history[-(self.max_history * 2):]
        return response_text, history
