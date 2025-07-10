import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        if self.verbose: print(f"✅ LLM initialised on {self.device}.")

    def to_gpu(self):
        """Moves the model to the GPU if available."""
        if torch.cuda.is_available() and self.device == "cpu":
            if self.verbose: print("Moving LLM to GPU...")
            self.device = "cuda:0"
            self.model.to(self.device)
            if self.verbose: print("✅ LLM on GPU.")

    def to_cpu(self):
        """Moves the model to the CPU."""
        if self.device != "cpu":
            if self.verbose: print("Moving LLM to CPU...")
            self.device = "cpu"
            self.model.to(self.device)
            if self.verbose: print("✅ LLM on CPU.")

    def get_response(self, prompt, history):
        """
        Gets a response from the LLM, maintaining conversation history.
        """
        history.append({"role": "user", "content": prompt})
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "assistant", "content": ""}]
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
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response_tokens = res[0][input_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": response_text})
        if len(history) > self.max_history * 2:
            history = history[-(self.max_history * 2):]
        return response_text, history