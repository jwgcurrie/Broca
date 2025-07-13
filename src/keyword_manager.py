
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class KeywordManager:
    """
    Manages a dynamic list of keywords extracted from conversation turns.
    """
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-360M-Instruct", max_keywords=10, verbose=False):
        self.verbose = verbose
        if self.verbose: print("Initialising keyword extraction model on CPU...")
        self.device = "cpu"
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.active_keywords = []
        self.max_keywords = max_keywords

        if self.verbose: print(f"✅ Keyword extraction model initialised on {self.device}.")

    def to_gpu(self):
        if torch.cuda.is_available() and self.device == "cpu":
            if self.verbose: print("Moving keyword extraction model to GPU...")
            self.device = "cuda:0"
            self.model.to(self.device)
            if self.verbose: print("✅ Keyword extraction model on GPU.")

    def to_cpu(self):
        if self.device != "cpu":
            if self.verbose: print("Moving keyword extraction model to CPU...")
            self.device = "cpu"
            self.model.to(self.device)
            if self.verbose: print("✅ Keyword extraction model on CPU.")

    def _extract_keywords_from_text(self, text):
        prompt = f"List up to 5 distinct, single-word key nouns or concepts from the following text. Separate them with commas and do not include any other text or punctuation. Text: {text}\nKeywords:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            res = self.model.generate(
                **inputs,
                max_new_tokens=20, # Very concise output
                do_sample=True,
                temperature=0.7, # Slightly higher temperature for better formatting adherence
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        extracted_text = self.tokenizer.decode(res[0][input_length:], skip_special_tokens=True).strip()
        
        # Robust parsing: split by comma, clean up, and ensure single words
        keywords = [k.strip().lower() for k in extracted_text.split(',') if k.strip() and ' ' not in k.strip()]
        return keywords

    def update_keywords(self, new_text):
        extracted = self._extract_keywords_from_text(new_text)
        
        for keyword in extracted:
            if keyword not in self.active_keywords:
                if len(self.active_keywords) >= self.max_keywords:
                    self.active_keywords.pop(0) # Remove oldest keyword
                self.active_keywords.append(keyword)
        
        if self.verbose: print(f"Active Keywords: {self.active_keywords}")

    def get_keyword_string(self):
        if not self.active_keywords:
            return ""
        return "Current discussion points: " + ", ".join(self.active_keywords) + "."
