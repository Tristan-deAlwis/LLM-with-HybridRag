from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LlamaModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, query):
        input_ids = self.tokenizer.encode(query, return_tensors="pt")
        output_ids = self.model.generate(input_ids, max_length=100)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)