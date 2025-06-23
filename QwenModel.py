import os, sys
import torch
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
from langchain.llms import HuggingFacePipeline

# Determine if CUDA (GPU) is available; if not, default to CPU
class QwenModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(torch.cuda.get_device_name(0))

        # Configuration for loading the model in 4-bit precision to save memory
        self.bitsquant_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Load model in 4-bit precision
            bnb_4bit_use_double_quant=True,       # Use double quantization for better compression
            bnb_4bit_quant_type="nf4",            # Specify quantization type as NF4 (a specific format for quantization)
            bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 as the compute data type for operations
        )

        # Alibaba Cloud's Qwen 2.5 7B model
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        # Alternative sharded model if needed
        sharded_model = "Qwen/Qwen2.5-7B-Instruct" 

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.mmodel = AutoModelForCausalLM.from_pretrained(
                sharded_model, 
                trust_remote_code=True, 
                quantization_config=self.bitsquant_config, 
                device_map="auto",
        )

        self.text_gen_pipeline = pipeline(
            model=self.mmodel,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.01, 
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.eos_token_id, 
            repetition_penalty=1.1, 
            return_full_text=False,
            max_new_tokens=1024
        )
        self.model = HuggingFacePipeline(pipeline=self.text_gen_pipeline)
    def generate_text(self, prompt):
        return self.model.invoke(prompt)   

if __name__ == "__main__":
    model = QwenModel()
    print(model.generate_text("<s>[INST]Who is Alibaba Cloud and what is Qwen? Please write a cohesive, apt answer that would be deemed college exam worthy.[/INST]"))
