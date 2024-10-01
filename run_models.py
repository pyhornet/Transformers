from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel

# Load the tokenizer
# Define the base model name (replace "name/model" with the actual model path)
model_name = "name/model"

# Set the compute data type to bfloat16 for efficient computation or float16
compute_dtype = getattr(torch, "bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the BitsAndBytes settings for loading the model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable loading the model in 4-bit precision
    bnb_4bit_quant_type="nf4",  # Specify the quantization type
    bnb_4bit_compute_dtype=compute_dtype,  # Set the compute data type
    bnb_4bit_use_double_quant=True,  # Enable double quantization for better performance
)

# Load the pre-trained causal language model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map={"": 0},  quantization_config=bnb_config # Map the model to the first GPU device, in this case it will be device 0. It will also apply the quantization configuration from above.
)

# Load the LoRA adapter
lora_adapter_name = "./my_adapter"  # Replace with the actual path or identifier for the LoRA adapter
model = PeftModel.from_pretrained(model, lora_adapter_name)

# Set the maximum sequence length
max_length = 100

# Example usage
input_text = "User: Tell more about you.\n\nAssistant:"
inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)

# Move inputs to the same device as the model
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=max_length)

# Decode the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)