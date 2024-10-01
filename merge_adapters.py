from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel

# Define the base model name (replace "name/model" with the actual model path)
base_model = "name/model"

# Set the compute data type to bfloat16 for efficient computation or float16
compute_dtype = getattr(torch, "bfloat16")

# Configure the BitsAndBytes settings for loading the model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable loading the model in 4-bit precision
    bnb_4bit_quant_type="nf4",  # Specify the quantization type
    bnb_4bit_compute_dtype=compute_dtype,  # Set the compute data type
    bnb_4bit_use_double_quant=True,  # Enable double quantization for better performance
)

# Load the pre-trained causal language model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    device_map={"": 0},  # Map the model to the first GPU device, in this case it will be device 0
    quantization_config=bnb_config  # Apply the quantization configuration from above
)

# Load the tokenizer for the specified model, using a fast tokenizer if available
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

# Define a function to generate text based on a given prompt
def generate(prompt):
    # Tokenize the input prompt and convert it to tensor format
    tokenized_input = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized_input["input_ids"].cuda()  # Move input IDs to GPU

    # Generate text using the model with specified parameters
    generation_output = model.generate(
        input_ids=input_ids,
        num_beams=1,  # Use beam search with 1 beam (greedy search)
        return_dict_in_generate=True,  # Return output as a dictionary
        output_scores=True,  # Include scores in the output
        max_new_tokens=130  # Limit the number of new tokens generated
    )

    # Decode and print each generated sequence
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq, skip_special_tokens=True)  # Decode the sequence
        print(output.strip())  # Print the output without special tokens

# Load a pre-trained adapter model and move it to CPU
model = PeftModel.from_pretrained(model, "adapter1", adapter_name="adapter1").to('cpu')

# Load another adapter into the model
model.load_adapter("adapter2", adapter_name="adapter2")

# Combine the two adapters with specified weights and save the combined adapter as "my_merged_model"
model.add_weighted_adapter(["adapter1", "adapter2"], [1.0, 1.0], combination_type="cat", adapter_name="my_merged_model") 
# Refer to https://huggingface.co/docs/peft/package_reference/lora for more info

# Delete the individual adapters from the model
model.delete_adapter("adapter1")
model.delete_adapter("adapter2")

# Save the combined adapter to a specified folder
model.save_pretrained("./adapter_save_folder")
