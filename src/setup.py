import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Download PreTrained Model")
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

