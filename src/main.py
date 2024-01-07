import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    print('start')

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

    prompt = """Please write a quick sort code in Python.
    Answer: """

    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            temperature=0.2,
            do_sample=True,
            max_new_tokens=256
        )
    

    output = tokenizer.decode(output_ids[0][token_ids.size(1) :])

    print(output)