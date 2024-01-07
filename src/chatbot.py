import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import gradio as gr

message_history = []

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


def chat(user_msg):
    global message_history, tokenizer, model

    print(user_msg)

    message_history.append({
        "role": "user",
        "content": user_msg
    })


    prompt = """{}

    Answer: """.format(user_msg)

    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            temperature=0.2,
            do_sample=True,
            max_new_tokens=128
        )

    assistant_msg = tokenizer.decode(output_ids[0][token_ids.size(1) :])

    print(assistant_msg)

    message_history.append({
        "role": "assistant",
        "content": assistant_msg
    })

    return [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(0, len(message_history)-1, 2)]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    input = gr.Textbox(show_label=False, placeholder="メッセージを入力してね")
    input.submit(fn=chat, inputs=input, outputs=chatbot)
    input.submit(fn=lambda: "", inputs=None, outputs=input)

demo.launch(share=False, server_name="0.0.0.0")