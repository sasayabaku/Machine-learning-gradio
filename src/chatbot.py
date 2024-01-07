import gradio as gr

message_history = []

def chat(user_msg):
    global message_history

    print(user_msg)

    message_history.append({
        "role": "user",
        "content": user_msg
    })

    
    assistant_msg = "Chatbot Response"

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

demo.launch()