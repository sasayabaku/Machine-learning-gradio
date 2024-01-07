FROM huggingface/transformers-pytorch-gpu:4.35.2

RUN pip install accelerate einops gradio

COPY ./src /workspace
WORKDIR /workspace

RUN python3 setup.py