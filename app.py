import streamlit as st
import torch
import json
import time
from huggingface_hub import hf_hub_download
from model_def import GPTModel, generate
import tiktoken

REPO_ID = "kartiksethi000/my-gpt-model" 
WEIGHTS_FILE = "model_weights.pt"
CONFIG_FILE = "config.json"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILE)
config_path = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILE)

with open(config_path, "r") as f:
    config = json.load(f)

model = GPTModel(config)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")


st.title("LLM Chat")
st.write("Enter a prompt and get a response from the trained LLM model.")

user_input = st.text_area("Enter your prompt:", "")

if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        inputs = torch.tensor([tokenizer.encode(user_input)]).to("cpu")

        tokens = generate(
            model=model,
            idx=inputs,
            max_new_tokens=400,
            context_size=config["context_length"],
            eos_id=50256
        )

        full_text = tokenizer.decode(tokens[0].tolist())
        if "Response:" in full_text:
            response_text = full_text.split("Response:")[-1].strip()
            if "Instruction:" in response_text:
                response_text = response_text.split("Instruction:")[0].strip()
        else:
            response_text = full_text

        st.markdown("### 📝 Response:")


        placeholder = st.empty()
        stream_text = ""
        for token in tokenizer.encode(response_text):
            stream_text += tokenizer.decode([token])
            placeholder.markdown(stream_text)
            time.sleep(0.01)
