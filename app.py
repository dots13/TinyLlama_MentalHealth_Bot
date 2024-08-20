import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Model and Tokenizer Configuration
MODEL_ID_MIND = "dots-13/llama-3-8B-chat-mindguardian-v1"
TOKENIZER_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID_MIND)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit Interface
st.title("Hugging Face Chatbot")

def formatted_prompt(user_input: str) -> str:
    system_message = (
        "You are a helpful and supportive assistant. "
        "You should respond as 'Assistant' and avoid repeating or imitating the user. "
        "Your responses should be original, supportive, and helpful."
    )
    return f"{system_message}\n\nUser: {user_input}\nAssistant:"

user_input = st.text_input("You:", "")

# Initialize Text Generation Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else -1  # Set to -1 if using CPU
)

# Generate and Display Response
if user_input:
    prompt = formatted_prompt(user_input)
    results = pipe(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_p=0.75,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256
    )
    generated_text = results[0]['generated_text']
    input_len = len(prompt)
    st.text_area("Bot:", generated_text[input_len:], height=200)