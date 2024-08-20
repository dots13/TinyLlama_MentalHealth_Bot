import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

@st.cache_resource
def load_model():
    model_id = "your-model-id"  # Replace with your model's ID on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("dots-13/llama-3-8B-chat-mindguardian-v1")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit Interface
st.title("Hugging Face Chatbot")

user_input = st.text_input("You:", "")

def formatted_prompt(user_input: str) -> str:
    system_message = (
        "You are a helpful and supportive assistant. "
        "You should respond as 'Assistant' and avoid repeating or imitating the user. "
        "Your responses should be original, supportive, and helpful."
    )
    return f"{system_message}\n\nUser: {user_input}\nAssistant:"

# Initialize Text Generation Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=-1
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
