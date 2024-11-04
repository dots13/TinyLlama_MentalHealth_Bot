import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

hf_token = st.secrets["huggingface"]["token"]

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained("dots-13/llama-3-8B-chat-mindguardian-v1")
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

if model and tokenizer:
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
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Set to -1 to use CPU
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

    except Exception as e:
        st.error(f"Error during generation: {e}")
else:
    st.error("Model or tokenizer failed to load.")
