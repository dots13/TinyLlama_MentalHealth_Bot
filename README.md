# TinyLlama Mental Health Chatbot
This repository hosts a chatbot application built with Streamlit, utilizing a fine-tuned TinyLlama model from Hugging Face to provide empathetic and supportive responses for mental health conversations.

- Live Demo: TinyLlama Mental [Health Bot](https://tinyllama-mentalhealthbot.streamlit.app/)
- Fine-tuning Notebook: Available on [Kaggle](https://www.kaggle.com/code/nastyadots/tinyllama-v1)
## Project Overview
This chatbot leverages:

- Streamlit: For the interactive web interface.
- Hugging Face Transformers: To manage the TinyLlama model and text generation pipeline.
- Fine-tuning: Conducted in a Kaggle notebook to enhance conversational support.
## Installation
1. Clone this repository.
2. Install required libraries:
```
pip install streamlit transformers torch
```
## Usage
To start the chatbot locally:
```
streamlit run app.py
```
## Model Information
- Tokenizer: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Fine-tuned Model: dots-13/llama-3-8B-chat-mindguardian-v1
## Features
- User-Friendly Interface: Designed with Streamlit for ease of use.
- Guided Responses: Structured prompts to ensure empathetic, supportive answers.
## Future Work
Further customization of response styles.
Expanded deployment options.
