import streamlit as st
from utils.peek import ChromaPeek
from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
load_dotenv()

PATH = "data/db"
peeker = ChromaPeek(PATH)
open_ai = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

st.set_page_config(page_title="RAG System",
                   layout="wide",
                   )

st.title("RAG System")

generator = st.sidebar.selectbox('generator model',
                                 options=['gpt-3.5-turbo-1106',
                                          'gpt-4-1106-preview',
                                          'mixtral',
                                          'mistral',
                                          'mistral_finetuned',
                                          'mistral_finetuned_completion_only',
                                          'llama',
                                          ],
                                 )
if generator == 'mixtral':
    model_path = 'models/mixtral/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf'
elif generator == 'mistral':
    model_path = 'models/mistral/leo-mistral-hessianai-7b-chat.Q4_K_M.gguf'
elif generator == 'llama':
    model_path = 'models/llama/leo-hessianai-70b-chat.Q4_K_M.gguf'
elif 'finetuned' in generator:
    model_path = f'models/{generator}/{generator}.gguf'

if 'gpt' not in generator:
    model = Llama(model_path=model_path,
                  n_ctx=3000,
                  use_mlock=False)


def get_prompt(question, context, generator):
    if 'gpt' in generator:
        return [
            {"role": "system", "content": "Sie sind ein hilfreicher Assistent."},
            {
                "role": "user",
                "content": f"""Beantworten Sie die folgende Frage basierend auf dem Kontext.
        Frage: {question}\n\n
        Kontext: {context}\n\n
        Antwort:\n""",
            },
        ]
    elif 'mixtral' in generator:
        return f'<s> [INST] Beantworten Sie die folgende Frage basierend auf dem Kontext.\nKontext: {context}\nFrage: {question}. [/INST]'
    else:
        system_prompt = "Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer.\nDer Assistent gibt ausführliche, hilfreiche und ehrliche Antworten."
        prompt_format = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nBeantworten Sie die folgende Frage basierend auf dem Kontext.\nKontext: {context}\nFrage: {question}<|im_end|>\n<|im_start|>assistant\n"
        return prompt_format


@st.cache_data()
def get_collections():
    return peeker.get_collections()


@st.cache_data()
def generate_answer(query, context, generator, temperature, top_p, max_tokens):
    if 'gpt' in generator:
        messages = get_prompt(query, context, generator)
        response = open_ai.chat.completions.create(
            model=generator,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content, messages[1]['content']
    else:
        message = get_prompt(query, context, generator)
        output = model(prompt=message, max_tokens=max_tokens, temperature=temperature, top_k=50, top_p=top_p)
        return output['choices'][0]['text'], message


embeddings_collection = st.sidebar.selectbox('embeddings collection',
                                             options=get_collections()
                                             )
top_k = st.sidebar.number_input('top_k', min_value=0, max_value=10, value=3)
temperature = st.sidebar.slider('temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.01)
top_p = st.sidebar.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.01)
max_tokens = st.sidebar.number_input('max_tokens', min_value=1, max_value=1000, value=250, step=50)
use_context = st.sidebar.checkbox('show context', value=False)
query = st.text_input("Enter Query", placeholder="query")
if st.button('Answer') and query:
    relevant_docs = peeker.query(query, embeddings_collection, top_k, dataframe=False)
    context = '\n\n'
    for i, (metadata, document) in enumerate(zip(relevant_docs['metadatas'], relevant_docs['documents']), 1):
        context += f"Dokument {i}: {metadata['document_name']}.\n{document}\n\n"
    response, question = generate_answer(query, context, generator, temperature, top_p, max_tokens)
    st.write(response)
    st.dataframe(pd.DataFrame(relevant_docs), use_container_width=True)
    if use_context:
        st.write(question)
