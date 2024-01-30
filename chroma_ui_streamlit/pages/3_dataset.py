import streamlit as st
import os

st.set_page_config(page_title="RAG System",
                   layout="wide",
                   )

st.title("Dataset")

PATH = 'data/images'
DATASET = 'data/qa'
qa_version = st.sidebar.selectbox('dataset version', options=[directory for directory in os.listdir(DATASET)
                                                 if os.path.isdir(f'{DATASET}/{directory}')])
documents = [directory for directory in os.listdir(PATH) if os.path.isdir(f'{PATH}/{directory}')]
document = st.selectbox('document', options=documents)

pages = [page for page in os.listdir(f'{PATH}/{document}') if page.endswith('.png')]
page = st.selectbox('page', options=pages)

with open(f'{DATASET}/{qa_version}/{document}/{page.replace(".png", ".txt")}', 'r') as f:
    text = f.read()
    if not text.startswith("```json"):
        text = "```json\n" + text
    if not text.endswith("```"):
        text += "```"
    st.write(text)

st.image(f'{PATH}/{document}/{page}')
