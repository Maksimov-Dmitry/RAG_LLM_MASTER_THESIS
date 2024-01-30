import streamlit as st
from utils.peek import ChromaPeek

st.set_page_config(page_title="Embeddings",
                   layout="wide",
                   )

st.title("Chroma")

# get uri of the persist directory
PATH = "data/db"

st.divider()

# load collections
peeker = ChromaPeek(PATH)


@st.cache_data()
def get_collections():
    return peeker.get_collections()


@st.cache_data()
def get_data(collection_name, dataframe=False):
    return peeker.get_collection_data(collection_name, dataframe)


@st.cache_data()
def get_response(query_str, collection_name, k=5, dataframe=False):
    return peeker.query(query_str, collection_name, k, dataframe)


col1, col2 = st.columns([1, 3])
with col1:
    collection_selected = st.radio("select collection to view",
                                   options=get_collections(),
                                   index=0,
                                   )

with col2:
    df = get_data(collection_selected, dataframe=True)

    st.markdown(f"<b>Data in </b>*{collection_selected}*", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=500)

st.divider()

k = st.slider("Select number of similar embeddings", min_value=1, max_value=10, value=3)
query = st.text_input(f"Enter Query to get {k} similar embeddings", placeholder=f"get {k} similar embeddings")
if query:
    result_df = get_response(query, collection_selected, k, dataframe=True)

    st.dataframe(result_df, use_container_width=True)

if collection_selected == 'images_clip':
    image_path = st.text_input("Enter image path to show")
    if image_path:
        st.image(image_path, use_column_width=True)
