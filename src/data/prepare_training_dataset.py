from src.entities.train_retriever_params import TrainRetrieverParams
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from datasets import load_from_disk
import chromadb
import boto3
import tempfile
import json


def _get_doc_ids(example, collection):
    docs = collection.get(where={
        '$and': [
            {'page': example['page']},
            {'document_name': example['document']}
            ]
            })
    return docs['ids']


def filter(queries, relevant_docs):
    t_query = list(relevant_docs.keys()).copy()
    t_docs = list(relevant_docs.values()).copy()
    for query, docs in zip(t_query, t_docs):
        if len(docs) == 0:
            queries.pop(query)
            relevant_docs.pop(query)


def prepare_retriever_datasets(params: TrainRetrieverParams):
    dataset = load_from_disk(params.input_data_local)
    client = chromadb.PersistentClient(params.chromadb_path)
    collection = client.get_collection(params.collection_name)
    docs = collection.get()
    corpus = {k: params.doc_prefix + doc_name['document_name'] + '\n' + v for k, v, doc_name in zip(docs['ids'], docs['documents'], docs['metadatas'])}
    s3_client = boto3.client('s3')
    for stage in ['train', 'val']:
        queries = {str(i): params.query_prefix + dataset[stage][i]['question'] for i in range(len(dataset[stage]))}
        relevant_docs = {
            str(i): _get_doc_ids(dataset[stage][i], collection) for i in range(len(dataset[stage]))
        }
        filter(queries, relevant_docs)
        dataset_stage = EmbeddingQAFinetuneDataset(queries=queries, corpus=corpus, relevant_docs=relevant_docs)
        with tempfile.NamedTemporaryFile(mode='w') as tmp_file:
            dataset_stage.save_json(tmp_file.name)
            s3_client.upload_file(tmp_file.name, params.bucket_name, f'{params.output_data_s3}/{stage}.json')
