import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from datasets import load_from_disk

from src.models.retriever import Model, ChromaDB
from src.evaluation.evaluation import calculate_DocHitRate, calculate_HitRate, retriever_llm_evaluate
from src.entities.predict_and_score_retriever_params import PredictAndScoreRetrieverParams, read_predict_and_score_retriever_params
from lightning.pytorch import Trainer
import os
from collections import defaultdict
from dotenv import load_dotenv
import logging
from aim import Run
from dataclasses import asdict
import cohere
from datasets.filesystems import S3FileSystem


# Set the logging level to ERROR to suppress INFO and DEBUG messages
httpx_logger = logging.getLogger('httpx')
openai_logger = logging.getLogger('openai._base_client')

# Set their level to ERROR or CRITICAL to suppress INFO logs
httpx_logger.setLevel(logging.ERROR)
openai_logger.setLevel(logging.ERROR)


def predict_and_score_pipeline(params: PredictAndScoreRetrieverParams):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    db = ChromaDB(params)
    run = Run(experiment='predict_retriever', capture_terminal_logs=False)
    run['hparams'] = asdict(params)
    logging.info(f"start predict and score pipeline with params {params}")
    dataset = load_from_disk(params.input)
    if params.model_name == 'embed-multilingual-v3.0':
        co = cohere.Client(os.getenv('COHERE_API_KEY'))
    else:
        model = Model(params)
        dataset_proc = dataset.map(model.preprocess_data, batched=True)
        dataset_proc.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        trainer = Trainer()
    for stage in dataset:
        if params.model_name == 'embed-multilingual-v3.0':
            embeddings = co.embed(dataset[stage]['question'], input_type="search_query", model="embed-multilingual-v3.0").embeddings
        else:
            dataloader = DataLoader(dataset_proc[stage], batch_size=params.batch_size, shuffle=False, num_workers=5, persistent_workers=True)
            result = trainer.predict(model, dataloader)
        contexts = []
        distances = []
        pages = []
        documents = []
        scores = defaultdict(list)
        if params.model_name == 'embed-multilingual-v3.0':
            res = db.retrieve(embeddings)
            contexts = res['documents']
            distances = res['distances']
            for i in res['metadatas']:
                pages.append([j['page'] for j in i])
                documents.append([j['document_name'] for j in i])
        else:
            for embeddings in result:
                res = db.retrieve(embeddings.tolist())
                contexts.extend(res['documents'])
                distances.extend(res['distances'])
                for i in res['metadatas']:
                    pages.append([j['page'] for j in i])
                    documents.append([j['document_name'] for j in i])
        dataset[stage] = dataset[stage].add_column('contexts', contexts)
        dataset[stage] = dataset[stage].add_column('distances', distances)
        dataset[stage] = dataset[stage].add_column('contexts_pages', pages)
        dataset[stage] = dataset[stage].add_column('contexts_documents', documents)
        hit_rate = calculate_HitRate(dataset[stage])
        doc_hit_rate = calculate_DocHitRate(dataset[stage])
        run.track(hit_rate, name="HitRate", context={'subset': stage})
        run.track(doc_hit_rate, name="DocHitRate", context={'subset': stage})
        if params.evaluator_model:
            scores = retriever_llm_evaluate(dataset[stage], params.evaluator_model)
            for key in scores.keys():
                dataset[stage] = dataset[stage].add_column(key, scores[key])
                logging.info(f"{key} for {stage} is {sum(scores[key])/len(scores[key]):.3f}")
    dataset.save_to_disk(params.output)
    s3 = S3FileSystem()
    dataset.save_to_disk(f's3://{params.bucket_name}/{params.output_data_s3}', fs=s3)


@hydra.main(version_base=None, config_path="../configs", config_name="predict_retriever_config")
def predict_and_score_pipeline_command(cfg: DictConfig):
    params = read_predict_and_score_retriever_params(cfg)
    load_dotenv()
    predict_and_score_pipeline(params)


if __name__ == "__main__":
    predict_and_score_pipeline_command()
