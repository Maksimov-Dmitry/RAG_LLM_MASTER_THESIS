import hydra
from omegaconf import DictConfig
import boto3
import json

from src.data.prepare_training_dataset import prepare_retriever_datasets
from src.entities.train_retriever_params import TrainRetrieverParams, read_train_retriever_params
import os
import logging
from sagemaker.remote_function import RemoteExecutor
import sagemaker
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
from aim import Run
from dataclasses import asdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _download_s3_folder(bucket_name, s3_folder, local_dir):
    """
    Download an entire folder from an S3 bucket to a local directory.

    :param bucket_name: Name of the S3 bucket.
    :param s3_folder: Folder path in the S3 bucket.
    :param local_dir: Local directory to which the folder will be downloaded.
    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):
            local_file_path = os.path.join(local_dir, os.path.relpath(obj['Key'], s3_folder))
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            s3_client.download_file(bucket_name, obj['Key'], local_file_path)


def _upload_directory_to_s3(bucket_name, s3_folder, local_directory):
    """
    Upload a directory to an S3 bucket
    """
    s3_client = boto3.client('s3')
    for root, _, files in os.walk(local_directory):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full S3 path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_folder, relative_path)

            # upload the file
            s3_client.upload_file(local_path, bucket_name, s3_path)


def _get_json_from_s3(bucket_name, key):
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket_name, key)
    file_content = obj.get()['Body'].read().decode('utf-8')
    return json.loads(file_content)


def train_retriever(params: TrainRetrieverParams):
    logger.info(f"start train retriever with params {params}")
    json_train = _get_json_from_s3(params.bucket_name, f'{params.output_data_s3}/train.json')
    json_val = _get_json_from_s3(params.bucket_name, f'{params.output_data_s3}/val.json')

    train = EmbeddingQAFinetuneDataset(queries=json_train['queries'], corpus=json_train['corpus'], relevant_docs=json_train['relevant_docs'])
    val = EmbeddingQAFinetuneDataset(queries=json_val['queries'], corpus=json_val['corpus'], relevant_docs=json_val['relevant_docs'])
    logger.info(f"train size: {len(train.queries)}")
    logger.info(f"val size: {len(val.queries)}")

    finetune_engine = SentenceTransformersFinetuneEngine(
        train,
        model_id=params.model_input_name,
        model_output_path=params.model_output_name,
        val_dataset=val,
        epochs=params.epochs,
        evaluation_steps=len(train.queries) // params.batch_size // 2 + 1,
        batch_size=params.batch_size,
        show_progress_bar=False,
    )

    finetune_engine.evaluator = InformationRetrievalEvaluator(
        finetune_engine.dataset.queries,
        finetune_engine.dataset.corpus,
        finetune_engine.dataset.relevant_docs,
        mrr_at_k=[1, params.top_k],
        ndcg_at_k=[1, params.top_k],
        accuracy_at_k=[1, params.top_k],
        precision_recall_at_k=[1, params.top_k],
        map_at_k=[1, params.top_k, 100],
        main_score_function='cos_sim'
    )

    finetune_engine.finetune()
    _upload_directory_to_s3(params.bucket_name, params.model_output_name, params.model_output_name)


@hydra.main(version_base=None, config_path="../configs", config_name="train_retriever_config")
def train_retriever_command(cfg: DictConfig):
    params = read_train_retriever_params(cfg)

    run = Run(experiment='train_retriever', capture_terminal_logs=False)
    run['hparams'] = asdict(params)
    prepare_retriever_datasets(params)
    sm_session = sagemaker.Session(
        boto_session=boto3.session.Session(region_name="eu-central-1"),
        default_bucket='tcr-algotrading'
    )
    with RemoteExecutor(
        sagemaker_session=sm_session,
        instance_type=params.sagemaker_instance,
        instance_count=1,
        job_name_prefix='retriever-training',
        dependencies='sagemaker_requirements.txt',
        include_local_workdir=True,
        role='arn:aws:iam::185705041424:role/SageMakerRole',
    ) as e:
        future = e.submit(train_retriever, params)
        print(future.result())
    _download_s3_folder(params.bucket_name, params.model_output_name, params.model_output_local_path)
    df = pd.read_csv(f"{params.model_output_local_path}/eval/Information-Retrieval_evaluation_results.csv")
    for _, row in df.filter(like='cos_sim-').iterrows():
        for k, v in row.items():
            run.track(v, name=k.replace('cos_sim-', ''))


if __name__ == "__main__":
    train_retriever_command()
