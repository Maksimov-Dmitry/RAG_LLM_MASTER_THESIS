from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class TrainRetrieverParams:
    model_input_name: str
    model_output_name: str
    model_output_local_path: str
    input_data_local: str
    output_data_s3: str
    batch_size: int
    epochs: int
    query_prefix: str
    doc_prefix: str
    chromadb_path: str
    collection_name: str
    top_k: int
    sagemaker_instance: str
    bucket_name: str


TrainRetrieverParamsSchema = class_schema(TrainRetrieverParams)


def read_train_retriever_params(cfg: DictConfig) -> TrainRetrieverParams:
    schema = TrainRetrieverParamsSchema()
    return schema.load(cfg)
