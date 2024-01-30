from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import Optional


@dataclass()
class PredictAndScoreRetrieverParams:
    input: str
    output: str
    model_name: str
    batch_size: int
    prefix: str
    max_length: int
    chromadb_path: str
    collection_name: str
    top_k: int
    evaluator_model: Optional[str]
    bucket_name: str
    output_data_s3: str


PredictAndScoreRetrieverParamsSchema = class_schema(PredictAndScoreRetrieverParams)


def read_predict_and_score_retriever_params(cfg: DictConfig) -> PredictAndScoreRetrieverParams:
    schema = PredictAndScoreRetrieverParamsSchema()
    return schema.load(cfg)
