from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from src.entities.embeddings_param import TextEmbeddings, CLIPEmbeddings


@dataclass()
class VectorDBParams:
    embedding_model: str
    input_path: str
    db_path: str
    collection_name: str
    distance_metric: str
    embeddings: str
    text_embeddings: TextEmbeddings
    clip_embeddings: CLIPEmbeddings


VectorDBParamsSchema = class_schema(VectorDBParams)


def read_vector_db_params(cfg: DictConfig) -> VectorDBParams:
    schema = VectorDBParamsSchema()
    return schema.load(cfg)
