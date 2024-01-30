from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class PredictGeneratorParams:
    model: str
    use_context: bool
    input: str
    output: str
    top_p: float
    max_new_tokens: int
    temperature: float
    sagemaker_instance: str
    bucket_name: str
    evaluator_model: str
    use_flash_attention_2: bool


PredictGeneratorParamsSchema = class_schema(PredictGeneratorParams)


def read_predict_generator_params(cfg: DictConfig) -> PredictGeneratorParams:
    schema = PredictGeneratorParamsSchema()
    return schema.load(cfg)
