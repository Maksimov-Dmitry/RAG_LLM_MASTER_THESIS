from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from src.entities.lora_config_params import LoraConfigParams
from src.entities.training_arguments_params import TrainingArgumentsParams


@dataclass()
class TrainGeneratorParams:
    model_input_name: str
    model_output_name: str
    input_data: str
    lora_config: LoraConfigParams
    use_DataCollatorForCompletionOnlyLM: bool
    training_arguments: TrainingArgumentsParams
    max_seq_length: int
    bucket_name: str
    sagemaker_instance: str


TrainGeneratorParamsSchema = class_schema(TrainGeneratorParams)


def read_train_generator_params(cfg: DictConfig) -> TrainGeneratorParams:
    schema = TrainGeneratorParamsSchema()
    return schema.load(cfg)
