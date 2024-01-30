from dataclasses import dataclass


@dataclass()
class TrainingArgumentsParams:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    optim: str
    bf16: bool
    learning_rate: float
    warmup_ratio: float
    num_train_epochs: int
    lr_scheduler_type: str
    run_name: str
