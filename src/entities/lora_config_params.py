from dataclasses import dataclass
from typing import List


@dataclass()
class LoraConfigParams:
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
