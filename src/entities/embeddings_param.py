from dataclasses import dataclass
from typing import List


@dataclass()
class TextEmbeddings:
    use_tables: bool
    max_tokens: int
    prefix: str


@dataclass()
class CLIPEmbeddings:
    font_size: int
    x_y_docname_loc: List[int]
    images_folder: str
