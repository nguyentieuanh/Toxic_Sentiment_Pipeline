from dataclasses import field, dataclass
from typing import Optional
import numpy.typing as npt

@dataclass
class DataPoint:
    text_array: npt.NDArray
    result: npt.NDArray
    len_text: int = field(default=0)
    text: Optional[str] = field(default=None)
    quality: int = field(default=0)
    vocab_size: int = field(default=0)
    sentiment: str = field(default=None)

    def to_json(self):
        return {
            "len_text": self.len_text,
            "text": self.text,
            "quality": self.quality,
            "result": self.result,
            "sentiment": self.sentiment
        }
