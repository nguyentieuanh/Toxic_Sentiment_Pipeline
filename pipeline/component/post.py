from pipeline.base_obj import PostProcessComponent
from pipeline.data_obj.datapoint import DataPoint
from typing import Optional
import numpy as np


class Mapper(PostProcessComponent):
    def serve(self, dp: DataPoint):
        result = dp.result
        mapper = {
            0: "Clean",
            1: "Chế nhạo, lăng mạ, công kích",
            2: "Tự hại, tự tử",
            3: "Quấy rối tình dục",
            4: "Lăng mạ giới tính"
        }
        dp.sentiment = mapper[int(result)]
        return dp