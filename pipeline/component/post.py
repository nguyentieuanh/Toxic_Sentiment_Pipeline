from pipeline.base_obj import PostProcessComponent
from pipeline.data_obj.datapoint import DataPoint
from typing import Optional
import numpy as np


class Mapper(PostProcessComponent):
    def serve(self, dp: DataPoint):
        # result = dp.result.detach().numpy()
        # number = np.argmax(result, axis=1)[0]
        # mapper = {
        #     0: "Clean",
        #     1: "Xúc phạm, lăng mạ",
        #     2: "Tự tử",
        #     3: "Quấy rối"
        # }
        # dp.sentiment = mapper[number]
        # return dp
        result = dp.result
        mapper = {
            0: "Clean",
            1: "Toxic"
        }
        dp.sentiment = mapper[int(result)]
        return dp