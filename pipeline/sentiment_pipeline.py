from pipeline.base_obj import BasePipeline, PipelineComponent, PreProcessComponent, PostProcessComponent
from pipeline.component.pre import LoadDataComponent
from pipeline.component.layout import SentimentAnalysis, BiToxicCmtAnalysis
from pipeline.component.post import Mapper
from typing import List


class SentimentPipeline(BasePipeline):
    def __init__(self,
                 pre_component: PreProcessComponent,
                 pipeline_component_list: List[PipelineComponent],
                 post_component: PostProcessComponent):
        self.pre_component = pre_component
        self.pipeline_component_list = pipeline_component_list
        self.post_component = post_component

    @classmethod
    def build(cls,
              pre_component: PreProcessComponent = None,
              d_bi_cls: PipelineComponent = None,
              d_sentiment: PipelineComponent = None,
              post_component: PostProcessComponent = None):
        if pre_component is None:
            pre_component = LoadDataComponent()

        components = []
        if d_bi_cls is None:
            d_bi_cls = BiToxicCmtAnalysis()
        components.append(d_bi_cls)

        if d_sentiment is None:
            d_sentiment = SentimentAnalysis()

        components.append(d_sentiment)

        if post_component is None:
            post_component = Mapper()

        return cls(
            pre_component,
            components,
            post_component
        )

    def analyze(self, text: str):
        dp = self.pre_component.serve(text)
        for c in self.pipeline_component_list:
            dp = c.serve(dp)
        dp = self.post_component.serve(dp)
        return dp
