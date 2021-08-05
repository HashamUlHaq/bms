from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import math
import pandas as pd


class ResolverPipeline():
    
    def __init__(self, spark):
        """
            spark: spark context to create a pipeline
        """
        
        documenter = DocumentAssembler().setInputCol('text').setOutputCol('document')

        sbert = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
            .setInputCols("document").setOutputCol("sbert_embeddings_sbert")

        rxnorm = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_dispo","en","clinical/models")\
            .setInputCols(f"document","sbert_embeddings_sbert").setOutputCol("resolution_rxnorm")#.setNeighbours(3)

        pipeline = Pipeline().setStages([
            documenter, sbert, rxnorm
        ])

        self.p_model = pipeline.fit(spark.createDataFrame([("",)]).toDF("text"))
        self.l_model = LightPipeline(self.p_model)
        
    def resolve(self, text_list):
        """
            text_list: list of text/names of drugs to run through the pipeline
        """
        return self.l_model.fullAnnotate(text_list)
    