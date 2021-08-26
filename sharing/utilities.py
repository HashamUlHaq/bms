from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl
import sparknlp
import pyspark
import pyspark.sql.functions as F
import pandas as pd
import pickle
import os
import json
import time
import numpy as np
import math

import re 

from collections import OrderedDict


def build_ner_pl(tptms, prefix=""):
    tms = []
    for t, (n, wl) in tptms.items():
        lt = n.lower()
        out = prefix+lt
        ner = MedicalNerModel.pretrained(t,"en","clinical/models")\
                            .setInputCols("sentence","token","embs")\
                            .setOutputCol(out)
        tms.append(ner)
        conv = NerConverterInternal().setInputCols("sentence","token",out)\
                                .setOutputCol(out+"_chunk")\
                                .setPreservePosition(False)
        if wl:
            conv.setWhiteList(wl)
        tms.append(conv)
    return tms

def merging_logic(fields):
    active = fields
    rounds = math.ceil(math.log(len(fields))/math.log(2))
    ret = []
    for r in range(rounds):
        if(len(active) % 2 == 1):
            active.append("seed")
        new_active = []
        for i in range(0, len(active), 2):
            pair = active[i:i+2]
            merged = "_".join(pair) if "seed" not in pair else pair[0]
            new_active.append((pair[0], pair[1], merged))
        active = [na[-1] for na in new_active]
        ret.append([na for na in new_active if "seed" not in na])
    return ret

def build_merging_pl(tptms, prefix=""):
    tms, logic_merge, last_tm = [], None, None
    if len(tptms) > 2:
        logic_merge = merging_logic(tptms)
        for fix in logic_merge: 
            for com in fix:
                cm = ChunkMergeApproach()\
                        .setInputCols(prefix+com[0],prefix+com[1])\
                        .setOutputCol(prefix+com[2])
                tms.append(cm)
            if fix==logic_merge[-1]:
                last_tm = prefix+fix[-1][2]
    return tms, logic_merge, last_tm   

def build_pipeline_ps(spark,regex_file_path="./patient_segment_module/base_regex.csv"):
    # Preprocessing pipeline
    da = DocumentAssembler()\
            .setInputCol("text")\
            .setOutputCol("document")
    sd = SentenceDetector()\
            .setInputCols("document")\
            .setOutputCol("sentence")\
            .setUseCustomBoundsOnly(True)\
            .setCustomBounds(["\n"])
    tk = Tokenizer()\
            .setInputCols("sentence")\
            .setOutputCol("token") #.setSplitPattern("( |/|&|-")
    emb = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
            .setOutputCol("embs")

    ners_to_merge = OrderedDict({
        "ner_diseases":("disease",[]),
        "ner_clinical":("base",["TEST"]), 
    #     "ner_ade_clinical":"ade", 
    #     "ner_events_clinical":"events",
    #     "ner_cellular":("cell",[]),
    #     "ner_bionlp":("bio",["Gene_or_gene_product"]), 
    #     "ner_genetic_variants":("gene",[]), 
    #     "ner_jsl":("jsl",["Test_Result","Gene_or_gene_product","Symptom"]), 
    #     "ner_cancer_genetics":("gene_cancer",[]),
    #     "ner_posology":("posology",[]), 
    #     "ner_human_phenotype_gene_clinical":("hpg",[]),
    #     "ner_anatomy":"anatomy", 
    })
    ner_pl = build_ner_pl(ners_to_merge)
    cms, logic_merge, last_tm = build_merging_pl([k[0].lower()+"_chunk" for k in ners_to_merge.values()])
    if cms:
        cms[-1].setOutputCol("all_chunk_init")
    else:
        ner_pl[-1].setOutputCol("all_chunk_init")
    rexm = RegexMatcher()\
                .setExternalRules(regex_file_path, ",", "TEXT")\
                .setStrategy("MATCH_ALL")\
                .setInputCols(["sentence"])\
                .setOutputCol("rex_chunk")
    cmrha = ChunkMergeApproach()\
                .setInputCols("rex_chunk","all_chunk_init")\
                .setOutputCol("all_chunk")\
                .setReplaceDictResource("./patient_segment_module/replace_dict.csv","TEXT", {"delimiter":","})\
                .setMergeOverlapping(False)
    ass_col = "disease_chunk"
    res_col = "disease_chunk"
    snomed_res_col = "disease_chunk"
    icdo_res_col = "rex_chunk"
    ass = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")\
                        .setInputCols(["sentence", ass_col, "embs"])\
                        .setOutputCol("assertion")
    c2d_snomed = Chunk2Doc()\
                    .setInputCols(snomed_res_col)\
                    .setOutputCol("sbert_doc")
    sbert_snomed = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
                                        .setInputCols("sbert_doc")\
                                        .setOutputCol("sbert_embeddings_sbert")
    snomed = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings","en","clinical/models")\
                    .setInputCols(f"sbert_doc","sbert_embeddings_sbert")\
                    .setOutputCol("resolution_snomed")\
                    .setNeighbours(3)
    pl = Pipeline().setStages([da,sd,tk,emb,rexm] + 
                              ner_pl + cms + 
                              [cmrha, ass, c2d_snomed, sbert_snomed, snomed])
    fdf = spark.createDataFrame([("",)]).toDF("text")
    plm = pl.fit(fdf)
    lplm= LightPipeline(plm)
    return plm, lplm
    

def process_lp_output(res,master_index_df,col_name='patient_segment_criteria_cleaned'):
    all_chunks=[]
    for r in res:
        all_chunk=[[a.begin,a.end, 
               [s.result for s in r['sentence'] if s.metadata['sentence']==a.metadata['sentence']][0],
               a.result,a.metadata['identifier'],'','',''] for a in r['all_chunk']]    
        all_chunk.extend([[a.begin,a.end, 
                        [s.result for s in r['sentence'] if s.metadata['sentence']==a.metadata['sentence']][0],
                         a.result,a.metadata['entity'],snomed.metadata['resolved_text'],
                         snomed.metadata['distance'],assertion.result] for a,snomed,assertion in zip(r['disease_chunk'],
                                                                                r['resolution_snomed'],
                                                                                r['assertion'])])  
        all_chunks.append(sorted(all_chunk,key=lambda x: x[0]))
    master_index_df_ps=master_index_df[['nct_id', col_name]]
    master_index_df_ps['ps_extraction']=all_chunks
    master_index_df_ps=master_index_df_ps.explode('ps_extraction')
    master_index_df_ps.reset_index(drop=True,inplace=True)
    master_index_df_ps=pd.concat([master_index_df_ps,
               pd.DataFrame(master_index_df_ps['ps_extraction'].fillna('[]').astype(str).apply(eval).tolist(), columns=[
        "begin","end","sentence","text","entity", "standard_text", "distance", "assertion"
    ])],axis=1)
    return master_index_df_ps.reset_index(drop=True)


# Function to add
def normalizer(standard_text, entity, text, stage_dic):
    if not standard_text:
        if entity in ['StageDesc', 'LineofTherapy']:
            return stage_dic.get(text.lower())
        if entity == 'StageRoman':
            return text.replace("stage ", "")
        if entity in ['Histology', 'DiseaseResectability', 'ProgressionStatus', 'TreatmentEligiblity']:
            return text.lower()
        else:
            return standard_text
    else:
        return standard_text

def biomarker_mutation_mapper(master_index_df_ps):
    grouped_out = master_index_df_ps.fillna('').groupby(['nct_id', 'patient_segment_criteria_cleaned'])
    for name, group in grouped_out:
        if group[group['entity'] == 'GENE'].shape[0] > 0 and group[group['entity'].str.contains('DNA')].shape[0] == 0:
            master_index_df_ps.at[(master_index_df_ps['entity'] == 'GENE') & (
                master_index_df_ps['nct_id'] == name[0]) & (
                master_index_df_ps['sentence'] == name[1]), 'resolution'] = 'DNAMutationOther'
        else:
            if group[group['entity'] == 'protein'].shape[0] > 0 and group[group['entity'].str.contains('DNA')].shape[0] == 0:
                master_index_df_ps.at[(master_index_df_ps['entity'] == 'protein') & (
                    master_index_df_ps['nct_id'] == name[0]) & (
                    master_index_df_ps['sentence'] == name[1]), 'resolution'] = 'DNAExpressionOther'
            else:
                if group[(group['entity'] == 'protein') | (
                    group['entity'] == 'GENE')].shape[0] > 0 and group[group['entity'].str.contains('DNA')].shape[0] > 0:
                    pos_biomarker = []
                    for i, row in group.iterrows():
                        if row['entity'] == 'protein' or row['entity'] == 'GENE':
                            pos_biomarker.append(i)
                        else:
                                if 'DNAExpression' in row['entity']:
                                    master_index_df_ps.loc[(
                                        master_index_df_ps.index == pos_biomarker[-1]), 'entity'] = 'protein'
                                    master_index_df_ps.loc[(
                                        master_index_df_ps.index == pos_biomarker[-1]), 'has_relationship'] = 1
                                    master_index_df_ps.at[i, 'relationship'] = str([pos_biomarker[-1]])
                                else:
                                    if 'DNAMutation' in row['entity']:
                                        if ((row['entity'] == 'DNAMutationTranslocation' and ':' in row['text']) or (
                                            row['entity'] == 'DNAMutationOther' and row['text'].lower() == 'fusion')) and len(
                                            pos_biomarker) > 1:
                                            master_index_df_ps.loc[(
                                                master_index_df_ps.index == pos_biomarker[-1]), 'entity'] = 'GENE'
                                            master_index_df_ps.loc[(
                                                master_index_df_ps.index == pos_biomarker[-2]), 'entity'] = 'GENE'
                                            master_index_df_ps.loc[(
                                                master_index_df_ps.index == pos_biomarker[-1]), 'has_relationship'] = 1
                                            master_index_df_ps.loc[(
                                                master_index_df_ps.index == pos_biomarker[-2]), 'has_relationship'] = 1
                                            master_index_df_ps.at[i, 'relationship'] = str([
                                                pos_biomarker[-2], pos_biomarker[-1]])
                                        else:
                                            if len(pos_biomarker) > 0:
                                                master_index_df_ps.loc[(
                                                    master_index_df_ps.index == pos_biomarker[-1]), 'entity'] = 'GENE'
                                                master_index_df_ps.loc[(
                                                    master_index_df_ps.index == pos_biomarker[-1]), 'has_relationship'] = 1
                                                master_index_df_ps.at[i, 'relationship'] =str([pos_biomarker[-1]])
                                            else:
                                                print("text!!! ", row['text'])
                                                print(row['init_text'])
    master_index_df_ps.relationship=master_index_df_ps.relationship.apply(lambda x: eval(x) if x else eval('[]'))
    master_index_df_ps.loc[(master_index_df_ps['entity'] == 'GENE') & (
        master_index_df_ps['has_relationship'] == 0), 'resolution'] = 'DNAMutationOther'
    master_index_df_ps.loc[(master_index_df_ps['entity'] == 'protein') & (
        master_index_df_ps['has_relationship'] == 0), 'resolution'] = 'DNAExpressionOther'
    master_index_df_ps=master_index_df_ps.fillna('')
    return master_index_df_ps.reset_index(drop=True)

def ps_sentence_reconstruction(master_index_df_ps):
    patient_segment_selected=[]
    for nct in master_index_df_ps.nct_id.unique():
        for sent in master_index_df_ps.loc[master_index_df_ps.nct_id==nct,'sentence'].unique():
            if sent!='':
                sub_df=master_index_df_ps.loc[(master_index_df_ps.nct_id==nct)&(master_index_df_ps.sentence==sent)]
                sub_df.entity=sub_df.entity.replace('StageRoman','StageDesc')
                for e in sub_df.entity.unique():
                    if e in [
                             'DNAMutationOther',
                             'DNAMutationDeletion',
                    'DNAMutationTranslocation']:
                        patient_segment_selected.append([nct,sent,
                                                         e,sub_df.loc[sub_df.entity==e,'text'].values.tolist()[0],
                                                        sub_df.loc[sub_df.entity==e,
                                                                   'text'].values.tolist()[0]+' ({})'.format(e.replace('DNA',
                                                                                                                       '')),
                                                        sub_df.loc[sub_df.entity==e,'resolution'].values.tolist()[0]])
                    elif e :
                        std_text=sub_df.loc[sub_df.entity==e,'standard_text'].values.tolist()[0] 
                        if e=='StageDesc': 
                            std_text= 'Stage ' +std_text if 'stage' not in std_text.lower() else std_text
                        if std_text=='':
                            std_text=sub_df.loc[sub_df.entity==e,'text'].values.tolist()[0]
                            if e=='StageDesc': 
                                std_text= 'Stage ' +std_text if 'stage' not in std_text.lower() else std_text
                        patient_segment_selected.append([nct,sent,
                                                         e,sub_df.loc[sub_df.entity==e,'text'].values.tolist()[0],
                                                         std_text,
                                                        sub_df.loc[sub_df.entity==e,'resolution'].values.tolist()[0]])
                    else:
                        patient_segment_selected.append([nct,sent, '','',''])
            else:
                patient_segment_selected.append([nct,sent, '','',''])

    refined_df=pd.DataFrame(patient_segment_selected,columns=['nct_id','sent','entity','text','standard_text',
                                                             'resolution'])
#     return refined_df
    refined_df_grouped=refined_df.groupby(['nct_id','sent']).agg({'standard_text':' '.join}).reset_index()

    refined_df_grouped=refined_df_grouped.rename(columns={'standard_text':'reconstructed_sentence'})

    return refined_df,refined_df_grouped.groupby('nct_id').agg({'reconstructed_sentence':'; '.join}).reset_index()

def fill_standard_text(x):
    e=x['entity'].replace('StageRoman','StageDesc')
    std_text=x['standard_text']
    text=x['text']
    if e in [
             'DNAMutationOther',
             'DNAMutationDeletion',
    'DNAMutationTranslocation']:
        return text+' ({})'.format(e.replace('DNA',''))
    elif e :
        if e=='StageDesc': 
            std_text= 'Stage ' +std_text if 'stage' not in std_text.lower() else std_text
        if std_text=='':
            std_text=text
            if e=='StageDesc': 
                std_text= 'Stage ' +std_text if 'stage' not in std_text.lower() else std_text
        return std_text
    else:
        return std_text
