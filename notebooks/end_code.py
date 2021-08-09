import sys
import json
sys.path.append("..")
from ctdi_treatment.env_setup_start import *
import time
license_keys = set_envvars('/home/ubuntu/hasham/jsl_keys.json')
spark = start_sparknlp(license_keys)

import pandas as pd

from ctdi_treatment.treatment_pipeline import build_treatment_pipeline, build_df, LightPipeline, PipelineModel

###### Change file name here ############
alldf = pd.read_csv("../data/raw_aac.csv")
print (alldf.shape)
print (alldf.columns)


name_sep = ": "
sent_sep = "."
tit_sep = "\n\n"


alldf["arm_title_desc"] = alldf["title"]+tit_sep+alldf["arm_desc"]

alldf["intervention_name_desc"] = alldf["name"]+name_sep+alldf["intervention_desc"]+sent_sep

pipeline = build_treatment_pipeline()
p_model = pipeline.fit(spark.createDataFrame([("",)]).toDF("text"))
l_model = LightPipeline(p_model)

annotation_df = alldf.groupby(["nct_id","design_group_id","group_type","title"]).agg(
    intervention_name_desc=("intervention_name_desc", tit_sep.join),
    arm_title_desc = ('arm_title_desc', tit_sep.join),
    intervention_type = ( 'intervention_type', '|'.join)
).reset_index()


intervention_res = l_model.fullAnnotate(annotation_df['intervention_name_desc'].values.tolist())
arm_res = l_model.fullAnnotate(annotation_df['arm_title_desc'].values.tolist())



def find_formula(k, l, mode='small', respect_section=None):
        
    try:
        lrgst = min(filter(lambda x: x > k, l))
    except:
        lrgst = None
    try:
        clst = min(list(filter(lambda x: x != k, l)), key=lambda x: abs(x-k))
    except:
        clst = None
    try:
        smlst = max(filter(lambda x: x < k, l))
    except:
        smlst = None
    
    return smlst, clst, lrgst

def find_drugs(res, resolution_res):
    
    resolutions = {}
    for i in resolution_res:
        resolutions[int(i.begin)] = i.metadata['resolved_text']
    
    drugs = []
    other_entities = []
    drugs = {}
    drugs_sents = {}
    for i in res:
        if i.metadata['entity'].lower().strip() == 'drug':
            #drugs[int(i.begin)] = { 'entity': 'drug', 'result': i.result, 'sentence': i.metadata['sentence'] }
            try:
                drugs[int(i.begin)] = {resolutions[int(i.begin)] : { 'Dosage' : '', 'Form': '', 'Route': '',
                                                                   'Strength': '', 'Frequency': '', 'Duration': ''}}
                drugs_sents[int(i.begin)] = int(i.metadata['sentence'])
            except:
                pass
    for i in res:
        
        if i.metadata['entity'].lower().strip() in ['strength', 'route', 'frequency', 'form', 'dosage','duration']:
            smlst, clst, lrgst = find_formula(int(i.begin), list(drugs.keys()))
            this_sent = int(i.metadata['sentence'])
            if smlst != None:
                drugs[smlst][resolutions[smlst]][i.metadata['entity'].strip().title()] = i.result
            if lrgst != None and drugs_sents[lrgst] == this_sent:
                drugs[lrgst][resolutions[lrgst]][i.metadata['entity'].strip().title()] = i.result
                
    return list(drugs.values())
                             

assert len(intervention_res) == annotation_df.shape[0] == len(arm_res)
all_res = []
for index in range(annotation_df.shape[0]):
    json_obj = {}
    
    ires_rec = intervention_res[index]
    ares_rec = arm_res[index]
    row_rec = annotation_df.iloc[index]
    
    found_drugs = False
    found_drugs_i = find_drugs(ires_rec['all_chunk'], ires_rec['resolution_rxnorm'] ) 
    found_drugs_a = find_drugs(ares_rec['all_chunk'], ares_rec['resolution_rxnorm'])
    if found_drugs_i:
        json_obj[row_rec['group_type'] + ' ' + row_rec['title']] = found_drugs_i
    elif found_drugs_a:
        json_obj[row_rec['group_type'] + ' ' + row_rec['title']] = found_drugs_a
    
    all_res.append(json_obj)
    
    
import json
with open('output.json', 'w') as f_:
    f_.write(json.dumps(all_res, indent=4))