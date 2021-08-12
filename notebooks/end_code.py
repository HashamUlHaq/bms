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

def relate_drugs(res, sentences, relations, resolution_res, intervention_type):
    
    drug_ent_list = ['drug', 'treatment']
    
    resolutions_list = {}
    for i in resolution_res:
        resolutions_list[int(i.begin)] = { 'resolved_text' : i.metadata['resolved_text'],
                                     'disposition' : 'pa'}
        
    sentences_list = {}
    for i in sentences:
        sentences_list[int(i.metadata['sentence'])] = i.result
    
    #print (sentences_list)
    
    drugs_list = {}
    other_entities = {}
    
    drug_in_this = False
    for i in res:
        if i.metadata['entity'].lower().strip() == 'drug':
            drug_in_this = True
    
    if drug_in_this == True:
        drug_ent_list = ['drug']
        
    
    for i in res:
        #print(int(i.begin))
        if i.metadata['entity'].lower().strip() in drug_ent_list:
            drugs_list[int(i.begin)] = { resolutions_list[int(i.begin)]['resolved_text'] : {
                                            "raw_sentence": sentences_list[int(i.metadata['sentence'])],
                                            "raw_drug" : i.result,
                                            "drug_entity": i.metadata['entity'],
                                            "resolved_drug": resolutions_list[int(i.begin)]['resolved_text'],
                                            'intervention_type': intervention_type,
                                            "associated_details" : { 'dosage' : '', 'form': '', 'route': '',
                                                                       'strength': '', 'frequency': '', 
                                                                        'duration': '', 
                                                'disposition': resolutions_list[int(i.begin)]['disposition'],
                                                                        'relativedate' : '', 
                                                                         'administration' : '', 'cyclelength' : '',
                                                                       }
                                           }
                                       }
        else:
            other_entities[int(i.begin)] = {'type': i.metadata['entity'].strip().lower(), 'chunk': i.result}
    
    for rel in relations:
        if rel.metadata['entity1'].lower().strip() in drug_ent_list:
            drug_ent_id = int(rel.metadata['entity1_begin'])
            oth_ent = other_entities[int(rel.metadata['entity2_begin'])]
            drugs_list[drug_ent_id][resolutions_list[drug_ent_id]['resolved_text']]['associated_details'][oth_ent['type']] = oth_ent['chunk']
        
        elif rel.metadata['entity2'].lower().strip() in drug_ent_list:
            drug_ent_id = int(rel.metadata['entity2_begin'])
            oth_ent = other_entities[int(rel.metadata['entity1_begin'])]
            drugs_list[drug_ent_id][resolutions_list[drug_ent_id]['resolved_text']]['associated_details'][oth_ent['type']] = oth_ent['chunk']
        
    return list(drugs_list.values())
                             

assert len(intervention_res) == annotation_df.shape[0] == len(arm_res)
all_res = []
for index in range(annotation_df.shape[0]):
    json_obj = {}
    
    ires_rec = intervention_res[index]
    ares_rec = arm_res[index]
    row_rec = annotation_df.iloc[index]
    
    found_drugs = False
    found_drugs_i = relate_drugs(ires_rec['full_chunk'], ires_rec['sentence'],
                                 ires_rec['relations'], ires_rec['resolution_rxnorm'] , 
                                 row_rec['intervention_type']) 
    found_drugs_a = relate_drugs(ares_rec['full_chunk'], ares_rec['sentence'],
                                 ares_rec['relations'], ares_rec['resolution_rxnorm'] , None)
    if found_drugs_i:
        json_obj[row_rec['group_type'] + ' ' + row_rec['title']] = found_drugs_i
    elif found_drugs_a:
        json_obj[row_rec['group_type'] + ' ' + row_rec['title']] = found_drugs_a
    
    all_res.append(json_obj)
    
    
import json
with open('output.json', 'w') as f_:
    f_.write(json.dumps(all_res, indent=4))