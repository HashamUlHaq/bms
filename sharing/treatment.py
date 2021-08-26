import sys
import time
import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz

def perform_cleaning(x):
            '''Function for preprocessing the text'''
            x = str(x)
            x = re.sub(r'[^\w\s]','',x)
            x = re.sub(' +',' ',x) #remove consecutive white space
            return x.strip().lower()


def drug_class_dictionary(file_path='./models/editable/cortellis_drug_mapping.csv'):
    mono_drugs = pd.read_csv(file_path)

    mono_drugs = mono_drugs.rename(columns={'Mechanism Of Action':'Mechanism_Of_Action'})


    mono_drugs=mono_drugs.drop(columns=['Unnamed: 0'])
    
    #mono_drugs[mono_drugs.flag=='0']
    mono_drugs=mono_drugs.dropna(subset=['drug_cleaned'])
    mono_drugs.fillna('',inplace=True)
    mono_drugs_req=mono_drugs[['drug_cleaned','Mechanism_Of_Action','synonym_list']]
    mono_drugs_req = mono_drugs_req.rename(columns={'drug_cleaned':'standard_drug'})
    mono_drugs_req['drug_cleaned'] = mono_drugs_req['standard_drug'].apply(perform_cleaning)
    mono_drugs_req=mono_drugs_req.drop_duplicates(subset=['drug_cleaned'])
    mono_drugs_req.set_index('drug_cleaned',inplace=True)
    drug_dict=mono_drugs_req.to_dict(orient='index')
    return drug_dict
        
def drugs_mapping(x,treatment_col,drug_key,drug_dict,use_fuzzy,fuzzy_thresh):
    drug_class = ''
    standard_drug = ''
    synonym_list = ''
    treatment_list = x[treatment_col].copy()
    treatment_list_new = []
    flag_std_drug=0
    flag_drug_class=0
    drug_list = list(drug_dict.keys())
    
    for treatment in treatment_list:
        #Exact
        for drug_name in drug_dict:
            if (perform_cleaning(treatment[drug_key]) in drug_name and len(treatment[drug_key])>3):
                standard_drug = drug_dict[drug_name]['standard_drug']
                drug_class = drug_dict[drug_name]['Mechanism_Of_Action']
                synonym_list = drug_dict[drug_name]['synonym_list']
                flag_drug_class = 'exact'
                if(len(str(drug_class))>0):
                    flag_drug_class = 'exact'
                if(len(str(standard_drug))>0):
                    flag_std_drug = 'exact'
        #Fuzzy
            if(len(drug_class)==0 and use_fuzzy and len(treatment[drug_key])>3):
                Ratios1 = process.extract(perform_cleaning(treatment[drug_key]),drug_list,scorer=fuzz.ratio, limit=1)
                drug_class = ''.join([drug_dict[i]['Mechanism_Of_Action'] for (i,k,m) in Ratios1 if k>fuzzy_thresh])
                standard_drug = ''.join([drug_dict[i]['standard_drug'] for (i,k,m) in Ratios1 if k>fuzzy_thresh])
                synonym_list = ''.join([drug_dict[i]['synonym_list'] for (i,k,m) in Ratios1 if k>fuzzy_thresh])
                if(len(str(standard_drug))>0):
                    flag_std_drug = 'fuzzy'
                if(len(str(drug_class))>0):
                    flag_drug_class ='fuzzy'
        treatment['standard_drug'] = standard_drug 
        treatment['drug_class'] = drug_class
        treatment['synonym_list'] = synonym_list
        treatment_list_new.append(treatment)
    
    return [treatment_list_new,flag_std_drug,flag_drug_class]

        

def extract_treatment_details(x,treatment_reconstructed_col='treatment_reconstructed', arm_col='arm',asso_details_key='associated_details', 
                             ass_details_not_req=['disposition','relativedate'], raw_drug_key='raw_drug', 
                              standard_drug_key='standard_drug'):
    treatment_list = x[treatment_reconstructed_col]
    try:
        arm_data = str(x[arm_col])
    except:
        arm_data = ''
    reconstructed_sentence = []
    for treatment in treatment_list:
        associated_details = treatment[asso_details_key]
        asso_sentence1 =' '.join([associated_details[key].strip() for key in associated_details if key not in ass_details_not_req]).strip()
        standard_drug = treatment[standard_drug_key]  if len(treatment[standard_drug_key])>0 else treatment[raw_drug_key]
        treatment_sentence = [standard_drug + ' ' + asso_sentence1]
        reconstructed_sentence.extend(treatment_sentence) 
    reconstructed_sentence = ' \n '.join(list(set(reconstructed_sentence))).strip()
    reconstructed_sentence = reconstructed_sentence if(len(reconstructed_sentence) >0) else arm_data.strip()
    
    return reconstructed_sentence


def standard_arm(x,sep=' + ',raw_drug_key='raw_drug',standard_drug_key='standard_drug'):
    treatment_list = x
    standard_arm =[]
    for treatment in treatment_list:
        standard_drug = [treatment[standard_drug_key]  if len(treatment[standard_drug_key])>0 else treatment[raw_drug_key]]
        standard_arm.extend(standard_drug)
    return sep.join(list(set(standard_arm)))

def treatment_reconstruction(dataset,drug_dict, is_prior=False,group_type_col='group_type', arm_data_col='arm_data', arm_col='arm',
                             intervention_data_col='intervention_data', 
                             treatment_extracted_col='treatment_extracted', 
                             raw_drug_key='raw_drug', asso_details_key='associated_details', 
                             ass_details_not_req=['disposition','relativedate']):
    
    dataset[['treatment_reconstructed','flag_std_drug','flag_drug_class']]=dataset.apply(lambda x: drugs_mapping(x,
                                                                                        treatment_extracted_col,
                                                                                        raw_drug_key,drug_dict,True,90),
                                                                                         axis=1,result_type='expand')
    if group_type_col in dataset.columns and intervention_data_col in dataset.columns:
        dataset['arm_intervention_raw'] = (dataset[group_type_col] + ' \n ' + dataset[arm_data_col] + ' \n ' + 
                                           dataset[intervention_data_col])
        dataset['arm_intervention_processed'] = (dataset[group_type_col] + ' \n ' +  
                                                 dataset.apply(extract_treatment_details,axis=1))

    else:
        dataset['arm_processed'] = dataset.apply(extract_treatment_details,axis=1)
    
    if not is_prior:
        dataset['standard_arm'] = dataset['treatment_reconstructed'].apply(standard_arm)
        
    dataset['generic_name'] = dataset['treatment_reconstructed'].apply(standard_arm,args=(' | ',))
    dataset['drug_class'] = dataset['treatment_reconstructed'].apply(
                                       lambda x :' | '.join(list(set([i['drug_class'] for i in x if len(i['drug_class'])>0]))))                                                                

    return dataset

    
def relate_drugs_updated(res, sentences, relations, resolution_res, intervention_type):
    
    drug_ent_list = ['drug', 'treatment']
    
    resolutions_list = {}
    for i in resolution_res:
        try:
            disp = i.metadata['all_k_aux_labels'].split(':::')[0]
        except:
            disp = ''
        if disp == '-': disp = ''
        resolutions_list[int(i.begin)] = { 'resolved_text' : i.metadata['resolved_text'],
                                           'rxnorm_code': i.result,
                                           'disposition' : disp,
                                         'distance':  i.metadata['distance']}
        
    sentences_list = {}
    for i in sentences:
        sentences_list[int(i.metadata['sentence'])] = i.result
    
    drugs_list = {}
    other_entities = {}
    
    drug_in_this = False
    for i in res:
        if i.metadata['entity'].lower().strip() == 'drug':
            drug_in_this = True
    
    if drug_in_this == True:
        drug_ent_list = ['drug']
        
    for i in res:
        try:
            resolution=resolutions_list[int(i.begin)]['resolved_text']
        except:
            resolution=''
        
        try:
            rxnorm_code=resolutions_list[int(i.begin)]['rxnorm_code']
        except:
            rxnorm_code=''
            
        try:
            disposition=resolutions_list[int(i.begin)]['disposition']
        except:
            disposition=''
        try:
            distance=resolutions_list[int(i.begin)]['distance']
        except:
            distance=np.nan
        #print(int(i.begin))
        if i.metadata['entity'].lower().strip() in drug_ent_list:
            drugs_list[int(i.begin)] = { 
                                            "raw_sentence": sentences_list[int(i.metadata['sentence'])],
                                            "raw_drug" : i.result,
                                            "drug_entity": i.metadata['entity'],
                                            "resolved_drug": resolution,
                                            "resolved_drug_distance": distance,
                                            'intervention_type': intervention_type,
                                            'rxnorm_code': rxnorm_code,
                                            "associated_details" : { 'dosage' : '', 'form': '', 'route': '',
                                                                       'strength': '', 'frequency': '', 
                                                                        'duration': '', 
                                                                        'disposition': disposition,
                                                                        'relativedate' : '', 
                                                                         'administration' : '', 'cyclelength' : '',
                                                                       }
                                           }
        elif i.metadata['entity'].lower().strip() != 'treatment':
            other_entities[int(i.begin)] = {'type': i.metadata['entity'].strip().lower(), 'chunk': i.result}
            
    done_other_entities = []
    for rel in relations:
        if rel.metadata['entity1'].lower().strip() in drug_ent_list:
            
            drug_ent_id = int(rel.metadata['entity1_begin'])
            
            oth_ent = other_entities[int(rel.metadata['entity2_begin'])]
            try:
                drugs_list[drug_ent_id]['associated_details'][oth_ent['type']] = oth_ent['chunk']
            except:
                print(oth_ent)
                pass
            done_other_entities.append(int(rel.metadata['entity2_begin']))
            
        elif rel.metadata['entity2'].lower().strip() in drug_ent_list:
            drug_ent_id = int(rel.metadata['entity2_begin'])
            oth_ent = other_entities[int(rel.metadata['entity1_begin'])]
            try:
                drugs_list[drug_ent_id]['associated_details'][oth_ent['type']] = oth_ent['chunk']
            except:
                print(oth_ent)
                pass
    
            done_other_entities.append(int(rel.metadata['entity1_begin']))
    
    remaining_entities = set(list(other_entities.keys())) - set(done_other_entities)
    for rem_ent in remaining_entities:
        oth_ent = other_entities[rem_ent]
        smlst, clst, lrgst = find_formula(rem_ent, list(drugs_list.keys()))
        if smlst != None:
            drugs_list[smlst]['associated_details'][oth_ent['type']] = oth_ent['chunk']
        elif lrgst != None:# and drugs_sents[lrgst] == this_sent:
            drugs_list[lrgst]['associated_details'][oth_ent['type']] = oth_ent['chunk']
    return list(drugs_list.values())


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


def extract_treatment_info(df,res_arm,res_intervention, prior_treatment_flag=False):
    all_res = []
    if res_arm and res_intervention:
        for index in range(df.shape[0]):
            json_obj = {}
            ires_rec = res_intervention[index]
            ares_rec = res_arm[index]
            row_rec = df.iloc[index]
            found_drugs = False
            found_drugs_i = relate_drugs_updated(ires_rec['full_chunk'], ires_rec['sentence'],
                                         ires_rec['relations'], ires_rec['resolution_rxnorm'] , 
                                         row_rec['intervention_type']) 
            found_drugs_a = relate_drugs_updated(ares_rec['full_chunk'], ares_rec['sentence'],
                                         ares_rec['relations'], ares_rec['resolution_rxnorm'] , None)
            if found_drugs_i:
                json_obj = found_drugs_i
            elif found_drugs_a:
                json_obj = found_drugs_a
            all_res.append(json_obj)
    elif res_arm:
        for index in range(df.shape[0]):
            json_obj = {}
            ares_rec = res_arm[index]
            row_rec = df.iloc[index]
            found_drugs = False
            if prior_treatment_flag:
                found_drugs_a = relate_drugs_treatments(ares_rec['full_chunk'], ares_rec['sentence'],
                                         ares_rec['relations'], ares_rec['resolution_rxnorm'] , None)
            else:
                found_drugs_a = relate_drugs_updated(ares_rec['full_chunk'], ares_rec['sentence'],
                                         ares_rec['relations'], ares_rec['resolution_rxnorm'] , None)
            if found_drugs_a:
                json_obj = found_drugs_a
        
            all_res.append(json_obj)
        
    elif res_intervention:
        for index in range(df.shape[0]):
            json_obj = {}
            ires_rec = res_intervention[index]
            
            row_rec = df.iloc[index]
            found_drugs = False
            found_drugs_i = relate_drugs_updated(ires_rec['full_chunk'], ires_rec['sentence'],
                                         ires_rec['relations'], ires_rec['resolution_rxnorm'] , 
                                         row_rec['intervention_type']) 
           
            if found_drugs_i:
                json_obj = found_drugs_i
            
            all_res.append(json_obj)
        
    
    df['treatment_extracted']=all_res
    return df


def treatment_extraction(l_model, final_treatment_design_arm_level, custom_col_name=None):
    if 'intervention_data' in final_treatment_design_arm_level.keys():
        intervention_res_design = l_model.fullAnnotate(final_treatment_design_arm_level['intervention_data'].values.tolist())
    else:
        intervention_res_design = None
        
    if 'arm_data' in final_treatment_design_arm_level.keys():        
        arm_res_design = l_model.fullAnnotate(final_treatment_design_arm_level['arm_data'].values.tolist())
    else:
        if custom_col_name:
            arm_res_design = l_model.fullAnnotate(final_treatment_design_arm_level[custom_col_name].values.tolist())
            final_treatment_design_w_treatment_info  = extract_treatment_info(final_treatment_design_arm_level,
                                                                     arm_res_design,intervention_res_design, True)
            return final_treatment_design_w_treatment_info
        else:
            arm_res_design = None
        
    final_treatment_design_w_treatment_info  = extract_treatment_info(final_treatment_design_arm_level,
                                                                     arm_res_design,intervention_res_design)
    
    final_treatment_design_w_treatment_info['arm_reconstructed'] = final_treatment_design_w_treatment_info[
        'treatment_extracted'].apply(
                                lambda  x:'+'.join(list(
                                set(sorted(
                                    [i['resolved_drug'] for i in x if len(
                                        i['resolved_drug'])>0],key=lambda x:len(x))))))
    
    def standard_arm(x,arm_reconstructed_col , arm_col):
        arm_reconstructed= x[arm_reconstructed_col]
        arm = x[arm_col]
        if(len(arm_reconstructed)>0):
            return arm_reconstructed
        else:
            return arm

    final_treatment_design_w_treatment_info['standard_arm']= final_treatment_design_w_treatment_info.apply(
                                                            lambda x:standard_arm(x,'arm_reconstructed','arm'),axis=1)
                                                                       
    return final_treatment_design_w_treatment_info


def treatment_extraction_full(l_model, final_treatment_design_arm_level, spark, custom_col_name=None):
    if 'intervention_data' in final_treatment_design_arm_level.keys():
        intervention_res_design = l_model.transform(spark.createDataFrame(final_treatment_design_arm_level[['intervention_data']].rename(columns={'intervention_data':'text'})).toDF("text")).collect()
        #intervention_res_design = l_model.fullAnnotate(final_treatment_design_arm_level['intervention_data'].values.tolist())
    else:
        intervention_res_design = None
        
    if 'arm_data' in final_treatment_design_arm_level.keys():        
        arm_res_design = l_model.transform(spark.createDataFrame(final_treatment_design_arm_level[['arm_data']].rename(columns={'arm_data':'text'})).toDF("text")).collect()
    else:
        if custom_col_name:
            arm_res_design = l_model.transform(spark.createDataFrame(final_treatment_design_arm_level[[custom_col_name]].rename(columns={custom_col_name:'text'})).toDF("text")).collect()
            final_treatment_design_w_treatment_info  = extract_treatment_info(final_treatment_design_arm_level,
                                                                     arm_res_design,intervention_res_design, True)
            return final_treatment_design_w_treatment_info
        else:
            arm_res_design = None
        
    final_treatment_design_w_treatment_info  = extract_treatment_info(final_treatment_design_arm_level,
                                                                     arm_res_design,intervention_res_design)
    
    final_treatment_design_w_treatment_info['arm_reconstructed'] = final_treatment_design_w_treatment_info[
        'treatment_extracted'].apply(
                                lambda  x:'+'.join(list(
                                set(sorted(
                                    [i['resolved_drug'] for i in x if len(
                                        i['resolved_drug'])>0],key=lambda x:len(x))))))
    
    def standard_arm(x,arm_reconstructed_col , arm_col):
        arm_reconstructed= x[arm_reconstructed_col]
        arm = x[arm_col]
        if(len(arm_reconstructed)>0):
            return arm_reconstructed
        else:
            return arm

    final_treatment_design_w_treatment_info['standard_arm']= final_treatment_design_w_treatment_info.apply(
                                                            lambda x:standard_arm(x,'arm_reconstructed','arm'),axis=1)
                                                                       
    return final_treatment_design_w_treatment_info


def relate_drugs_treatments(res, sentences, relations, resolution_res, intervention_type):
    
    drug_ent_list = ['drug', 'treatment']
    
    resolutions_list = {}
    for i in resolution_res:
        try:
            disp = i.metadata['all_k_aux_labels'].split(':::')[0]
        except:
            disp = ''
        if disp == '-': disp = ''
        resolutions_list[int(i.begin)] = { 'resolved_text' : i.metadata['resolved_text'],
                                          'distance' : i.metadata['distance'],
                                          'rxnorm_code': i.result,
                                     'disposition' : disp}
        
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
        try:
            resolution=resolutions_list[int(i.begin)]['resolved_text']
        except:
            resolution=''
        
        try:
            rxnorm_code=resolutions_list[int(i.begin)]['rxnorm_code']
        except:
            rxnorm_code=''
            
        try:
            disposition=resolutions_list[int(i.begin)]['disposition']
        except:
            disposition=''
        try:
            distance=resolutions_list[int(i.begin)]['distance']
        except:
            distance=np.nan
            
        if i.metadata['entity'].lower().strip() in drug_ent_list:
            drugs_list[int(i.begin)] = {
                                            "raw_sentence": sentences_list[int(i.metadata['sentence'])],
                                            "raw_drug" : i.result,
                                            "drug_entity": i.metadata['entity'],
                                            "resolved_drug": resolution,
                                            "resolved_drug_distance": distance,
                                            'intervention_type': intervention_type,
                                            'rxnorm_code': rxnorm_code,
                                            "associated_details" : { 'dosage' : '', 'form': '', 'route': '',
                                                                       'strength': '', 'frequency': '', 
                                                                        'duration': '', 
                                                                        'disposition': disposition,
                                                                        'relativedate' : '', 
                                                                         'administration' : '', 'cyclelength' : '',
                                                                       }
                                           }
        elif i.metadata['entity'].lower().strip() == 'treatment':
            drugs_list[int(i.begin)] = {
                                            "raw_sentence": sentences_list[int(i.metadata['sentence'])],
                                            "raw_drug" : i.result,
                                            "drug_entity": i.metadata['entity'],
                                            "resolved_drug": resolution,
                                            "resolved_drug_distance": distance,
                                            'intervention_type': intervention_type,
                                            'rxnorm_code': rxnorm_code,
                                            "associated_details" : { 'dosage' : '', 'form': '', 'route': '',
                                                                       'strength': '', 'frequency': '', 
                                                                        'duration': '', 
                                                                        'disposition': disposition,
                                                                        'relativedate' : '', 
                                                                         'administration' : '', 'cyclelength' : '',
                                                                       }
                                           }
        elif i.metadata['entity'].lower().strip() != 'treatment' and i.metadata['entity'].lower().strip() != 'drug':
            other_entities[int(i.begin)] = {'type': i.metadata['entity'].strip().lower(), 'chunk': i.result}
    
    done_other_entities = []
    for rel in relations:
        if rel.metadata['entity1'].lower().strip() in drug_ent_list:
            drug_ent_id = int(rel.metadata['entity1_begin'])
            oth_ent = other_entities[int(rel.metadata['entity2_begin'])]
            drugs_list[drug_ent_id]['associated_details'][oth_ent['type']] = oth_ent['chunk']
            done_other_entities.append(int(rel.metadata['entity2_begin']))
            
        elif rel.metadata['entity2'].lower().strip() in drug_ent_list:
            drug_ent_id = int(rel.metadata['entity2_begin'])
            oth_ent = other_entities[int(rel.metadata['entity1_begin'])]
            drugs_list[drug_ent_id]['associated_details'][oth_ent['type']] = oth_ent['chunk']
        
            done_other_entities.append(int(rel.metadata['entity1_begin']))
    remaining_entities = set(list(other_entities.keys())) - set(done_other_entities)
    for rem_ent in remaining_entities:
        oth_ent = other_entities[rem_ent]
        smlst, clst, lrgst = find_formula(rem_ent, list(drugs_list.keys()))
        #this_sent = int(i.metadata['sentence'])
        if smlst != None:
            drugs_list[smlst]['associated_details'][oth_ent['type']] = oth_ent['chunk']
        elif lrgst != None:# and drugs_sents[lrgst] == this_sent:
            drugs_list[lrgst]['associated_details'][oth_ent['type']] = oth_ent['chunk']
    return list(drugs_list.values())

