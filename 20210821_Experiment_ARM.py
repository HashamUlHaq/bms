#!/usr/bin/env python
# coding: utf-8

# ## Environment and Spark Session setup

# In[1]:


import sys
sys.path.append("..")
from ctdi_treatment.env_setup_start import *
import time
license_keys = set_envvars('../keys/nlp_keys_312.json')
spark = start_sparknlp(license_keys)


# ## Getting the joined `design_group` + `intervention` table

# In[2]:


import pandas as pd
alldf = pd.read_csv("../data/raw_aac.csv")
alldf.columns


# ### Also calculating the fields that represent the actual input to the pipeline
# #### Concatenation of ARM Title and Desc, Intervention Name and Desc,  Intervention Type, Name and Desc

# In[3]:


name_sep = ": "
sent_sep = "."
tit_sep = "\n\n"

alldf["arm_title_desc"] = alldf["title"]+tit_sep+alldf["arm_desc"]
alldf["intervention_name_desc"] = alldf["name"]+name_sep+alldf["intervention_desc"]+sent_sep
alldf["intervention_type_name_desc"] = alldf["intervention_type"]+tit_sep+alldf["intervention_name_desc"]
alldf.shape


# In[5]:


annotation_df = alldf.groupby(["nct_id","design_group_id","title","arm_desc"]).agg(
    intervention_type_name_desc=("intervention_type_name_desc", tit_sep.join)
).reset_index()

# Exps Quadruplets: 1. DataFrame, 2. PipelineField, 3. AACT fallback field for drugs
exps = [
    (alldf[["nct_id","design_group_id","title","arm_desc","arm_title_desc"]].drop_duplicates(), "arm_title_desc", "title"),
    (alldf[["nct_id","design_group_id","title","arm_desc","arm_title_desc","name","intervention_type","intervention_name_desc"]].drop_duplicates(), "intervention_name_desc", "name"),
    (annotation_df, "intervention_type_name_desc", "name"),
]


# ## Pipeline Execution

# ### Building or Loading the Spark Pipeline

# In[6]:


from ctdi_treatment.treatment_pipeline import build_treatment_pipeline, build_df, LightPipeline, PipelineModel
pl_name = "20210722_ner_pl_from_arms"
build_or_load = "build"#"load"
s = time.time()
if build_or_load=="build":
    pl = build_treatment_pipeline()
    plm = pl.fit(spark.createDataFrame([("",)]).toDF("text"))
else:
    plm = PipelineModel.load(f"models/{pl_name}")
print(time.time()-s)


# ### Saving the Pipeline

# In[7]:


save_pipeline = False
if save_pipeline:
    s = time.time()
    plm.write().overwrite().save(f"models/{pl_name}")
    print(time.time()-s)


# ### Creating SparkNLP LightPipeline to avoid Spark overhead

# In[8]:


lpl = LightPipeline(plm)


# In[9]:


use_light_pipeline = True
s = time.time()
dfs = []
for e in exps:
    if use_light_pipeline:
        dfs.append((e[-1],build_df(*e[:-1], lpl)))
    else:
        plm.stages[0].setInputCol(e[1])
        # These line here makes no sense in reality but just for reference if you were to use a cluster
        dfs.append((e[-1],plm.transform(spark.createDataFrame(e[0])).toPandas()))
print(time.time()-s)


# In[10]:


for _,o in dfs:
    print(o.shape)


# ## Running postprocessing for all experiments

# In[13]:


import pandas as pd
from scipy.stats import mode
from ctdi_treatment.postprocessing import build_dict_acc, prepare_output_acc, aggregate_entity_dict, dict_diff_acc, dict_join_append_acc
for i, (ff,o) in enumerate(dfs):
    dfs[i][1]["entity_dict"] = o["full_chunk"].apply(build_dict_acc(by_sentence=False))
    dfs[i][1]["entity_dict_sent"] = o["full_chunk"].apply(build_dict_acc())
    dfs[i][1]["num_sents"] = o["sentence"].apply(len)
    dfs[i][1]["num_drugs"] = o["entity_dict"].apply(lambda x: len(x.get("Drug",[])))
    dfs[i][1]["output"] = o.apply(prepare_output_acc(fallback_field=ff, name_sep=name_sep, sent_sep=sent_sep, tit_sep=tit_sep), axis=1)
    dfs[i][1][['output_class', 'missing_entities', 'output']] = pd.DataFrame(o["output"].tolist(), index=o.index)


# In[14]:


(_,dfx), (_,dfz), (_,dfa) = dfs


# ### Using ARM and Intervention level information together and calculate `combined` final output

# ### Aggregation of the Intervention level output

# In[15]:


dfz_ = dfz.groupby(["nct_id","design_group_id"])    .agg({"name":name_sep.join,"title":max,"arm_desc":max,
           "intervention_name_desc":tit_sep.join,"entity_dict":aggregate_entity_dict,
          "output_class":lambda x: mode(x)[0],"missing_entities":sum,"output":sum,"num_drugs":sum}).reset_index()
dfz_.shape


# In[16]:


arm_int = pd.merge(dfx,dfz.drop(["nct_id","title","arm_desc","arm_title_desc"],axis=1),on="design_group_id")


# In[17]:


required = ["Drug","Strength","Administration"]
arm_int["entity_diff_xy"] = arm_int[["entity_dict_x","entity_dict_y"]].apply(dict_diff_acc(False,required), axis=1)
arm_int["label_diff_xy"] = arm_int[["entity_dict_x","entity_dict_y"]].apply(dict_diff_acc(True,required), axis=1)
arm_int["labels_x"] = arm_int["entity_dict_x"].apply(lambda x: [y for y in x])
arm_int["labels_y"] = arm_int["entity_dict_y"].apply(lambda x: [y for y in x])


# In[18]:


arm_int["output_read_x"] = arm_int["output_x"].apply(lambda x: "\n".join([str(y) for y in x]))
arm_int["output_read_y"] = arm_int["output_y"].apply(lambda x: "\n".join([str(y) for y in x]))


# In[19]:


arm_int["combined"] = arm_int.apply(dict_join_append_acc(),axis=1)
arm_int["combined_read"] = arm_int["combined"].apply(lambda x: "\n".join([str(y) for y in x]))


# In[20]:


arm_int[["nct_id","title","arm_title_desc","name","intervention_name_desc","output_class_x","output_class_y","output_x","output_y","entity_diff_xy","combined","output_read_x","output_read_y","combined_read"]].to_csv("../data/arm_title_desc_int_name_desc_combined.csv",index=False)


# ## Create some preannotated data for the AnnotationLab

# In[21]:


ann_df = pd.merge(dfx,dfa.drop(["nct_id","title","arm_desc"],axis=1),on="design_group_id")


# In[22]:


from ctdi_treatment.annotation_lab import write_json_files
write_json_files(ann_df, ["nct_id","title"],[("arm","arm_title_desc","full_chunk_x","relations_x"),
                  ("int","intervention_type_name_desc","full_chunk_y","relations_y")], out_path="../data")


# In[ ]:




