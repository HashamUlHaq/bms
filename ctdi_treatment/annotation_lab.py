import os
import json
import pandas as pd
import random as rand


def generate_hash(length=10):    
    nums = list(range(48,58))
    uppers = list(range(65,91))
    lowers = list(range(97,123))
    all_chars = nums+uppers+lowers
    return "".join([chr(all_chars[rand.randint(0, len(all_chars)-1)]) for x in range(length)])


def build_label(chunk, start, end, label, toname):
    label_json = {
            "from_name": toname+"_l",
            "id": generate_hash(),
            "source": f"${toname}",
            "to_name": toname,
            "type": "labels",
            "value": {
              "end": end,
              "labels": [
                label
              ],
              "start": start,
              "text": chunk
            }
          }
    return label_json


def build_relation(from_id, to_id):
    label_json = {
            "direction": "right",
            "from_id": from_id,
            "to_id": to_id,
            "type": "relation"
          }
    return label_json


def write_json_files(view_df, title_cols, label_relation_cols, groupby="batch", step=1000, out_path="data"):
    if type(title_cols)==str:
        title_cols = [title_cols]
    elif type(title_cols)!=list:
        raise ValueError("title_cols should be string or list")
        
    if groupby not in view_df.columns:
        view_df[groupby] = "_"
        
    for g,gr in view_df.groupby([groupby]).groups.items():
        for z in range(0, gr.shape[0], step):
            import_json = []
            temp_df = view_df.loc[gr]
            min_top = min(z+step, temp_df.shape[0])
            for t in temp_df.sort_values("nct_id").iloc[z:z+step].iterrows():
                # print(t[1])
                # fixed right edge +1              
                result = []
                data = {"title":" ".join([t[1][tc] for tc in title_cols])}
                for ton,f,l,r in label_relation_cols:
                    entity_dict = {}
                    for x in t[1][l]:
                        if not pd.isna(x):
                            result.append(build_label(x.result, x.begin, x.end+1, x.metadata["entity"], ton))
                            entity_dict[(x.begin, x.end)] = result[-1]["id"]
                    for x in t[1][r]:
                        if not pd.isna(x):
                            result.append(build_relation(
                                entity_dict[int(x.metadata["entity1_begin"]),int(x.metadata["entity1_end"])],
                                  entity_dict[int(x.metadata["entity2_begin"]),int(x.metadata["entity2_end"])]
                             ))
                    data[ton] = t[1][f]
                import_json.append({"predictions": [{
                        "id":0,
                        "result":result,
                    }],
                    "data":data,
                })
            json.dump(import_json, open(os.path.join(out_path, f"import_{g}_{z}_{min_top}.json"), "w"))
