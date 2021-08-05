import re as rex
import itertools
import pylcs

def build_dict_acc(head_label="Drug", by_sentence=True, schema = ["chunk","sentence","result","begin","end"]):
    """Returns a function that nests the entity label from the Annotation list"""
    def build_dict_inner(x):
        """Returns a dictionary with entity labels in its keys and the respective list of Annotations in its values"""
        def aggregate_by_sentence(l):
            it = itertools.groupby(l, lambda x:x.metadata["sentence"])
            for key, subiter in it:
                rv = dict(zip(schema, sorted((xi.metadata["chunk"],xi.metadata["sentence"],xi.result,xi.begin,xi.end) for xi in subiter)))
                yield key, rv
                
        if by_sentence:
            return list(aggregate_by_sentence(x))
        else:
            rt={}
            for xi in x:
                cs = rt.get(xi.metadata["entity"], [])
                cs.append(dict(zip(schema,(xi.metadata["chunk"],xi.metadata["sentence"],xi.result,xi.begin,xi.end))))
                rt[xi.metadata["entity"]] = cs
        return rt
    return build_dict_inner


def prepare_output_acc(dict_field="entity_dict",relations_field="relations",resolution_field="resolution_rxnorm",
                       assertion_field="assertion",fallback_field="title",head_label="Drug",
                       entity_order=["Drug","Form","Dosage","Strength","Administration","Frequency","CycleLength","Treatment"],
                      schema = ["chunk","sentence","result","begin","end"], name_sep=":", sent_sep=".", tit_sep="\n\n"):
    """Returns a function to process SparkNLP output fields into BMS CTDI Designed Taxonomy for Treatments of 
    ["Drug","Form","Dosage","Strength","Administration","Frequency","CycleLength","Treatment","_DrugGeneric"]
    """
    def prepare_output_inner(row):
        """Returns a function to process SparkNLP output into BMS CTDI Designed Taxonomy for Treatments
        For each row it will return a triplet of:
        (Object Calculation Algorithm, Missing Entities from the Taxonomy, Output List of Treatment Objects)
        """
        def be_rel(x, node=1):
            return int(x.metadata[f"entity{node}_begin"]), int(x.metadata[f"entity{node}_end"])
        def is_related(dij, x):
            xm1 = be_rel(x,1)
            xm2 = be_rel(x,2)
            cm = dij["begin"],dij["end"]
            rv = (xm1==cm) or (xm2==cm)
            return rv
        
        d = row[dict_field]
        r = row[relations_field]
        ord_label = f"_{head_label}_begin"
        std_label = f"_{head_label}Generic"
        std_label_temp = f"_{head_label}Generic_"
        dispo_label = f"_{head_label}Class"
        ## Nest begin and end in resolution field
        ssc = [std_label+"Code",std_label,std_label+"Distance",dispo_label]
        s = {(rs.begin,rs.end):
             (rs.result,rs.metadata["resolved_text"],rs.metadata["distance"],rs.metadata.get("aux_label",""))
             for rs in row[resolution_field]}
        
        f = row[fallback_field] if fallback_field in row else ""
        
        # Is there a head_label like 'Drug'
        missing_entities = []
        for e in entity_order:
            if e not in d:
                missing_entities.append(e)
        if head_label in d:
            rv = [{k:set([dii["result"] if k!=head_label else dii["result"].title() for dii in di]) for k,di in d.items()}]
            #Remove Placebo from distinct count
            deduped_head = set([x["result"].title() for x in d[head_label] if "placebo" not in x["result"].lower()])
            deduped_std = set([s[(dh["begin"],dh["end"])] for dh in d[head_label]])
            
            # Just one
            if len(deduped_head)==1:
                for rvi in rv:
                    rvi[std_label_temp] = set([s[(dh["begin"],dh["end"])] for dh in d[head_label]])
                cl = "Single Drug"
            # multiple
            else:
                # Iterate rels explore all paths
                cl = ("_Mul" if len(deduped_head) > 0 else "_None")+("Rels" if r else "")
                # Nest begin and end in entity dictionary dropping label
                be_ent = dict([((x["begin"],x["end"]),x) for _,y in d.items() for x in y])
                if not r:
                    pass
                else:
                    rv = []
                    for di in d:
                        if di==head_label:
                            ## Aggregate if necessary
                            ahi = itertools.groupby(d[head_label], lambda x:x["result"].title())
                            for tt, dii in ahi:
                                dii_ = list(dii)
                                new_dict = {head_label: {tt}, ord_label:min([x["begin"] for x in dii_]),
                                           std_label_temp:set(s[(dii_j["begin"],dii_j["end"])] for dii_j in dii_)}
                                for dij in dii_:
                                    #TODO: Merge this filter with the next cycle
                                    rels = filter(lambda z: is_related(dij, z), r)
                                    for rdij in rels:
                                        xm1 = be_rel(rdij,1)
                                        xm2 = be_rel(rdij,2)
                                        cm = dij["begin"],dij["end"]
                                        rmd = rdij.metadata
                                        arg_pair = 2 if cm==xm1 else 1
                                        arg_pair = (f"entity{arg_pair}",f"chunk{arg_pair}")
                                        cs = new_dict.get(rmd[arg_pair[0]], set())
                                        cs.add(rmd[arg_pair[1]])
                                        new_dict[rmd[arg_pair[0]]] = cs
                                rv.append(new_dict)
                    rv = sorted(rv, key=lambda x: x[ord_label])
        else:
            cl = "_DrugInferred"
            d[head_label] = [dict(zip(schema+[ord_label],(-1,-1,fi.title(),-1,-1,-1))) 
                             for fi in rex.split(f"(?:{name_sep}|{sent_sep})", f)]
            rv = [{k:set([dii["result"] for dii in di]) for k,di in d.items()}]
        for r in rv:
            if ord_label in r:
                del(r[ord_label])
            if std_label_temp in r:
                std_dicts = []
                for std in r[std_label_temp]:
                    std_dicts.append(dict(zip(ssc, std)))
                r[std_label] = std_dicts
                del(r[std_label_temp])
        return (cl,missing_entities,rv)
    return prepare_output_inner


def aggregate_entity_dict(x):
    all_ents = set([y for xi in x for y in xi])
    merged_dict = {}
    for e in all_ents:
        for xi in x:
            merged_dict[e] = merged_dict.get(e,[])+xi.get(e,[])
    return merged_dict


def dict_diff_acc(just_labels=False, filter_label=""):
    def dict_diff_inner(x):
        def get_keys_or_values(a, just_labels=just_labels, filter_label=filter_label):
            if just_labels:
                return [x for x in a.keys() if not filter_label or x in filter_label]
            else:
                z= {kai:[aij["result"] for aij in ai] for kai, ai in a.items() if ai and (not filter_label or kai in filter_label)}
                return z
        a,b = x
        sa = get_keys_or_values(a, just_labels, filter_label)
        sb = get_keys_or_values(b, just_labels, filter_label)

        if just_labels:
            rd = {"left":set(sa).difference(sb),"inter":set(sa).intersection(sb),"right":set(sb).difference(sa)}
        else:
            l = {sak:set(sa[sak]).difference(sb.get(sak,[])) for sak in sa if not filter_label or sak in filter_label}
            t = {sak:set(sa[sak]).intersection(sb.get(sak,[])) for sak in sa if not filter_label or sak in filter_label}
            r = {sbk:set(sb[sbk]).difference(sa.get(sbk,[])) for sbk in sb if not filter_label or sbk in filter_label}
            rd = {"left":{k:li for k,li in l.items() if li},
           "inter":{k:li for k,li in t.items() if li},
           "right":{k:li for k,li in r.items() if li}}
        return rd
    return dict_diff_inner


def dict_join_append_acc(x_field="output_x", y_field="output_y",how="right",head_label="Drug",sim_pct=0.9):
    def combine_01_inner(row):
        card, rest = (x_field, y_field) if how=="left" else (y_field, x_field)
        c,r = row[card], row[rest]
        for ci in c:
            if head_label in ci:
                dr = list(ci[head_label])[0]
                ris = []
                for ri in r:
                    drr = list(ri.get(head_label,set()))
                    if drr and pylcs.lcs(drr[0],dr)/max(1,min(len(drr[0]),len(dr)))>=sim_pct:
                        for kri, vri in ri.items():
                            if not kri.startswith("_"):
                                last_l = len(ci.get(kri,[]))
                                ci[kri] = ci.get(kri, set()).union(vri)
        return c
    return combine_01_inner