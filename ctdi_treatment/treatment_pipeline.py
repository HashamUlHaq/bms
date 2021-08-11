from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import math
import pandas as pd

def build_ner_pl(tptms, prefix=""):
    tms = []
    for t, (n, wl) in tptms.items():
        lt = n.lower()
        out = prefix+lt
        ner = MedicalNerModel.pretrained(t,"en","clinical/models").setInputCols("sentence","token","embs")\
            .setOutputCol(out)
        tms.append(ner)
        conv = NerConverterInternal().setInputCols("sentence","token",out).setOutputCol(out+"_chunk")\
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


def build_merging_pl(tptms, merge_overlapping=True, prefix=""):
    tms, logic_merge, last_tm = [], None, None
    if len(tptms) > 2:
        logic_merge = merging_logic(tptms)
        for fix in logic_merge: 
            for com in fix:
                cm = ChunkMergeApproach().setInputCols(prefix+com[0],prefix+com[1]).setOutputCol(prefix+com[2])\
                    .setMergeOverlapping(merge_overlapping)
                tms.append(cm)
            if fix==logic_merge[-1]:
                last_tm = prefix+fix[-1][2]
    return tms, logic_merge, last_tm 


def build_treatment_pipeline(editable_models_path="../models/editable",fixed_models_path="../models/fixed"):
    # Preprocessing pipeline
    da = DocumentAssembler().setInputCol("text").setOutputCol("document")
    #sd = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    
    sd = SentenceDetector().setCustomBounds(["\r\n","\n","\r",": ","; ","\. "])\
        .setInputCols(["document"]) \
        .setOutputCol("sentence")
    
    tk = Tokenizer().setInputCols("sentence").setOutputCol("token")
    emb = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models").setOutputCol("embs")
    
    pos = PerceptronModel().pretrained("pos_clinical", "en", "clinical/models") \
        .setInputCols(["sentence", "token"]).setOutputCol("pos_tags")
    dependency_parser = DependencyParserModel()\
        .pretrained("dependency_conllu", "en")\
        .setInputCols(["sentence", "pos_tags", "token"])\
        .setOutputCol("dependencies")

    from collections import OrderedDict
    ners_to_merge = OrderedDict({
        "ner_posology":("posology",[]),
        "ner_clinical":("base",["TREATMENT"]), 
        "ner_clinical_large":("large",["TREATMENT"]), 
        "ner_jsl":("jsl",["TREATMENT"]), 
    })

    ner_pl = build_ner_pl(ners_to_merge)

    cms, logic_merge, last_tm = build_merging_pl([k[0].lower()+"_chunk" for k in ners_to_merge.values()])
    if cms:
        cms[-1].setOutputCol("all_chunk")
    else:
        ner_pl[-1].setOutputCol("all_chunk") ## so, all_chunk is everything - Treatment, full posology - NER

    rexm = RegexMatcher()\
      .setExternalRules(f"{editable_models_path}/arms_regex.csv", ",", "TEXT")\
      .setStrategy("MATCH_ALL").setInputCols(["sentence"]).setOutputCol("rex_chunk")

    tm = TextMatcher().setInputCols("sentence","token").setOutputCol("textmatch_chunk").setEntityValue("Treatment")\
        .setEntities(f"{editable_models_path}/arms_treatment_textmatcher.csv").setCaseSensitive(False).setMergeOverlapping(True)\
        .setBuildFromTokens(True)

    ass = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
        .setInputCols(["sentence", "treat_chunk", "embs"]) \
        .setOutputCol("assertion")

    cmrh = ChunkMergeApproach().setInputCols("rex_chunk","textmatch_chunk").setOutputCol("rex_text_chunk")\
        .setMergeOverlapping(False)\
        .setReplaceDictResource(f"{editable_models_path}/replace_dict.csv","TEXT", {"delimiter":","})\
        .setFalsePositivesResource(f"{editable_models_path}/fp_dict.csv","TEXT", {"delimiter":","})
    
    cmrha = ChunkMergeApproach().setInputCols("rex_text_chunk","all_chunk").setOutputCol("full_chunk")\
        .setMergeOverlapping(False) \
        .setReplaceDictResource(f"{editable_models_path}/replace_dict.csv","TEXT", {"delimiter":","})\
        .setFalsePositivesResource(f"{editable_models_path}/fp_dict.csv","TEXT", {"delimiter":","})

    #conv_drug = ChunkFilterer().setInputCols("sentence","full_chunk").setOutputCol("drug_chunk").setWhiteList(["Drug","drug","DRUG"])
    conv_drug = ChunkMergeApproach()\
        .setInputCols("full_chunk","all_chunk")\
        .setOutputCol("drug_chunk")\
        .setMergeOverlapping(False)\
        .setReplaceDictResource(f"{fixed_models_path}/replace_dict_drug.csv","TEXT", {"delimiter":","})
    
    #conv_treat = ChunkFilterer().setInputCols("sentence","full_chunk").setOutputCol("treat_chunk").setWhiteList(["TREATMENT"])
    
    conv_treat = ChunkMergeApproach()\
        .setInputCols("full_chunk","full_chunk")\
        .setOutputCol("treat_chunk")\
        .setMergeOverlapping(False)\
        .setReplaceDictResource(f"{fixed_models_path}/replace_dict_treat.csv","TEXT", {"delimiter":","})

    
    pair_nd = ["strength","duration","frequency","dosage","route","form",
                           "relativedate","administration","cyclelength"]
    pair_dr = ['drug', 'treatment']
    pairs = [ f'{j}-{i}' for i in pair_nd for j in pair_dr]
    pairs += [ i.split('-')[1]+'-'+i.split('-')[0] for i in pairs ]

    posology_re = RelationExtractionModel()\
        .pretrained("posology_re")\
        .setInputCols(["embs", "pos_tags", "full_chunk", "dependencies"])\
        .setOutputCol("relations")\
        .setMaxSyntacticDistance(5)\
        .setRelationPairs(pairs)

    c2d = Chunk2Doc().setInputCols("drug_chunk").setOutputCol("sbert_doc")

    sbert = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
        .setInputCols("sbert_doc").setOutputCol("sbert_embeddings_sbert")


    rxnorm = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_dispo","en","clinical/models")\
        .setInputCols(f"sbert_doc","sbert_embeddings_sbert").setOutputCol("resolution_rxnorm").setNeighbours(3)
    
#     snomed = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings","en","clinical/models")\
#         .setInputCols(f"sbert_doc","sbert_embeddings_sbert").setOutputCol("resolution_rxnorm").setNeighbours(3)
    
    pl = Pipeline().setStages([da,sd,tk,emb,pos,dependency_parser,rexm,tm] + 
                          ner_pl + cms + [cmrh, cmrha, conv_drug, conv_treat, posology_re, c2d, sbert, rxnorm])
    return pl


def build_df(input_data, field, lpl):
    data = lpl.fullAnnotate(input_data[field].tolist())
    for j,((_,i),d) in enumerate(zip(input_data.iterrows(), data)):
        for c in input_data.columns:
            data[j][c] = i[c]
    return pd.DataFrame.from_dict(data)