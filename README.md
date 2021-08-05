# bms-treatment

This repo contains the [CTDI Treatment](ctdi_treatment) module. In it you will find
how to build a pipeline and execute an NLP Pipeline with the
following tasks on excerpts of natural language text:
- Clinical Word Embeddings
- Named Entity Recognition for Drugs and Treatments
  - Regex Matcher
  - Text Matcher
  - Deep Learning NER
- Assertion for Treatments (Past/Present/Absent...)
- RxNorm Entity Resolution for Drugs and their `has_disposition` relationships

It also contains [a notebook](notebooks/20210821_Experiment_ARM.ipynb) to understand
how to use the module for two particular use cases:
- Run the Pipeline on ARM and Detailed Intervention information from AACT Database and parsing it into
a revised Treatment taxonomy while complementing the objects at Intervention level
with the information at ARM level. The output of this use case is a `csv` with some columns in `json`.
- Run the Pipeline on ARM and Aggregated Intervention information from AACT Database and preannotate it
in order to upload it to the Annotation Lab. The output of this use case is a `json`.

In the [models](models/editable) folder you can adjust the behavior of the following models:
- RegexMatcher: Use the [arms_regex.csv](models/editable/arms_regex.csv) to add additional regex patterns
- TextMatcher: Use the [arms_treatment_textmatcher.csv](models/editable/arms_regex.csv) to add additional keyword paths
- NER Output: Use the [fp_dict.csv](models/editable/arms_regex.csv) to prevent the inclusion of certain known false positives
