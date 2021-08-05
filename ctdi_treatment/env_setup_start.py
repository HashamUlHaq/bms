import os
import json

import sparknlp
import sparknlp_jsl


def set_envvars(keys_path):
                
    with open(keys_path, 'r') as f:
        license_keys = json.load(f)
    license_keys.keys()

    secret = license_keys['SECRET']
    os.environ['SPARK_NLP_LICENSE'] = license_keys['SPARK_NLP_LICENSE']
    #os.environ['JSL_OCR_LICENSE'] = license_keys['JSL_OCR_LICENSE']
    os.environ['AWS_ACCESS_KEY_ID'] = license_keys['AWS_ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY'] = license_keys['AWS_SECRET_ACCESS_KEY']
    sparknlp_version = license_keys["PUBLIC_VERSION"]
    jsl_version = license_keys["JSL_VERSION"]
    print ('Desired SparkNLP Version:', sparknlp_version)
    print ('Desired SparkNLP-JSL Version:', jsl_version)
    return license_keys


def start_sparknlp(license_keys, params=None):
    if params is None:
        params = {"spark.driver.memory":"16G",
        "spark.kryoserializer.buffer.max":"2000M",
        "spark.driver.maxResultSize":"2000M"}

    spark = sparknlp_jsl.start(license_keys['SECRET'],params=params)

    print ("Real Spark NLP Version :", sparknlp.version())
    print ("Real Spark NLP_JSL Version :", sparknlp_jsl.version())
    return spark
