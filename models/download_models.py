import os
import json
import boto3
from glob import glob
from zipfile import ZipFile

# license json path
LICENSE_JSON = "/home/ubuntu/hasham/jsl_keys.json"

# download path
MODELS_DIR = "/home/ubuntu/hasham/temp_models"

# json file that contains models to be downloaded
INPUT_JSON = "./models_list.json"

# source bucket where models are present
BUCKET = "auxdata.johnsnowlabs.com"


if not os.path.isdir(MODELS_DIR):
    print('The directory is not present. Creating a new one..')
    os.mkdir(MODELS_DIR)
else:
    print('The directory is present.')

with open(LICENSE_JSON, 'r') as f_:
    keys_json = json.load(f_)

def download_models():
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=keys_json["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=keys_json["AWS_SECRET_ACCESS_KEY"]
    )
    bucket = s3.Bucket(BUCKET)
    with open(INPUT_JSON) as f:
        models = json.load(f)

    for model_name, model_info in models.items():
        print(f"Downloading {model_name}",)
        bucket.download_file(
            model_info["version"],
            os.path.join(MODELS_DIR, model_info["version"].rsplit("/")[-1])
        )


def unzip_models():
    for _file in glob(f"{MODELS_DIR}/*.zip"):
        print(f'Unzipping {_file}')
        dir_to_extract = _file.rsplit('.', 1)[0]
        with ZipFile(_file) as zip_ref:
            zip_ref.extractall(dir_to_extract)
        os.remove(_file)


if __name__ == '__main__':
    print('DOWNLOAD BEGINS....')
    download_models()
    print('UNZIP BEGINS....')
    unzip_models()