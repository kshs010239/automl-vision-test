#!/usr/bin/env python3

# TODO(developer): Uncomment and set the following variables



from google.cloud import automl_v1beta1 as automl
from google.cloud.automl_v1beta1 import enums
from sys import argv
from create_csv import create_csv, upload_data
from time import sleep
import os
import json

client = automl.AutoMlClient()
automl_client = client

# A resource that represents Google Cloud Platform location.
def read_config():
    config = dict()
    try:
        with open(argv[2], "r") as f:
            for line in f:
                if ':' not in line:
                    continue
                name, data = line.split(':')[:2]
                data = data.split()
                if len(data) > 0:
                    config[name] = data[0] 
    except IOError:
        print('File "', argv[2], '" is not found.', sep = "")
        exit()

    return config 

def get(Dict, key, default = None):
    if key not in Dict or Dict[key] == "":
        return default
    if Dict[key] == "True":
        return True
    return Dict[key]

def main():
    if len(argv) < 3:
        print("automl_vision.py: invalid argv")
        exit()

    if argv[1] == 'config':
        with open(argv[2], "w") as f:
            print("*** Don't input whitespace. ***")
            print("[Required]")
            required = ["project_id: ", "dataset_name: "]
            for item in required:
                while 1:
                    answer = input(item)
                    if len(answer.split()) > 0:
                        f.write(item + answer + "\n")
                        break
            required2 = ["multilabel"]
            for item in required2:
                answer = input(item + "?(y/n):")
                if len(answer.split()) > 0 and (answer[0] == "y" or answer[0] == "Y"):
                    f.write(item + ": True\n")


            print()
            print("[Optional]")
            print("[enter] will use default setting.")
            optional = ["model_name: "]
            for item in optional:
                answer = input(item)
                f.write(item + answer + "\n")

        exit()

    config = read_config()
    project_id = get(config, "project_id")
    compute_region = get(config, "compute_region", "us-central1")
    dataset_name = get(config, "dataset_name")
    model_name = get(config, "model_name", dataset_name)
    model_id = get(config, "model_id")
    multilabel = get(config, "multilabel", False)
    
    if argv[1] == 'create':
        data_path = argv[3]

        dataset_id = create_dataset(project_id, compute_region, dataset_name, multilabel)

        create_csv(project_id, dataset_name, data_path)
        remote_data_path = upload_data(project_id, dataset_name, data_path, "/tmp/" + dataset_name + '.csv')
        
        import_data(project_id, compute_region, dataset_id, remote_data_path)
        model_id = train_model(project_id, compute_region, dataset_id, dataset_name)
        with open(argv[2], "a") as f:
            f.write("model_id: " + model_id + "\n")

        evaluate_model()
   
    elif argv[1] == 'check':
        operation_id = argv[2]
        check_op(operation_id)


    elif argv[1] == 'predict':
        data_path = argv[3]
        if os.path.isfile(data_path):
            predict(project_id, compute_region, model_id, data_path)
            
        else:
            for root, dir, files in os.walk(data_path):
                for f in files:
                    file_path = os.path.join(root, f)
                    if len(file_path) == 0 or not os.path.isfile(file_path):
                        continue
                    predict(project_id, compute_region, model_id, file_path)
    else:
        print('Command "' + argv[1] + '" not found')



def create_dataset(project_id, compute_region, dataset_name, multilabel = False):
    project_location = client.location_path(project_id, compute_region)

    # Classification type is assigned based on multilabel value.
    classification_type = "MULTICLASS"
    if multilabel:
        classification_type = "MULTILABEL"

    # Specify the image classification type for the dataset.
    dataset_metadata = {"classification_type": classification_type}
    # Set dataset name and metadata of the dataset.
    my_dataset = {
    "display_name": dataset_name,
    "image_classification_dataset_metadata": dataset_metadata,
    }

    # Create a dataset with the dataset metadata in the region.
    dataset = client.create_dataset(project_location, my_dataset)
    global dataset_id
    dataset_id = dataset.name.split("/")[-1]

    # Display the dataset information.
    print("Dataset name: {}".format(dataset.name))
    print("Dataset id: {}".format(dataset_id))
    print("Dataset display name: {}".format(dataset.display_name))
    print("Image classification dataset metadata:")
    print("\t{}".format(dataset.image_classification_dataset_metadata))
    print("Dataset example count: {}".format(dataset.example_count))
    print("Dataset create time:")
    print("\tseconds: {}".format(dataset.create_time.seconds))
    print("\tnanos: {}".format(dataset.create_time.nanos))
    return dataset_id   


def import_data(project_id, compute_region, dataset_id, path):
    # Get the full path of the dataset.
    dataset_full_id = client.dataset_path(
        project_id, compute_region, dataset_id
    )

    # Get the multiple Google Cloud Storage URIs.
    input_uris = path.split(",")
    input_config = {"gcs_source": {"input_uris": input_uris}}

    print(dataset_full_id)
    print(input_config)


    response = client.import_data(dataset_full_id, input_config)

    print("Processing import...")
    print("Data imported. {}".format(response.result()))


def train_model(
    project_id, compute_region, dataset_id, model_name, train_budget = 1, wait_for_train = True
):
    project_location = client.location_path(project_id, compute_region)
    # Set model name and model metadata for the image dataset.
    my_model = {
        "display_name": model_name,
        "dataset_id": dataset_id,
        "image_classification_model_metadata": {"train_budget": train_budget}
        if train_budget
        else {},
    }

    response = client.create_model(project_location, my_model)
    op = response.operation


    print("Training operation name: {}".format(response.operation.name))
    print("Training started")
    print("operation id: ", response.operation.name)
    if wait_for_train:
        model_id = str(response.result()).split('/')[-1][:-2]
        print("model id:", model_id)
        return model_id



def check_op(operation_id):
    output = os.popen('curl \
      -X GET \
      -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
      -H "Content-Type: application/json" \
      https://automl.googleapis.com/v1beta1/' + operation_id + ' 2>/dev/null').read()

    if 'error' in output:
        print(output)
        return 

    output = json.loads(output)
    if 'error' in output:
        print(output)
        return 

    if 'done' in output:
        print('Operation done.')
        model_id = output['response']['name'].split('/')[-1]
        print('model id:', model_id)
    else:
        print('Not done yet.')


def evaluate_model():
    # TODO
    pass

def predict(
    project_id, compute_region, model_id, file_path, score_threshold="0.1"
):
    model_full_id = automl_client.model_path(
        project_id, compute_region, model_id
    )

    prediction_client = automl.PredictionServiceClient()

    with open(file_path, "rb") as image_file:
        content = image_file.read()
    payload = {"image": {"image_bytes": content}}

    params = {}
    if score_threshold:
        params = {"score_threshold": score_threshold}
    
    response = prediction_client.predict(model_full_id, payload, params)
    
    '''
    # full result
    print("Prediction results:")
    for result in response.payload:
        print("Predicted class name: {}".format(result.display_name))
        print("Predicted class score: {}".format(result.classification.score))
    '''

    # csv result
    output = []
    for result in response.payload:
        output.append((result.classification.score, result.display_name))

    output.sort(reverse = True)
    print(file_path, end = '')
    for result in output:
        print(",", result[1], end = '')
    print()

if __name__ == "__main__" :
    main()
    

