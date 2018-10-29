#!/usr/bin/env python3

# TODO(developer): Uncomment and set the following variables


project_id = 'PROJECT_ID_HERE'
compute_region = 'COMPUTE_REGION_HERE'
dataset_name = 'DATASET_NAME_HERE'
dataset_id = 0
data_path = ""
multilabel = 0#True for multilabel or False for multiclass
model_id = 0

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

def require():
    #project_id = input('PROJECT_ID: ')
    #compute_region = input('COMPUTE_REGION: ')
    #dataset_name = input('DATASET_NAME: ')
    #multilabel = input('MULTILABLE?(y/n): ')[0] == 'y' ? True : False
    if len(argv) < 6:
        print("invalid argu")
        exit()
    global project_id, compute_region, dataset_name, data_path
    project_id = argv[2]
    compute_region = argv[3]
    dataset_name = argv[4]
    data_path = argv[5]


def main():
    global project_id, compute_region, model_id, data_path
    if len(argv) < 3:
        print("automl_vision.py: invalid argv")
        exit()

    if argv[1] == 'init':
        require()
        print("start")
        create_csv(project_id, dataset_name, data_path)
        remote_data_path = upload_data(project_id, dataset_name, data_path, "/tmp/" + dataset_name + '.csv')
        
        create_dataset()
        import_data(remote_data_path)
        operation_id = train_model()
        print("operation_id:", operation_id)

        evaluate_model()
   
    elif argv[1] == 'check':
        operation_id = argv[2]
        check_op(operation_id)


    elif argv[1] == 'predict':
        project_id = argv[2]
        compute_region = argv[3]
        model_id = argv[4]
        data_path = argv[5]
        if os.path.isfile(data_path):
            predict(argv[5])
            
        else:
            for root, dir, files in os.walk(argv[5]):
                for f in files:
                    filename = os.path.join(root, f)
                    if len(filename) == 0 or not os.path.isfile(filename):
                        continue
                    predict(filename)



def create_dataset():
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


def import_data(path):
    # Get the full path of the dataset.
    dataset_full_id = client.dataset_path(
        project_id, compute_region, dataset_id
    )

    # Get the multiple Google Cloud Storage URIs.
    input_uris = path.split(",")
    input_config = {"gcs_source": {"input_uris": input_uris}}

    # Import data from the input URI.
    response = client.import_data(dataset_full_id, input_config)

    print("Processing import...")
    # synchronous check of operation status.
    print("Data imported. {}".format(response.result()))


def train_model():
    project_location = client.location_path(project_id, compute_region)
    model_name = dataset_name
    # Set model name and model metadata for the image dataset.
    train_budget = 1
    my_model = {
        "display_name": model_name,
        "dataset_id": dataset_id,
        "image_classification_model_metadata": {"train_budget": train_budget}
        if train_budget
        else {},
    }

    # Create a model with the model metadata in the region.
    response = client.create_model(project_location, my_model)
    op = response.operation


    print("Training operation name: {}".format(response.operation.name))
    print("Training started...")
    return response.operation.name

def check_op(operation_id):
    output = os.popen('curl \
      -X GET \
      -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
      -H "Content-Type: application/json" \
      https://automl.googleapis.com/v1beta1/' + operation_id + ' 2>/dev/null').read()

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



def check_model(model_id):
	# Get the full path of the model.
	model_full_id = client.model_path(project_id, compute_region, model_id)

	# Get complete detail of the model.
	model = client.get_model(model_full_id)

	# Retrieve deployment state.
	if model.deployment_state == enums.Model.DeploymentState.DEPLOYED:
		deployment_state = "deployed"
	else:
		deployment_state = "undeployed"

	# Display the model information.
	print("Model name: {}".format(model.name))
	print("Model id: {}".format(model.name.split("/")[-1]))
	print("Model display name: {}".format(model.display_name))
	print("Image classification model metadata:")
	print(
		"Training budget: {}".format(
			model.image_classification_model_metadata.train_budget
		)
	)
	print(
		"Training cost: {}".format(
			model.image_classification_model_metadata.train_cost
		)
	)
	print(
		"Stop reason: {}".format(
			model.image_classification_model_metadata.stop_reason
		)
	)
	print(
		"Base model id: {}".format(
			model.image_classification_model_metadata.base_model_id
		)
	)
	print("Model create time:")
	print("\tseconds: {}".format(model.create_time.seconds))
	print("\tnanos: {}".format(model.create_time.nanos))
	print("Model deployment state: {}".format(deployment_state))

def evaluate_model():
    # TODO
    pass

def predict(file_path):
    model_full_id = automl_client.model_path(
        project_id, compute_region, model_id
    )

    # Create client for prediction service.
    prediction_client = automl.PredictionServiceClient()

    # Read the image and assign to payload.
    with open(file_path, "rb") as image_file:
        content = image_file.read()
    payload = {"image": {"image_bytes": content}}

    # params is additional domain-specific parameters.
    # score_threshold is used to filter the result
    # Initialize params
    params = {}
    score_threshold = "0.1"
    if score_threshold:
        params = {"score_threshold": score_threshold}
    
    response = prediction_client.predict(model_full_id, payload, params)
    '''print("Prediction results:")
    for result in response.payload:
        print("Predicted class name: {}".format(result.display_name))
        print("Predicted class score: {}".format(result.classification.score))
    '''
    print(file_path, end = '')
    for result in response.payload:
        print(",", result.display_name, end = '')
    print()

if __name__ == "__main__" :
    main()
    

