#!/usr/bin/env python3

from google.cloud import automl_v1beta1 as automl
from google.cloud.automl_v1beta1 import enums

client = automl.AutoMlClient()
automl_client = client

def create_dataset(project_id, compute_region, dataset_name, multilabel = False):
    project_location = client.location_path(project_id, compute_region)
    print()
    print("[create dataset]")

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
    print()
    print("[import data]")
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
    print()
    print("[train model]")
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
    print()
    print("[check operation]")
    
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

def display_evaluation(project_id, compute_region, model_id, filter_ = ""):
    print()
    print("[display evaluation]")
    model_full_id = client.model_path(project_id, compute_region, model_id)

    # List all the model evaluations in the model by applying filter.
    response = client.list_model_evaluations(model_full_id, filter_)

    # Iterate through the results.
    for element in response:
        # There is evaluation for each class in a model and for overall model.
        # Get only the evaluation of overall model.
        if not element.annotation_spec_id:
            model_evaluation_id = element.name.split("/")[-1]

    # Resource name for the model evaluation.
    model_evaluation_full_id = client.model_evaluation_path(
        project_id, compute_region, model_id, model_evaluation_id
    )

    # Get a model evaluation.
    model_evaluation = client.get_model_evaluation(model_evaluation_full_id)

    class_metrics = model_evaluation.classification_evaluation_metrics
    confidence_metrics_entries = class_metrics.confidence_metrics_entry

    # Showing model score based on threshold of 0.5
    for confidence_metrics_entry in confidence_metrics_entries:
        if confidence_metrics_entry.confidence_threshold == 0.5:
            print("Precision and recall are based on a score threshold of 0.5")
            print(
                "Model Precision: {}%".format(
                    round(confidence_metrics_entry.precision * 100, 2)
                )
            )
            print(
                "Model Recall: {}%".format(
                    round(confidence_metrics_entry.recall * 100, 2)
                )
            )
            print(
                "Model F1 score: {}%".format(
                    round(confidence_metrics_entry.f1_score * 100, 2)
                )
            )
            print(
                "Model Precision@1: {}%".format(
                    round(confidence_metrics_entry.precision_at1 * 100, 2)
                )
            )
            print(
                "Model Recall@1: {}%".format(
                    round(confidence_metrics_entry.recall_at1 * 100, 2)
                )
            )
            print(
                "Model F1 score@1: {}%".format(
                    round(confidence_metrics_entry.f1_score_at1 * 100, 2)
                )
            )

    # [END automl_vision_display_evaluation]


def predict(
    project_id, compute_region, model_id, file_path, score_threshold="0.5"
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


