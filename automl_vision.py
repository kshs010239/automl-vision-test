#!/usr/bin/env python3

from sys import argv
from time import sleep
import os
import json

from create_csv import create_csv, upload_data
from automl_vision_function import *


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

def check_argv(n):
    if len(argv) < n:
        print("automl_vision.py: invalid argv")
        exit() 

def append_config(content):
    with open(argv[2], "a") as f:
        f.write(content + "\n")
    

def main():
    check_argv(2)

    if argv[1] == 'config':
        create_config()
        exit()


    config = read_config()
    project_id = get(config, "project_id")
    compute_region = get(config, "compute_region", "us-central1")
    dataset_name = get(config, "dataset_name")
    model_name = get(config, "model_name", dataset_name)
    model_id = get(config, "model_id")
    multilabel = get(config, "multilabel", False)
    
    if argv[1] == 'create':
        check_argv(4)
        data_path = argv[3]

        dataset_id = create_dataset(project_id, compute_region, dataset_name, multilabel)

        create_csv(project_id, dataset_name, data_path)
        remote_data_path = upload_data(project_id, dataset_name, data_path, "/tmp/" + dataset_name + '.csv')
        
        import_data(project_id, compute_region, dataset_id, remote_data_path)
        
        model_id = train_model(project_id, compute_region, dataset_id, dataset_name)
        append_config("model_id: " + model_id + "\n")
        
        display_evaluation(project_id, compute_region, model_id);
   
    elif argv[1] == 'check':
        operation_id = argv[2]
        check_op(operation_id)


    elif argv[1] == 'predict':
        check_argv(4)
        data_path = argv[3]
        Predict(project_id, compute_region, model_id, data_path)
    else:
        print('Command "' + argv[1] + '" not found')


def Predict(project_id, compute_region, model_id, data_path):
    print()
    print("[predict]")
    if os.path.isfile(data_path):
        predict(project_id, compute_region, model_id, data_path)
        
    else:
        for root, dir, files in os.walk(data_path):
            for f in files:
                file_path = os.path.join(root, f)
                if len(file_path) == 0 or not os.path.isfile(file_path):
                    continue
                predict(project_id, compute_region, model_id, file_path)
    

def create_config():
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

if __name__ == "__main__" :
    main()
    

