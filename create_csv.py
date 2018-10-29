
from sys import argv
from os import listdir, system
from os.path import join, isfile, isdir

def usage():
    pass
    exit()


project_id = ""

def create_csv(project_id, dataset_name, dir_path):
    header = "gs://" + project_id + "-vcm/img/" + dataset_name + "/"
    ret = ""
    for item in listdir(dir_path):
        item_path = join(dir_path, item)
        if isfile(item_path):
            ret += header + item + "\n"

        elif isdir(item_path):
            for file_name in listdir(item_path):
                file_path = join(item_path, file_name)
                if not isfile(file_path):
                    continue
                ret += header + join(item, file_name) + "," + item + "\n"
    with open("/tmp/" + dataset_name + ".csv", "w") as f:
        f.write(ret)


def DIR(dir_path):
    if dir_path[-1] == '/':
        return dir_path[:-1]
    else:
        return dir_path
            
            
def upload_data(project_id, dataset_name, dir_path, csv_path):
    system('gsutil cp ' + csv_path + ' gs://' + project_id + '-vcm/csv/')
    remote_data_path = 'gs://' + project_id + '-vcm/csv/' + dataset_name + '.csv' 
    s= ('gsutil -m cp -r ' + DIR(dir_path) + ' gs://' + project_id + '-vcm/img/' + dataset_name)
    print(s)
    system(s)
    return remote_data_path
    

if __name__ == "__main__":
    #if len(argv) < 3:
    #    usage()
    project_id = argv[3]
    create_csv(argv[1], argv[2])
    upload_data(argv[1], argv[2], "/tmp/" + argv[1] + '.csv')
        

