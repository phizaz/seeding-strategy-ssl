import subprocess

import os

list_file = './datasets-list.txt'
storage_path = './datasets'

files = []
with open(list_file) as f:
    files = f.readlines()

def remove_newline(files):
    return map(
            lambda x: x[:-1],
            files)

files = remove_newline(files)

def wget(files):
    for file in files:
        dataset_name = (file.split(sep="/"))[-2]
        print('name:', dataset_name)
        output_path = storage_path + '/' + dataset_name


        if os.path.exists(output_path):
            continue

        subprocess.call(['/usr/local/bin/wget',
                         '-r', '-nd', '-np',
                         '-P', output_path,
                         file])

wget(files)

print(list(files))