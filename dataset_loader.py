import subprocess
from concurrent.futures import ThreadPoolExecutor

import os

list_file = './datasets-list.txt'
storage_path = './datasets'

files = []
with open(list_file) as f:
    files = f.readlines()

def remove_newline(files):
    return map(
            lambda x: x.strip(),
            files)

files = list(remove_newline(files))

def load_url(url, to):
    subprocess.call(['/usr/local/bin/wget',
                     '-r', '-nd', '-np',
                     '-P', to,
                     url])

def wget(line):
    if len(line) == 0:
        return

    name, url = line.split(sep=' ')
    # print('name:', name)
    # print('url:', url)
    output_path = storage_path + '/' + name
    if os.path.exists(output_path):
        return

    load_url(url, output_path)

def wget_all(files):
    # download all of them concurrently
    with ThreadPoolExecutor(max_workers=len(files)) as executor:
        executor.map(wget, files)

wget_all(files)
print('all datasets are loaded! count:', len(files))