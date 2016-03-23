from os import listdir
from os.path import isfile, join

path = './datasets/auslan/tctodd'
subfolders = [
    'tctodd1',
    'tctodd2',
    'tctodd3',
    'tctodd4',
    'tctodd5',
    'tctodd6',
    'tctodd7',
    'tctodd8',
    'tctodd9',
]

def files_in_folder(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]

    folder_result = []
    for file in files:
        label, *_ = file.split('-')

        result = []
        for line in open(join(folder, file)):
            line = line.strip()
            elements = line.split('\t')
            elements.append(label)
            result.append(' '.join(elements))

        folder_result += result

    return folder_result

for subfolder in subfolders:
    folder_result = files_in_folder(join(path, subfolder))
    with open(join(path, subfolder + '.txt'), 'w') as file:
        for line in folder_result:
            file.write(line + '\n')

print('finished !')