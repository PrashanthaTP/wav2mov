import os
from tqdm import tqdm

FROM_DIR = r""
TO_DIR = r""
FILENAME = 'filenames.txt'

if len(FROM_DIR)==0:
    FROM_DIR = os.path.dirname(os.path.abspath(__file__))
    FROM_DIR = os.path.join(FROM_DIR,'grid_dataset')
    TO_DIR = FROM_DIR

def get_folders_list():
    folders = [folder for _,folder,_ in os.walk(FROM_DIR)][0]
    folders_strs = []
    for folder in tqdm(folders):
        if os.path.isdir(os.path.join(FROM_DIR,folder)):
            folders_strs.append(folder+'\n')
    return folders_strs
    
def write_to_text_file(folders_list):
    with open(os.path.join(TO_DIR,FILENAME),'w') as file:
        file.writelines(folders_list)
    print(f'list written to {os.path.join(TO_DIR,FILENAME)}')
    
if __name__ == '__main__':
    folders = get_folders_list()
    TO_DIR = os.path.dirname(FROM_DIR)
    write_to_text_file(folders)