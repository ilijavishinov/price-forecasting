import json
import os
from typing import Dict

def ensure_dir_exists(dir_path,
                      check_parent_dir = False):
    """
    Creates a directory if it does not exist
    Can be made recursve if needed
    """
    os.makedirs(dir_path, exist_ok = True)
    
    return dir_path


def save_json_metadata(metadata_dir = None,
                       file_name = None,
                       data = None):
    """
    Save json file
    """
    if not file_name.endswith('.json'):
        file_name += '.json'
    
    if metadata_dir:
        ensure_dir_exists(metadata_dir)
        with open(os.path.join(metadata_dir, file_name), 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii = False)


def rename_columns(columns,
                   rename_dict: Dict):
    """
    Rename columns in a pd.DataFrame
    """
    for i in range(len(columns)):
        if columns[i] in rename_dict.keys():
            columns[i] = rename_dict[columns[i]]
    return columns


    










