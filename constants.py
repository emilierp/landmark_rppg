"""
    Datasets
"""

def get_dataset_paths(dataset_name):
    if dataset_name == 'mr_nirp':
        dataset_name = 'MR-NIRP_indoor'  
    path = f'D:/datasets_rppg/{dataset_name}'
    return path, path  # dir containing videos,  dir containing BVPs GT

def get_patch_size(datset_name):
    if datset_name == 'mr_nirp':
        return 60
    elif datset_name == 'lgi_ppgi':
        return 40
    else:
        raise NotImplementedError

"""
    Cheat Sheet VS Code
"""

# %load_ext autoreload
# %autoreload 2