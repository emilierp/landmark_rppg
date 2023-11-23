"""
    Datasets
"""
import numpy as np

def get_dataset_paths(dataset_name):
    if dataset_name == 'mr_nirp':
      dataset_name = 'MR-NIRP_indoor'
    path = f'D:/datasets_rppg/{dataset_name}'
    return path, path,  # dir containing videos,  dir containing BVPs GT

def get_video_settings(dataset_name):
    if dataset_name == 'mr_nirp':
      videos = {'MOTION':[0,2,4,6,9,11,13],'STILL':[1,3,5,7,8,10,12]}
    elif dataset_name == 'lgi_ppgi':
      videos = {'GYM':np.arange(0,23, 4),'RESTING':np.arange(1,23, 4),'ROTATION':np.arange(2,23, 4),'TALK':np.arange(3,23, 4)}
    
    return videos

def get_patch_size(datset_name):
    if datset_name == 'mr_nirp':
        return 60
    elif datset_name == 'lgi_ppgi':
        return 40
    else:
        raise NotImplementedError

def get_rois():
  return {
    'forehead': [   
        'lower_medial_forehead','glabella','left_lower_lateral_forehead','right_lower_lateral_forehead'
      ],
  'nose': [
      'upper_nasal_dorsum','lower_nasal_dorsum','left_mid_nasal_sidewall','right_mid_nasal_sidewall','left_lower_nasal_sidewall',
      'right_lower_nasal_sidewall','nasal_tip','soft_triangle','left_ala','right_ala'
    ],
    'cheeks':[
      'left_malar','right_malar', 'left_lower_cheek','right_lower_cheek'
    ],
    'jaw':[
      'left_marionette_fold','right_marionette_fold','chin'
    ],
    'temple':[
      'left_temporal','right_temporal'
    ],
    'mustache':[
      'left_nasolabial_fold','right_nasolabial_fold','left_upper_lip','right_upper_lip','philtrum'
    ],
  }

"""
    Cheat Sheet VS Code
"""

# %load_ext autoreload
# %autoreload 2