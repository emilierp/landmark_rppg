import numpy as np
import pyVHR

# list of eliminated subjects due to bad quality PPG signal
eliminated_subjects = ['s14', 's15', 's21', 's22', 's24', 's26', 's33', 's35', 's38', 's4', 's48', 's54', 's55', 's56']

def get_dataset_paths(dataset_name):
    """
    Returns the (videos dir, BVPs GT dirs) of the dataset (change path if necessary)
    """
    if dataset_name == 'mr_nirp' or dataset_name == 'MR_NIRP':
      dataset_name = 'MR-NIRP_indoor'
    if dataset_name == 'ubfc_phys':
       dataset_name = 'UBFC-Phys'
    path = f'D:/datasets_rppg/{dataset_name}'

    return path, path

def get_subject_name(dataset_name, videoFileName):  
    """
    Returns the video subject's name of the dataset
    Args:
        dataset_name (str): name of the dataset (mr_nirp, lgi_ppgi, ubfc_phys). Add if necessary.
        videoFileName (str): name of the video file witg path.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'mr_nirp': 
        videoFileName = videoFileName.split('\\')[-1].split('.')[0] # ex: Subject1_motion_940
    elif  dataset_name == 'ubfc_phys': 
        videoFileName = videoFileName.split('\\')[-1].split('.')[0].split('_')[1] # ex: s1
    elif dataset_name == 'lgi_ppgi': 
        videoFileName = videoFileName.split('\\')[2]  # ex: alex_gym
    else:
        raise Exception('videoFileName for dataset not supported', dataset_name)
    
    return videoFileName

def get_video_settings(dataset_name):
    """
    Get the settings of the videos in the dataset (subject has beard, glasses, is female, male, etc.)
    Args:
        dataset_name (str): name of the dataset (mr_nirp, lgi_ppgi, ubfc_phys). Add if necessary.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'mr_nirp':
        videos = {
            'MOTION': [0, 2, 4, 6, 9, 11, 13],
            'STILL': [1, 3, 5, 7, 8, 10, 12, 14],
            'MALE': [0, 1, 4, 5, 6, 7, 8, 9, 10, 13, 14],
            'FEMALE': [2, 3, 11, 12],
            'BEARD': [0, 1, 4, 5, 8, 9, 10, 13, 14],
            'DARK': [0, 1, 2, 3, 4, 5, 13, 14]
        }
    elif dataset_name == 'lgi_ppgi':
        videos = {
            'GYM': np.arange(0, 23, 4),
            'STILL': np.arange(1, 23, 4),
            'ROTATION': np.arange(2, 23, 4),
            'TALK': np.arange(3, 23, 4),
            'MALE': np.arange(4, 24, 1),
            'FEMALE': [0, 1, 2, 3],
            'BEARD': np.arange(4, 24, 1),
            'GLASSES': [0, 1, 2, 3]
        }
    elif dataset_name == 'ubfc_phys':
        videos = {
            'STILL': np.arange(0, 56*3, 3),
            'MALE': [0, 30, 33, 51, 63, 84, 93, 123, 126, 129],
            'FEMALE': [3, 6, 9, 12, 15, 18, 21, 24, 27, 36, 39, 42, 45, 48, 54, 57, 60, 66, 69, 72, 75, 78, 81, 87, 90, 96, 99, 102, 105, 108, 111, 114, 117, 120, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165],
            'GLASSES': [18, 21, 24, 30, 69, 81, 84, 99, 108, 117, 156, 159, 165],
            'BEARD': [30, 93, 126, 129],
            'DARK': [90, 126, 135, 153, 159],
            'BANG': [54, 57, 63, 69, 123, 138, 141, 162]
        }

    return videos
    
def get_fps(dataset_name):
    """
    Returns the videoFPS, sigFPS of the dataset    
    """
    if dataset_name.lower() == 'mr_nirp':
        return pyVHR.datasets.mr_nirp.MR_NIRP.frameRate, pyVHR.datasets.mr_nirp.MR_NIRP.SIG_SampleRate
    elif dataset_name.lower() == 'lgi_ppgi':
        return pyVHR.datasets.lgi_ppgi.LGI_PPGI.frameRate, pyVHR.datasets.lgi_ppgi.LGI_PPGI.SIG_SampleRate
    elif dataset_name.lower() == 'ubfc_phys':
       return pyVHR.datasets.ubfc_phys.UBFC_PHYS.frameRate, pyVHR.datasets.ubfc_phys.UBFC_PHYS.SIG_SampleRate
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")