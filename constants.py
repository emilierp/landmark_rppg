"""
    Datasets
"""
import numpy as np

def get_dataset_paths(dataset_name):
    if dataset_name == 'mr_nirp' or dataset_name == 'MR_NIRP':
      dataset_name = 'MR-NIRP_indoor'
    if dataset_name == 'ubfc_phys':
       dataset_name = 'UBFC-Phys'
    path = f'D:/datasets_rppg/{dataset_name}'
    return path, path,  # dir containing videos,  dir containing BVPs GT

def get_video_settings(dataset_name):
    if dataset_name.lower() == 'mr_nirp':
      videos = {'MOTION':[0,2,4,6,9,11,13],'STILL':[1,3,5,7,8,10,12,14],
                'MALE':[0,1,4,5,6,7,8,9,10,13,14],'FEMALE':[2,3,11,12],
                'BEARD':[0,1,4,5,8,9,10,13,14],'DARK':[0,1,2,3,4,5,13,14]}
    elif dataset_name.lower() == 'lgi_ppgi':
      videos = {'GYM':np.arange(0,23, 4),'STILL':np.arange(1,23, 4),'ROTATION':np.arange(2,23, 4),'TALK':np.arange(3,23, 4),
                'MALE':np.arange(4,24,1), 'FEMALE':[0,1,2,3],'BEARD':np.arange(4,24,1),'GLASSES':[0,1,2,3]}
    elif dataset_name.lower() == 'ubfc_phys':
       videos = {
          'STILL':[3, 9, 15, 21, 27, 30, 33, 36, 39, 45, 57, 60, 63, 66, 72, 75, 84, 87, 90, 93, 96, 102, 105, 108, 111, 114, 117, 120, 123, 132, 135, 138, 156, 159],
          'MALE':[0, 30, 33, 51, 63, 84, 93, 123, 126, 129],
          'FEMALE':[3, 6, 9, 12, 15, 18, 21, 24, 27, 36, 39, 42, 45, 48, 54, 57, 60, 66, 69, 72, 75, 78, 81, 87, 90, 96, 99, 102, 105, 108, 111, 114, 117, 120, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165],
          'GLASSES':[18, 21, 24, 30, 69, 81, 84, 99, 108, 117, 156, 159, 165],
          # 'ROTATION':[0, 12, 24, 48, 54, 69, 81, 150, 153, 165], # only by video inspection
          'ROTATION': [0, 6, 12, 18, 24, 42, 48, 51, 54, 69, 78, 81, 99, 126, 129, 141, 144, 147, 150, 153, 162, 165],
          'BEARD':[30, 93, 126, 129], # [30, 33, 93, 126, 129]
          'DARK':[90, 126, 135, 153, 159],
          'BANG':[54, 57, 63, 69, 123, 138, 141, 162] # [54, 57, 63, 69, 123, 138, 141, 162]
          }
    
    return videos

# TODO 
def get_patch_size(datset_name):
    if datset_name == 'mr_nirp':
        return 60
    elif datset_name == 'lgi_ppgi':
        return 40
    elif datset_name == 'ubfc_phys':
        return 60
    else:
        raise NotImplementedError
    
def get_fps(dataset_name):
    """
        Returns the videoFPS, sigFPS of the dataset    
    """
    if dataset_name.lower() == 'mr_nirp':
        return 30, 60
    elif dataset_name.lower() == 'lgi_ppgi':
        return 25, 60
    elif dataset_name.lower() == 'ubfc_phys':
       return 35.138, 64
    else:
        raise NotImplementedError

def get_lgi_ppgi_rotation_segments():
  times_rot = {
    'alex_rotation': np.concatenate([np.arange(23,32),np.arange(51,59)]),
    'angelo_rotation': np.concatenate([np.arange(23,29),np.arange(41,48)]),
    'cpi_rotation': np.concatenate([np.arange(20,26),np.arange(41,48), np.arange(60,67)]),
    'david_rotation': np.concatenate([np.arange(20,25),np.arange(36,40), np.arange(48,53)]),
    'felix_rotation': np.concatenate([np.arange(21,28),np.arange(39,46), np.arange(48,53)]),
    'harun_rotation': np.concatenate([np.arange(18,28),np.arange(40,51)]),
  }
  return times_rot

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


def get_palette(palette_set='Spectral'):
    """
      Palette for each individual landmark (28) ordered by ROI color
    """
    palette = {
      'left_lower_cheek': (0.6872741253364091,0.07896962706651288,0.2748173779315648),
      'left_malar': (0.7633986928104575, 0.1633986928104575, 0.2928104575163399),
      'right_lower_cheek': (0.8376778162245291,0.2467512495194156,0.308881199538639),
      'right_malar': (0.8805843906189927, 0.3118031526336025, 0.2922722029988466),
      'glabella': (0.9234909650134564, 0.3768550557477893, 0.27566320645905423),
      'left_lower_lateral_forehead': (0.958246828143022,0.43744713571703187,0.267358708189158),
      'lower_medial_forehead': (0.9707035755478662, 0.5274125336409072, 0.3088811995386389),
      'right_lower_lateral_forehead': (0.9831603229527105,  0.6173779315647827,  0.3504036908881199),
      'chin': (0.9925413302575933, 0.7015763168012302, 0.39653979238754317),
      'left_marionette_fold': (0.993925413302576, 0.7707804690503652, 0.4546712802768165),
      'right_marionette_fold': (0.9953094963475586,0.8399846212995001,0.5128027681660899),
      'left_nasolabial_fold': (0.9965397923875433, 0.8927335640138409, 0.5690888119953863),
      'left_upper_lip': (0.9979238754325259, 0.9356401384083044, 0.6410611303344866),
      'philtrum': (0.9993079584775086, 0.9785467128027682, 0.7130334486735871),
      'right_nasolabial_fold': (0.9826989619377164, 0.9930795847750865, 0.7220299884659749),
      'right_upper_lip': (0.9480968858131489,0.9792387543252595,0.6680507497116495),
      'left_ala': (0.9134948096885814, 0.9653979238754326, 0.6140715109573243),
      'left_lower_nasal_sidewall': (0.8565936178392929,0.942329873125721,0.6053056516724337),
      'left_mid_nasal_sidewall': (0.7749327181853136, 0.9091118800461362, 0.6219146482122261),
      'lower_nasal_dorsum': (0.6932718185313343,0.8758938869665515,0.6385236447520184),
      'nasal_tip': (0.60161476355248, 0.8396770472895041, 0.6441368704344483),
      'right_ala': (0.5061130334486736, 0.8023068050749712, 0.6455209534794311),
      'right_lower_nasal_sidewall': (0.4106113033448674,0.7649365628604383,0.6469050365244137),
      'right_mid_nasal_sidewall': (0.34402153018069975,0.6983467896962706,0.6728950403690889),
      'soft_triangle': (0.27204921184159936, 0.6180699730872741, 0.7061130334486736),
      'upper_nasal_dorsum': (0.20007689350249905, 0.5377931564782776, 0.7393310265282584),
      'left_temporal': (0.25359477124183005,0.45882352941176474, 0.7058823529411765),
      'right_temporal': (0.31449442522106885,0.37993079584775086,0.6685121107266436),
      }
    
    return palette

"""
    Cheat Sheet VS Code
"""

# %load_ext autoreload
# %autoreload 2