import pandas as pd
import numpy as np
import pyVHR as vhr
import scipy
import constants
import os.path
from numpy.lib.arraysetops import isin
from importlib import import_module, util
from pyVHR.datasets.dataset import datasetFactory
from pyVHR.utils.errors import getErrors, printErrors, displayErrors, BVP_windowing
from pyVHR.extraction.sig_processing import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *
from inspect import getmembers, isfunction
from pyVHR.deepRPPG.mtts_can import *
from pyVHR.deepRPPG.hr_cnn import *
from pyVHR.extraction.utils import *
import dtw
from scipy import signal
import dtaidistance as dtwai



def get_dataset(dataset_name, video_DIR=None):
    """
    Return the dataset object
    """
    if video_DIR is None:
        video_DIR, BVP_DIR = constants.get_dataset_paths(dataset_name)
    else:
        _, BVP_DIR = constants.get_dataset_paths(dataset_name)
    dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
    return dataset


def get_signal_data(videoIdx, dataset, dataset_name, winsize, stride=1):
    """
      Read the signal data from the dataset and return the PPG_win, bpmGT, timesGT, videoFileName
    """
    videoFPS, sigFPS = constants.get_fps(dataset_name)
    fname = dataset.getSigFilename(videoIdx)
    videoFileName = dataset.getVideoFilename(videoIdx)
    fps = vhr.extraction.get_fps(videoFileName)
    if dataset_name == 'mr_nirp':
        videoFileName = videoFileName.split('\\')[-1][:-4]  
    elif  dataset_name == 'ubfc_phys':
        videoFileName = videoFileName.split('\\')[-1].split('_')[1]
    elif dataset_name == 'lgi_ppgi':
        videoFileName = videoFileName.split('\\')[2]  
    else:
        raise Exception('videoFileName for dataset not supported', dataset_name)
    # print("videoFileName : ", videoFileName)

    sigGT = dataset.readSigfile(fname)
    bpmGT, timesGT = sigGT.getBPM(winsize) # STFT 42-240 BPM

    PPG = sigGT.data.T.reshape(-1,1,1) # shape (n,nb_ldmk,RGB)
    # PPG = (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG)) # normalize PPG [0,1]
    PPG_win, timesGTwin = sig_windowing(PPG, winsize, stride, sigFPS)
    PPG_win = np.array([PPG_win[i].reshape(-1) for i in range(len(PPG_win))]) # from (n,1,1,m) to (n,m) PPG_win
    # print("Before sampling: ", PPG_win[0].shape)
    if dataset_name == 'mr_nirp' or dataset_name == 'lgi_ppgi':
        PPG_win = [scipy.signal.resample(PPG_win[i], int(fps/sigFPS*PPG_win[i].shape[-1]), axis=0) for i in range(len(PPG_win))]
    elif dataset_name == 'ubfc_phys':
        PPG_win = [scipy.signal.resample(PPG_win[i], np.ceil(fps/sigFPS*PPG_win[i].shape[-1]).astype(int), axis=0) for i in range(len(PPG_win))]
    else:
        raise Exception('PPG sampling for dataset not supported', dataset_name)
    PPG_win = np.array([(ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg)) for ppg in PPG_win])
    # print("After resampling: ", PPG_win[0].shape)   

    return PPG_win, bpmGT, timesGT, videoFileName

def run_extraction(ldmks_list, videoFileName, roi_approach, sampling_method='all', nb_sample_points=1500, roi_method='convexhull', seconds=0, verb=False, winsize=8, stride=1):

    RGB_LOW_HIGH_TH=(0,240)
    Skin_LOW_HIGH_TH=(0, 240)
    cuda = True
    patch_size=40.0

    # test video filename
    assert os.path.isfile(videoFileName), "The video file does not exists!"
    if verb:
        print("Video: ", videoFileName)

    sig_processing = SignalProcessing()

    if cuda and verb:
        # sig_processing.display_cuda_device()
        sig_processing.choose_cuda_device(0)

    start_video = time.time()

    ## 1. set skin extractor
    target_device = 'GPU' if cuda else 'CPU'
    if roi_method == 'convexhull':
        sig_processing.set_skin_extractor(SkinExtractionConvexHull(target_device))
    elif roi_method == 'faceparsing':
        sig_processing.set_skin_extractor(SkinExtractionFaceParsing(target_device))
    else:
        raise ValueError("Unknown 'roi_method'")
            
    ## 2. set patches # CHANGED , suppose only custom landmarks
    if roi_approach == 'patches':
        if len(ldmks_list) == 0: # take all landmarks
            ldmks_list = [
                x for x in list(pyVHR.extraction.CustomLandmarks().__dict__)
            ]
        all_landmarks  = pyVHR.extraction.CustomLandmarks().get_all_landmarks()
        ldmk_values = list(np.unique(sum([all_landmarks[ldmk] for ldmk in ldmks_list], [])))
        sig_processing.set_landmarks(ldmk_values)
        sig_processing.set_square_patches_side(float(patch_size))
    if roi_approach == 'landmark':
        if len(ldmks_list) == 0: # take all landmarks
            ldmks_list = [
                x for x in list(pyVHR.extraction.CustomLandmarks().__dict__)
            ]
        all_landmarks  = pyVHR.extraction.CustomLandmarks().get_all_landmarks()
        ldmk_values = [all_landmarks[ldmk] for ldmk in ldmks_list]
        sig_processing.set_landmarks(ldmk_values)
        sig_processing.set_patch_sampling(sampling_method=sampling_method, nb_sample_points=nb_sample_points)


    if verb:
        print(f"Landmarks list : {ldmks_list}")

    # set sig-processing and skin-processing params
    SignalProcessingParams.RGB_LOW_TH = RGB_LOW_HIGH_TH[0]
    SignalProcessingParams.RGB_HIGH_TH = RGB_LOW_HIGH_TH[1]
    SkinProcessingParams.RGB_LOW_TH = Skin_LOW_HIGH_TH[0]
    SkinProcessingParams.RGB_HIGH_TH = Skin_LOW_HIGH_TH[1]

    if verb:
        print('\nProcessing Video ' + videoFileName)
    fps = get_fps(videoFileName)
    sig_processing.set_total_frames(seconds*fps)

    ## 3. ROI selection
    sig = []
    if roi_approach == 'holistic':
        # SIG extraction with holistic
        sig = sig_processing.extract_holistic(videoFileName)
    elif roi_approach == 'patches':
        # SIG extraction with patches
        sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')
    elif roi_approach == 'landmark':
        # SIG extraction with landmarks
        sig = sig_processing.extract_patches(videoFileName, 'landmark', 'mean')
        
    ## 4. sig windowing
    windowed_sig, timesES = sig_windowing(sig, winsize, stride, fps)
    if verb:
        print(f" - Extraction approach: {roi_approach} with {len(windowed_sig)} windows")
    
    return windowed_sig, timesES

def get_bpm(res, videoFileName, landmarks, windowed_sig,
            timesES, bpmGT, timesGT, PPG_win, videoFPS, sigFPS, dataset_name,
            cuda=True, 
            roi_method='convexhull', 
            roi_approach='landmark', 
            methods=['cupy_CHROM', 'cpu_LGI'], 
            estimate='median', 
            movement_thrs=[10, 5, 2],
            winsize = 8,
            patch_size=30, 
            RGB_LOW_HIGH_TH = (0,240),
            Skin_LOW_HIGH_TH = (0, 240),
            pre_filt=True, 
            post_filt=True, 
            verb=False):
    
    minHz = 0.65
    maxHz = 4.0
    fps = videoFPS

    ## 5. PRE FILTERING
    filtered_windowed_sig = windowed_sig

    if pre_filt:
        module = import_module('pyVHR.BVP.filters')
        method_to_call = getattr(module, 'BPfilter')
        filtered_windowed_sig = apply_filter(filtered_windowed_sig,
                                                method_to_call, 
                                                fps=fps, 
                                                params={'minHz':minHz, 
                                                        'maxHz':maxHz, 
                                                        'fps':'adaptive', 
                                                        'order':6})

    ## 6. BVP extraction multimethods
    bvps_win = []
    for method in methods:
        # try:
        if verb:
            print(" - Extraction method: " + method)
        module = import_module('pyVHR.BVP.methods')
        method_to_call = getattr(module, method)
        
        if 'cpu' in method:
            method_device = 'cpu'
        elif 'torch' in method:
            method_device = 'torch'
        elif 'cupy' in method:
            method_device = 'cuda'

        if 'POS' in method:
            pars = {'fps':'adaptive'}
        elif 'PCA' in method or 'ICA' in method:
            pars = {'component': 'all_comp'}
        else:
            pars = {}

        bvps_win = RGB_sig_to_BVP(filtered_windowed_sig, 
                                fps, device_type=method_device, 
                                method=method_to_call, params=pars)

        ## 7. POST FILTERING
        if post_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps_win = apply_filter(bvps_win, 
                                method_to_call, 
                                fps=fps, 
                                params={'minHz':minHz, 'maxHz':maxHz, 'fps':'adaptive', 'order':6})

        ## 8. BPM extraction

        if roi_approach == 'holistic':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=minHz, maxHz=maxHz)
            else:
                bpmES = BVP_to_BPM(bvps_win, fps, minHz=minHz, maxHz=maxHz)

        elif roi_approach == 'patches' or roi_approach == 'landmark':
            if estimate == 'clustering':
                #if cuda and False:
                #    bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=minHz, maxHz=maxHz)
                #else:
                #bpmES = BPM_clustering(sig_processing, bvps_win, winsize, movement_thrs=[15, 15, 15], fps=fps, opt_factor=0.5)
                ma = MotionAnalysis(sig_processing, winsize, fps)
                bpmES = BPM_clustering(ma, bvps_win, fps, winsize, movement_thrs=movement_thrs, opt_factor=0.5)
                
            elif estimate == 'median':
                if cuda:
                    bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=minHz, maxHz=maxHz)
                else:
                    bpmES = BVP_to_BPM(bvps_win, fps, minHz=minHz, maxHz=maxHz)
        #         if bpmES[0].ndim > 0:
        #             bpmES, MAD = BPM_median(bpmES)
        #         else:
        #             MAD = 0

        else:
            raise ValueError("Estimation approach unknown!")

        # ## 9. error metrics
        # RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps_win, fps, bpmES, bpmGT, timesES, timesGT)

        # # DTW
        # bvps_median = np.median(bvps_win, axis=1) # median of all estimates
        # bvps_max, bvps_min = np.max(bvps_median, axis=1).max(), np.min(bvps_median, axis=1).min()
        # bvps_median = (bvps_median - bvps_min) / (bvps_max - bvps_min) # normalize BVP [0,1]
        # DTW = []
        # CC = []
        # for w in range(min(len(bvps_median), len(PPG_win))):
        #     alignment = dtw.dtw(bvps_median[w], PPG_win[w], keep_internals=True)
        #     DTW.append(alignment.distance)
        #     r, p = stats.pearsonr(PPG_win[w], bvps_median[w])
        #     CC.append(r)
    

    # Aggregate BVP and BPM
    # bvps_win = [[np.array(arr) for arr in bvps_win]]
    BVP_agg = [[np.array(arr) for arr in bvps_win]]
    for i,bvp_win in enumerate(BVP_agg):
        BVP_agg[i] = np.mean(bvp_win, axis=1) # take the mean in case the landmark is symmetrics
    BVP_agg = np.mean(BVP_agg, axis=0) # mean of all landmarks


    bpmES_agg = [[np.array(arr) for arr in bpmES]]
    for i, bpm in enumerate(bpmES_agg):   
        if np.array(bpm).ndim >1: # take the mean in case the landmark is symmetrics
            bpmES_agg[i] = np.mean(bpm, axis=1)
        else: # need to reduce the dimension
            bpmES_agg[i] = np.array(bpm)
    bpmES_agg = np.mean(bpmES_agg, axis=0) # mean of all landmarks

    # BPM Error metrics
    RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps_win, videoFPS, bpmES_agg, bpmGT, timesES, timesGT)

    # BVP Error metrics
    PPG_win = np.array(PPG_win[:len(BVP_agg)]) # Take only 60s
    BVP_agg = np.array([(bvps - np.min(bvps)) / (np.max(bvps) - np.min(bvps)) for bvps in BVP_agg])
    DTW = []
    CC = []
    for w in range(min(len(BVP_agg), len(PPG_win))):
        dist = dtwai.dtw.distance(PPG_win[w], BVP_agg[w])
        DTW.append(dist)
        r, p = stats.pearsonr(PPG_win[w], BVP_agg[w])
        CC.append(r)
    DTW = np.mean(DTW)
    CC = np.mean(CC)

    # -- save results
    res.newDataSerie()
    res.addData('method', str(method))
    res.addData('RMSE', RMSE)
    res.addData('MAE', MAE)
    res.addData('MAX', MAX)
    res.addData('PCC', PCC)
    res.addData('CCC', CCC)
    res.addData('SNR', SNR)
    res.addData('bpmGT', bpmGT)
    res.addData('bpmES', bpmES_agg)
    res.addData('timeDTW', np.mean(DTW))
    res.addData('timePCC', CC)
    res.addData('timeGT', timesGT)
    res.addData('timeES', timesES)
    res.addData('videoFilename', videoFileName)
    res.addData('landmarks', tuple(landmarks))
    res.addDataSerie()                    

    return res, bvps_win, BVP_agg

class TestResult():
    """ 
    This class manages the results on a given dataset using multiple rPPG methods.
    """

    def __init__(self, filename=None):

        if filename == None:
            self.dataFrame = pd.DataFrame()
        else:
            self.dataFrame = pd.read_hdf(filename)
        self.dict = None

    def addDataSerie(self):
        # -- store serie
        if self.dict != None:
            self.dataFrame = self.dataFrame._append(self.dict, ignore_index=True)
            # self.dataFrame = self.dataFrame.append(self.dict, ignore_index=True)

    def newDataSerie(self):
        # -- new dict
        D = {}
        D['method'] = ''
        D['dataset'] = ''
        D['videoIdx'] = ''        # video filename
        D['videoFilename'] = ''   # GT signal filename
        D['RMSE'] = ''
        D['MAE'] = ''
        D['PCC'] = ''
        D['CCC'] = ''
        D['SNR'] = ''
        D['MAX'] = ''
        D['MAD'] = ''
        D['bpmGT'] = ''             # GT bpm
        D['bpmES'] = ''
        D['bpmES_mad'] = ''
        D['timeGT'] = ''            # GT bpm
        D['timeES'] = ''
        D['landmarks'] = ''
        D['timePCC'] = ''
        D['timeDTW'] = ''
        self.dict = D


    def addData(self, key, value):
        self.dict[key] = value

    def saveResults(self, outFilename=None):
        """
        Save the test results in a HDF5 library that can be opened using pandas.
        You can analyze the results using :py:class:`pyVHR.analysis.stats.StatAnalysis`
        """
        if outFilename is None:
            outFilename = "testResults.h5"
        else:
            self.outFilename = outFilename

        # -- save data
        self.dataFrame.to_hdf(outFilename, key='self.dataFrame', mode='w')


#### LANDMARKS CHOICE
        
def get_combinations(elements, min_len=2, max_len=3):
    combs = list(chain.from_iterable(combinations(elements, r) for r in range(min_len,max_len)))
    return [tuple(i) for i in combs]

def get_landmarks_combination(elements, min_len=2, max_len=3):
    """
    Combine landmarks in given list of elements
    Args:
        case: 'combine_random_landmarks', 'combine_roi_landmarks'
              'combine_random_landmarks': combine random landmarks in all face
              'combine_roi_landmarks': combine random landmarks in a roi
        elements: all landmarks or landmarks in a roi
        min_len: minimum number of landmarks to combine
        max_len: maximum number of landmarks to combine, None means all landmarks (e.g. in a ROI)
    """

    asym_elements = [ele.replace('left_','') for ele in elements if 'left_' in ele]
    elements = list(set(name.replace('left_', '').replace('right_', '').strip() for name in elements))

    if max_len is None:
        max_len = len(elements)
    all_landmarks = list(chain.from_iterable(combinations(elements, r) for r in range(min_len,max_len+1)))
    all_landmarks = [list(i) for i in all_landmarks]

    for i,comb in enumerate(all_landmarks):
        for ldmk in comb:
            if ldmk in asym_elements:
                comb = list(all_landmarks[i])
                comb.remove(ldmk)
                all_landmarks[i] = comb + [f'left_{ldmk}', f'right_{ldmk}']
        all_landmarks[i] = set(sorted(all_landmarks[i]))


    # y = pd.DataFrame({'landmarks':all_landmarks})
    # y['landmarks_names'] = y['landmarks'].apply(lambda x: set([name.replace('left_', '').replace('right_', '').strip() for name in x]))
    # y['landmarks_len'] = y['landmarks_names'].apply(lambda x: len(x))
    # y['landmarks'] = y['landmarks'].apply(lambda x: tuple(x))
    # print(y.shape, y.landmarks.unique())
    # y.query("landmarks_len == 5").landmarks.unique().size
    return all_landmarks

def get_landmarks(case,  min_len=2, max_len=3, all_landmarks_names=None, rois=None):
    """
    Args:
        case: 'each_18', 'each_28', 'all_in_roi', 'combine_random_landmarks', 'combine_roi_landmarks', 'combine_roi'
        'each_18': each 18 landmarks, (left and right) as one   
        'each_28': each 28 landmarks, (left and right) as separate
        'all_in_roi': all landmarks in a roi
        'combine_random_landmarks': combine random landmarks
        'combine_roi_landmarks': combine landmarks within a roi
        'combine_roi': combine different rois    
    """
    landmarks_keys = None
    if all_landmarks_names is None:
        all_landmarks_names = vhr.extraction.utils.CustomLandmarks().get_all_landmarks()
    if rois is None:
        rois = constants.get_rois()

    if case == 'each_28':
        all_landmarks_names = list(all_landmarks_names.keys())
        all_landmarks = [[landmark] for landmark in all_landmarks_names]
    elif case == 'each_18':
        all_landmarks = []
        all_landmarks_names = list(all_landmarks_names.keys())
        for landmark in all_landmarks_names:
            if 'left' in landmark:
                all_landmarks.append([landmark, landmark.replace('left', 'right')])
            if 'right' not in landmark and 'left' not in landmark:
                all_landmarks.append([landmark])
    elif case == 'all_in_roi':
        all_landmarks = [rois[roi] for roi in rois]
    elif case == 'combine_random_landmarks':
        elements = list(all_landmarks_names.keys())
        all_landmarks = get_landmarks_combination(elements, min_len, max_len)
    elif case == 'combine_roi_landmarks':
        all_landmarks = []
        for roi in rois:
            elements = list(rois[roi])
            tmp = get_landmarks_combination(elements, min_len=2, max_len=None)
            all_landmarks.extend(tmp)
    elif case == 'combine_roi':
        roi_list = list(rois.keys())
        roi_combinations = get_combinations(roi_list, 1, len(roi_list))

        landmarks_dict = dict()
        for roi_combination in roi_combinations:
            landmarks_dict[roi_combination] = [ldmk for roi in list(map(rois.get, roi_combination)) for ldmk in roi]
        all_landmarks = list(landmarks_dict.values())
        landmarks_keys = ['_'.join(landmarks_key) for landmarks_key in  list(landmarks_dict.keys())]
    else:
        raise Exception('case not supported', case)

    return all_landmarks, landmarks_keys