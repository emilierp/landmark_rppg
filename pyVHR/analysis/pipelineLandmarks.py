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

class PipielineLandmarks():
    """
    This class runs the pyVHR pipeline for the landmark approach.    
    """

    minHz = 0.65 # min hear frequency in Hz
    maxHz = 4.0 # max hear frequency in Hz

    def __init__(self):
        self.dataset = None
        self.sig_processing = None

    def get_dataset(self, dataset_name, video_DIR=None):
        """
        Return the dataset object for the given dataset name
        Args:
            dataset_name (str): Name of the dataset
            video_DIR (str): Path to the video directory. Can specify path to the video directory if not default.
        """
        # Default video path
        if video_DIR is None:
            video_DIR, BVP_DIR = constants.get_dataset_paths(dataset_name)
        else:
            _, BVP_DIR = constants.get_dataset_paths(dataset_name)
        dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
        self.dataset = dataset

        return dataset

    def get_signal_data(self, videoIdx, dataset, winsize=10, stride=1):
        """
        Read the signal data from the dataset and return the PPG_win, bpmGT, timesGT, subject_name
        Args:
            videoIdx (int): Index of the video in the dataset
            dataset (object): Dataset object
            winsize (int): Window size for the signal windowing in seconds. 
            stride (int): Stride for the signal windowing in seconds. 
        """

        videoFPS, sigFPS = constants.get_fps(dataset.name)
        signame = dataset.getSigFilename(videoIdx)
        videoFileName = dataset.getVideoFilename(videoIdx)
        fps = vhr.extraction.get_fps(videoFileName)
        subject_name = constants.get_subject_name(dataset.name, videoFileName)

        sigGT = dataset.readSigfile(signame)
        bpmGT, timesGT = sigGT.getBPM(winsize) # STFT 42-240 BPM

        # Get PPG signal and window it
        PPG = sigGT.data.T.reshape(-1,1,1) # shape (n,nb_ldmk,RGB)
        PPG_win, timesGTwin = sig_windowing(PPG, winsize, stride, sigFPS) # window PPG signal
        PPG_win = np.array([PPG_win[i].reshape(-1) for i in range(len(PPG_win))]) # from (n,1,1,m) to (n,m) PPG_win

        # Resample PPG signal to the video FPS
        if dataset.name == 'MR_NIRP' or dataset.name == 'LGI_PPGI':
            PPG_win = [scipy.signal.resample(PPG_win[i], int(fps/sigFPS*PPG_win[i].shape[-1]), axis=0) for i in range(len(PPG_win))]
        elif dataset.name == 'UBFC_PHYS':
            PPG_win = [scipy.signal.resample(PPG_win[i], np.ceil(fps/sigFPS*PPG_win[i].shape[-1]).astype(int), axis=0) for i in range(len(PPG_win))]
        else:
            raise Exception('PPG sampling for dataset not supported', dataset.name)
        
        # Normalize PPG signal to [0,1]
        PPG_win = np.array([(ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg)) for ppg in PPG_win])

        return PPG_win, bpmGT, timesGT, subject_name

    def extract_rgb(self, ldmks_list=None, videoFileName=None, roi_approach='landmark', sampling_method='all', nb_sample_points=2000, seconds=0, winsize=10, stride=1, visualize=False, verb=False, cuda=True):
        """
        Extract RGB of landmarks from the video file and return the windowed signal.

        Args:
            ldmks_list: a list of landmarks to extract 
                    - ex: [glabella, chin] for roi_approach='landmark'
                    - ex: [1,2,3,4] for roi_approach='patches'
            videoFileName: The video filenane to analyse
            roi_approach:
                - 'holistic', uses the holistic approach, i.e. the whole face skin
                - 'patches', uses multiple patches as Regions of Interest
                - 'landmark', uses custom landmarks as Regions of Interest
            sampling_method: Sampling method for point inside the landmark when roi_approach='landmark'
                - 'all': sample all points inside the landmark patch
                - 'random': sample random points inside the landmark patch
            nb_sample_points: Number of sample points to extract from the landmark patch if sampling_method='random'
            seconds: The number of seconds to process (0 for all video)
            winsize: The size of the window in frame
            stride: The stride of the window in frame
            cuda: Enable computations on GPU
            verb: True, shows the main steps  
            cuda: True - Enable computations on GPU
        """

        RGB_LOW_HIGH_TH=(0,240) # Thresholds for RGB channels
        Skin_LOW_HIGH_TH=(0, 240) # Thresholds for skin pixel values
        cuda = True

        # test video filename
        assert os.path.isfile(videoFileName), "The video file does not exists!"
        if verb:
            print("Video: ", videoFileName)

        # test roi approach
        assert roi_approach in ['holistic', 'patches', 'landmark'], "The roi approach is not recognized!"

        sig_processing = SignalProcessing()

        if cuda and verb:
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(0)

        if visualize:
            sig_processing.set_visualize_skin_and_landmarks(True, True, False, True)

        ## 1. set skin extractor: Convex Hull (MediaPipe)
        target_device = 'GPU' if cuda else 'CPU'
        sig_processing.set_skin_extractor(SkinExtractionConvexHull(target_device))

        ## 2. set patches
        if roi_approach == 'patches':
            # set landmark list
            if not ldmks_list:
                ldmks_list = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]
            sig_processing.set_landmarks(ldmks_list)
            sig_processing.set_square_patches_side(square_side=float(40.0))       
        if roi_approach == 'landmark':
            if len(ldmks_list) == 0: # take all landmarks
                ldmks_list = [x for x in list(pyVHR.extraction.CustomLandmarks().__dict__)]
            all_landmarks  = pyVHR.extraction.CustomLandmarks().get_all_landmarks()
            ldmk_values = [all_landmarks[ldmk] for ldmk in ldmks_list]
            sig_processing.set_landmark_roi(ldmk_values, sampling_method=sampling_method, nb_sample_points=nb_sample_points)

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
        if verb:
            print('\nRoi processing...')
        sig = []
        if roi_approach == 'holistic':
            # SIG extraction with holistic
            sig = sig_processing.extract_holistic(videoFileName)
        elif roi_approach == 'patches':
            # SIG extraction with patches
            sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')
        elif roi_approach == 'landmark':
            # SIG extraction with landmarks
            sig = sig_processing.extract_landmarks(videoFileName)
            
        ## 4. sig windowing
        windowed_sig, timesES = sig_windowing(sig, winsize, stride, fps)
        if verb:
            print(f' - Number of windows: {len(windowed_sig)}')
            print(' - Win size: (#ROI, #landmarks, #frames) = ', windowed_sig[0].shape)
            print(f" - Extraction approach: {roi_approach} with {len(windowed_sig)} windows")

        self.sig_processing = sig_processing
        
        return windowed_sig, timesES

    def extract_bpm(self, windowed_sig, timesES, fps, roi_approach='landmark', method='cupy_CHROM', estimate='mean', 
                    movement_thrs=[10, 5, 2], winsize=10, pre_filt=True, post_filt=True, verb=False, cuda=True
        ):
        
        """
        Extract the BVP and BPM from the windowed signal.
        Args:
            windowed_sig: Windowed rPPG signal
            timesES: The time of the windowed signal
            fps: The frames per second of the video
            roi_approach: The approach to use to extract the ROI
                - 'holistic', uses the holistic approach, i.e. the whole face skin
                - 'patches', uses multiple patches as Regions of Interest
                - 'landmark', uses face landmarks as Regions of Interest
            methods: A collection of rPPG methods defined in pyVHR
            estimate: if patches: 'medians', 'clustering', the method for BPM estimate on each window 
            movement_thrs: Thresholds for movements filtering (eg.:[10, 5, 2])
            winsize: The size of the window in seconds
            pre_filt: True, uses bandpass filter on the windowed RGB signal
            post_filt: True, uses bandpass filter on the estimated BVP signal
            verb: True, shows the main steps  
            cuda: True - Enable computations on GPU
        """

        ## 5. PRE FILTERING
        if verb:
            print('\nPre filtering...')
        filtered_windowed_sig = windowed_sig

        if pre_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            filtered_windowed_sig = apply_filter(filtered_windowed_sig,
                                                    method_to_call, 
                                                    fps=fps, 
                                                    params={'minHz':self.minHz, 
                                                            'maxHz':self.maxHz, 
                                                            'fps':'adaptive', 
                                                            'order':6})
        if verb:
            print(f' - Pre-filter applied: {method_to_call.__name__}')

        ## 6. BVP extraction multimethods
        bvps_win = []
        # for method in methods:
        if verb:
            print("\nBVP extraction...")
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

        bvps_win_m  = RGB_sig_to_BVP(filtered_windowed_sig, 
                                fps, device_type=method_device, 
                                method=method_to_call, params=pars)

        ## 7. POST FILTERING
        if post_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps_win_m  = apply_filter(bvps_win_m , 
                                method_to_call, 
                                fps=fps, 
                                params={'minHz':self.minHz, 'maxHz':self.maxHz, 'fps':'adaptive', 'order':6})

        if verb:
            print(f' - Post-filter applied: {method_to_call.__name__}')
        # collect
        if len(bvps_win) == 0:
            bvps_win = bvps_win_m 
        else:
            for i in range(len(bvps_win_m)):
                bvps_win[i] = np.concatenate((bvps_win[i], bvps_win_m[i]))
                if i == 0: print(bvps_win[i].shape)

        ## 8. BPM extraction
        if verb:
            print("\nBPM estimation...")  
            print(f" - roi appproach: {roi_approach}") 

        if roi_approach == 'holistic':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
            else:
                bpmES = BVP_to_BPM(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
            bvps_win = [arr.squeeze() for arr in bvps_win]
            
        elif roi_approach == 'patches':
            if estimate == 'clustering':
                #if cuda and False:
                #    bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=minHz, maxHz=maxHz)
                #else:
                #bpmES = BPM_clustering(sig_processing, bvps_win, winsize, movement_thrs=[15, 15, 15], fps=fps, opt_factor=0.5)
                ma = MotionAnalysis(self.sig_processing, winsize, fps)
                bpmES = BPM_clustering(ma, bvps_win, fps, winsize, movement_thrs=movement_thrs, opt_factor=0.5)
                
            elif estimate == 'median':
                if cuda:
                    bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
                else:
                    bpmES = BVP_to_BPM(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
                bpmES,_ = BPM_median(bpmES)

        elif roi_approach == 'landmark':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
            else:
                bpmES = BVP_to_BPM(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
            
            # Aggregate BVP and BPM into mean of all landmarks
            BVP_agg = [[np.array(arr) for arr in bvps_win]]
            for i,bvp_win in enumerate(BVP_agg):
                BVP_agg[i] = np.mean(bvp_win, axis=1) # take the mean in case the landmark is symmetrics
            bvps_win = np.mean(BVP_agg, axis=0) # mean of all landmarks

            bpmES_agg = [[np.array(arr) for arr in bpmES]]
            for i, bpm in enumerate(bpmES_agg):   
                if np.array(bpm).ndim >1: # take the mean in case the landmark is symmetrics
                    bpmES_agg[i] = np.mean(bpm, axis=1)
                else: # need to reduce the dimension
                    bpmES_agg[i] = np.array(bpm)
            bpmES = np.mean(bpmES_agg, axis=0) # mean of all landmarks
        
        else:
            raise ValueError("Estimation approach unknown!")
        
        if verb:
            print('\n...done!\n')

        return bvps_win, timesES, bpmES
        
    def evaluate_extraction(self, bvps_win, bpmES, bpmGT, timesES, timesGT, PPG_win, videoFPS, res=None, method=None, videoFileName=None, landmarks=None, verb=False):
        """
        Evaluate the extraction results and return the results.
        Args:
            bvps_win: The extracted BVP signal (windowed)
            bpmES: The estimated BPM signal
            bpmGT: The ground truth BPM signal
            timesES: The time of the extracted signal
            timesGT: The time of the ground truth signal
            PPG_win: The ground truth PPG signal (windowed)
            videoFPS: The frames per second of the video
            res: The test results object
            method: The method used for the extraction
            landmarks: The landmarks used for the extraction
            verb: True, shows the main steps
        """

        # BPM Error metrics
        RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps_win, videoFPS, bpmES, bpmGT, timesES, timesGT)

        # BVP Error metrics
        try:
            PPG_win = np.array(PPG_win[:len(bvps_win)]) # Take only same length as BVP, in case of different length. PPG_win is already normalized.
            bvps_win = np.array([(bvps - np.min(bvps)) / (np.max(bvps) - np.min(bvps)) for bvps in bvps_win])
            DTW = []
            rPPG_PCC = []
            for w in range(min(len(bvps_win), len(PPG_win))):
                dist = dtwai.dtw.distance(PPG_win[w], bvps_win[w])
                DTW.append(dist)
                r, p = stats.pearsonr(PPG_win[w], bvps_win[w])
                rPPG_PCC.append(r)
            DTW = np.mean(DTW)
            rPPG_PCC = np.mean(rPPG_PCC)
        except:
            print("Error in DTW and rPPG_PCC computation")
            DTW = 0
            rPPG_PCC = 0

        # -- save results
        if res is None: 
            res = TestResult()
        res.newDataSerie()
        res.addData('videoFilename', videoFileName)
        res.addData('method', str(method))
        res.addData('RMSE', RMSE[0])
        res.addData('MAE', MAE[0])
        res.addData('MAX', MAX[0])
        res.addData('PCC', PCC[0])
        res.addData('CCC', CCC[0])
        res.addData('SNR', SNR[0])
        res.addData('bpmGT', bpmGT)
        res.addData('bpmES', bpmES)
        res.addData('DTW', np.mean(DTW))
        res.addData('rPPG_PCC', rPPG_PCC)
        res.addData('timeGT', timesGT)
        res.addData('timeES', timesES)
        res.addData('landmarks', tuple(landmarks))
        res.addDataSerie()   

        if verb:
            print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, PCC = %.2f, CCC = %.2f, SNR = %.2f DTW = %.2f rPPG_PCC = %.2f"  % 
                  (RMSE, MAE, MAX, PCC, CCC, SNR, DTW, rPPG_PCC))                

        return res
    
    def run_pipeline(self, dataset_name, videoIdx, res=None, landmarks=['glabella'], roi_approach='landmark', sampling_method='random', nb_sample_points=2000, seconds=60,
                     winsize=10, stride=1, methods=['cupy_CHROM', 'cpu_LGI'], estimate='mean', visualize=False, verb=False, cuda=True):
        """
        Run the pyVHR pipeline, from video processing to BPM estimation.
        Args:
            dataset_name: The name of the dataset
            videoIdx: The index of the video in the dataset
            res: The test results object
            landmarks: The landmarks to extract (list)
            roi_approach: The approach to use to extract the ROI
                - 'holistic', uses the holistic approach, i.e. the whole face skin
                - 'patches', uses multiple patches as Regions of Interest
                - 'landmark', uses face landmarks as Regions of Interest
            sampling_method: Sampling method for point inside the landmark when roi_approach='landmark'
                - 'all': sample all points inside the landmark patch
                - 'random': sample random points inside the landmark patch
            nb_sample_points: Number of sample points to extract from the landmark patch if sampling_method='random'
            seconds: The number of seconds to process (0 for all video)
            winsize: The size of the window in seconds
            stride: The stride of the window in seconds
            methods: A collection of rPPG methods defined in pyVHR
            estimate: if patches: 'medians', 'clustering', the method for BPM estimate on each window 
            visualize: True, save the processed frames
            verb: True, shows the main steps  
            cuda: True - Enable computations on GPU
        """ 
        ### Check args
        if res is None:
            res = TestResult()

        av_meths = getmembers(pyVHR.BVP.methods, isfunction)
        available_methods = [am[0] for am in av_meths]
        if not isinstance(methods, list):
            methods = [methods]
        for m in methods:
            assert m in available_methods, "\nrPPG method not recognized!!"

        ### Get dataset object
        if self.dataset is None:
            dataset = self.get_dataset(dataset_name)
        else:
            dataset = self.dataset
        videoFPS, sigFPS = constants.get_fps(dataset_name)
        videoFileName = dataset.getVideoFilename(videoIdx)
        if verb:
            print("Video: ", videoFileName)

        ### Get PPG signal data
        PPG_win, bpmGT, timesGT, subject_name = self.get_signal_data(videoIdx, dataset)

        ### Extract RGB signal and BVP for all methods
        windowed_sig, timesES = self.extract_rgb(landmarks, videoFileName, roi_approach, sampling_method, nb_sample_points, seconds, winsize, stride, visualize, verb, cuda)
        for method in methods:
            print(method)
            bvps_win, timesES, bpmES = self.extract_bpm( windowed_sig, timesES, videoFPS, roi_approach, method, estimate, winsize=winsize, verb=verb, cuda=cuda)
            res = self.evaluate_extraction(bvps_win, bpmES, bpmGT, timesES, timesGT, PPG_win, videoFPS, res, method, videoFileName, landmarks, verb)
        
        return res


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

    def newDataSerie(self):
        # -- new dict
        D = {}
        D['method'] = ''
        D['dataset'] = ''
        D['videoIdx'] = ''        # video filename
        D['videoFilename'] = ''   # GT signal filename
        D['landmarks'] = ''
        D['RMSE'] = ''
        D['MAE'] = ''
        D['PCC'] = ''
        D['CCC'] = ''
        D['SNR'] = ''
        D['MAX'] = ''
        D['MAD'] = ''
        D['rPPG_PCC'] = ''
        D['DTW'] = ''
        D['bpmGT'] = ''             # GT bpm
        D['bpmES'] = ''
        D['timeGT'] = ''            # GT bpm
        D['timeES'] = ''
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