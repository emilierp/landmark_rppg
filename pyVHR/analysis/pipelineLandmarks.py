import configparser
import ast
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np
import time
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
import os.path
from pyVHR.deepRPPG.mtts_can import *
from pyVHR.deepRPPG.hr_cnn import *
from pyVHR.extraction.utils import *

class LandmarksPipeline():
    """ 
    This class runs the pyVHR pipeline on a single video or dataset
    """

    minHz = 0.65 # min heart frequency in Hz
    maxHz = 4.0  # max heart frequency in Hz

    def __init__(self):
        self.res = None

    def run_on_video_multimethods(self, videoFileName, 
                    winsize, 
                    bpmGT,
                    timesGT,
                    ldmks_list=None,
                    cuda=True, 
                    roi_method='convexhull', 
                    roi_approach='patches', 
                    methods=['cupy_CHROM', 'cpu_POS', 'cpu_LGI'], 
                    estimate='median', 
                    movement_thrs=[10, 5, 2],
                    patch_size=30, 
                    RGB_LOW_HIGH_TH = (0,240),
                    Skin_LOW_HIGH_TH = (0, 240),
                    pre_filt=True, 
                    post_filt=True, 
                    verb=True):
        """ 
        Runs the pipeline on a specific video file.

        Args:
            videoFileName:
                - The video filenane to analyse
            winsize:
                - The size of the window in frame
            ldmks_list:
                - (default None) a list of custom landmakrs to uses
            cuda:
                - True - Enable computations on GPU
            roi_method:
                - 'convexhull', uses MediaPipe's lanmarks to compute the convex hull on the face skin
                - 'faceparsing', uses BiseNet to parse face components and segment the skin
            roi_approach:
                - 'holistic', uses the holistic approach, i.e. the whole face skin
                - 'patches', uses multiple patches as Regions of Interest
            methods:
                - A collection of rPPG methods defined in pyVHR
            estimate:
                - if patches: 'medians', 'clustering', the method for BPM estimate on each window 
            movement_thrs:
                - Thresholds for movements filtering (eg.:[10, 5, 2])
            patch_size:
                - the size of the square patch, in pixels
            RGB_LOW_HIGH_TH: 
                - default (75,230), thresholds for RGB channels 
            Skin_LOW_HIGH_TH:
                - default (75,230), thresholds for skin pixel values 
            pre_filt:
                - True, uses bandpass filter on the windowed RGB signal
            post_filt:
                - True, uses bandpass filter on the estimated BVP signal
            verb:
                - True, shows the main steps  
        """
        # -- catch data (object)
        res = TestResult()

        # set landmark list
        if not ldmks_list:
            ldmks_list = ['chin']
        
        # test video filename
        assert os.path.isfile(videoFileName), "The video file does not exists!"
        if verb:
            print("Video: ", videoFileName)


        sig_processing = SignalProcessing()
        av_meths = getmembers(pyVHR.BVP.methods, isfunction)
        available_methods = [am[0] for am in av_meths]

        for m in methods:
            assert m in available_methods, "\nrPPG method not recognized!!"

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
        sig_processing.set_total_frames(0)

        ## 3. ROI selection
        sig = []
        if roi_approach == 'holistic':
            # SIG extraction with holistic
            sig = sig_processing.extract_holistic(videoFileName)
        elif roi_approach == 'patches':
            # SIG extraction with patches
            sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')

        ## 4. sig windowing
        windowed_sig, timesES = sig_windowing(sig, winsize, 1, fps)
        if verb:
            print(f" - Extraction approach: {roi_approach} with {len(windowed_sig)} windows")

        ## 5. PRE FILTERING
        filtered_windowed_sig = windowed_sig
        
        # -- color threshold - applied only with patches
        # if roi_approach == 'patches':
        #    filtered_windowed_sig = apply_filter(windowed_sig,
        #                                        rgb_filter_th,
        #                                        params={'RGB_LOW_TH': RGB_LOW_HIGH_TH[0],
        #                                                'RGB_HIGH_TH': RGB_LOW_HIGH_TH[1]})

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
        
        end_video = time.time()

        ## 6. BVP extraction multimethods
        bvps_win = []
        for method in methods:
            start_method = time.time()
            try:
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
                                        params={'minHz':self.minHz, 'maxHz':self.maxHz, 'fps':'adaptive', 'order':6})

                ## 8. BPM extraction

                if roi_approach == 'holistic':
                    if cuda:
                        bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
                    else:
                        bpmES = BVP_to_BPM(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)

                elif roi_approach == 'patches':
                    if estimate == 'clustering':
                        #if cuda and False:
                        #    bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
                        #else:
                        #bpmES = BPM_clustering(sig_processing, bvps_win, winsize, movement_thrs=[15, 15, 15], fps=fps, opt_factor=0.5)
                        ma = MotionAnalysis(sig_processing, winsize, fps)
                        bpmES = BPM_clustering(ma, bvps_win, fps, winsize, movement_thrs=movement_thrs, opt_factor=0.5)

                    elif estimate == 'median':
                        if cuda:
                            bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
                        else:
                            bpmES = BVP_to_BPM(bvps_win, fps, minHz=self.minHz, maxHz=self.maxHz)
                        bpmES, MAD = BPM_median(bpmES)
                    if verb:
                        print(f" - roi approach: {roi_approach} ({estimate}) ({ldmks_list})")
                else:
                    raise ValueError("Estimation approach unknown!")

                ## 9. error metrics
                RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps_win, fps, bpmES, bpmGT, timesES, timesGT)

            except Exception as e:
                print("Error: ", e)
                RMSE, MAE, MAX, PCC, CCC, SNR, MAD = [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]
                bpmGT, bpmES, timesGT, timesES = np.nan, np.nan, np.nan, np.nan

            end_method = time.time()
            # print(f"Check clock {start_video}, {end_video}, {start_method}, {end_method}, method:{end_method - start_method}, beginning:{end_video - start_video}, total:{end_method - start_method + end_video - start_video}")

            # -- save results
            res.newDataSerie()
            res.addData('method', str(method))
            res.addData('RMSE', RMSE)
            res.addData('MAE', MAE)
            res.addData('MAX', MAX)
            res.addData('PCC', PCC)
            res.addData('CCC', CCC)
            res.addData('SNR', SNR)
            res.addData('MAD', MAD)
            res.addData('bpmGT', bpmGT)
            res.addData('bpmES', bpmES)
            res.addData('timeGT', timesGT)
            res.addData('timeES', timesES)
            res.addData('TIME_REQUIREMENT', (end_method - start_method) + (end_video - start_video))
            res.addData('videoFilename', videoFileName)
            res.addData('landmarks', ldmks_list)
            res.addDataSerie()
            self.res = res

            if verb:
                printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)

        return res


    def run_on_dataset(self, configFilename, verb=True):
        """ 
        Like the 'run_on_video' function, it runs on all videos of a specific 
        dataset as specified by the loaded configuration file.
        CHANGED: Try different combinations of landmarks

        Args:
            configFilename:
                - The path to the configuration file
            verb:
                - False - not verbose
                - True - show the main steps
        """
        # -- cfg file  
        self.configFilename = configFilename
        self.parse_cfg(self.configFilename)
        
        # -- cfg parser
        parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        parser.optionxform = str
        if not parser.read(self.configFilename):
            raise FileNotFoundError(self.configFilename)
        if verb:
            self.__verbose('a')

        # -- dataset & cfg params
        if 'path' in self.datasetdict and self.datasetdict['path'] != 'None':
            dataset = datasetFactory(self.datasetdict['dataset'], 
                                    videodataDIR=self.datasetdict['videodataDIR'], 
                                    BVPdataDIR=self.datasetdict['BVPdataDIR'], 
                                    path=self.datasetdict['path'])
        else:
            dataset = datasetFactory(self.datasetdict['dataset'], 
                                    videodataDIR=self.datasetdict['videodataDIR'], 
                                    BVPdataDIR=self.datasetdict['BVPdataDIR'])

        # -- catch data (object)
        res = TestResult()
        start_process = time.time()

        # -- SIG processing
        sig_processing = SignalProcessing()
        if eval(self.sigdict['cuda']) and verb:
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(int(self.sigdict['cuda_device']))
        if verb:
            print(f" -  cuda device: {self.sigdict['cuda']}")

        ## 1. set skin extractor
        target_device = 'GPU' if eval(self.sigdict['cuda']) else 'CPU'
        if self.sigdict['skin_extractor'] == 'convexhull':
            sig_processing.set_skin_extractor(
                SkinExtractionConvexHull(target_device))
        elif self.sigdict['skin_extractor'] == 'faceparsing':
            sig_processing.set_skin_extractor(
                SkinExtractionFaceParsing(target_device))
        else:
            raise ValueError("Unknown roi method extraction!")
        if verb:
            print(f" -  skin extractor: {self.sigdict['skin_extractor']}")
 
        ## 2. set patches             # CHANGED , suppose only custom landmarks
        if self.sigdict['approach'] == 'patches':
            ldmks_config = ast.literal_eval(
                    self.sigdict['landmarks_list'])

            if len(ldmks_config) == 0: # take all landmarks
                self.sigdict['landmarks_list'] = [
                    [x] for x in list(pyVHR.extraction.CustomLandmarks().__dict__)
                ]
            else: # take only the specified landmarks
                self.sigdict['landmarks_list'] = ldmks_config
            
            all_landmarks  = pyVHR.extraction.CustomLandmarks().get_all_landmarks()
            ldmks_config = []
            for ldmk_list in self.sigdict['landmarks_list']:
                # take all unique values for each landmark in the list
                ldmk_values = list(np.unique(sum([all_landmarks[ldmk] for ldmk in ldmk_list], [])))
                ldmks_config.append(ldmk_values)
        else:
            raise Exception("Can only use patches approach with custom landmarks")

        print("Landmarks list : ", self.sigdict['landmarks_list'])
        print("Landmarks config : ", ldmks_config)

        for ldmk_idx,ldmks_list in enumerate(ldmks_config):
            start_ldmk = time.time()
            if verb:
                print("\nTesting landmarks: ", self.sigdict['landmarks_list'][ldmk_idx])
            if len(ldmks_list) > 0:
                sig_processing.set_landmarks(ldmks_list)
            if self.sigdict['patches'] == 'squares':
                # set squares patches side dimension
                sig_processing.set_square_patches_side(
                    float(self.sigdict['squares_dim']))
            elif self.sigdict['patches'] == 'rects':
                # set rects patches sides dimensions
                rects_dims = ast.literal_eval(
                    self.sigdict['rects_dims'])
                if len(rects_dims) > 0:
                    sig_processing.set_rect_patches_sides(
                        np.array(rects_dims, dtype=np.float32))     
            # if verb:
            #     print(f" -  ROI approach: {self.sigdict['approach']}")
            
            # set sig-processing and skin-processing params
            SignalProcessingParams.RGB_LOW_TH = np.int32(
                self.sigdict['sig_color_low_threshold'])
            SignalProcessingParams.RGB_HIGH_TH = np.int32(
                self.sigdict['sig_color_high_threshold'])
            SkinProcessingParams.RGB_LOW_TH = np.int32(
                self.sigdict['skin_color_low_threshold'])
            SkinProcessingParams.RGB_HIGH_TH = np.int32(
                self.sigdict['skin_color_high_threshold'])

            # set video idx
            # CHANGED
            if (len(self.videoIdx) == 0):
                self.videoIdx = [int(v) for v in range(len(dataset.videoFilenames))]

            print("Dataset : ", self.datasetdict['dataset'], self.datasetdict['videodataDIR'], self.datasetdict['BVPdataDIR'])
            print("Number of videos in folder : ", len(dataset.videoFilenames))
            print("Number of videos in index : ", len(self.videoIdx))

            end_process = time.time()
                
            # -- loop on videos
            for v in self.videoIdx:
                start_video = time.time()
                if verb:
                    print("\n## videoID: %d" % (v))

                # -- ground-truth signal
                try:
                    fname = dataset.getSigFilename(v)
                    sigGT = dataset.readSigfile(fname)
                except:
                    continue
                winSizeGT = int(self.sigdict['winSize'])
                bpmGT, timesGT = sigGT.getBPM(winSizeGT)

                # -- video file name
                videoFileName = dataset.getVideoFilename(v)
                print(videoFileName)
                fps = get_fps(videoFileName)

                sig_processing.set_total_frames(
                    int(self.sigdict['tot_sec'])*fps)

                ## 3. ROI selection
                sig = []
                if str(self.sigdict['approach']) == 'holistic':
                    # mean extraction with holistic
                    sig = sig_processing.extract_holistic(videoFileName)
                elif str(self.sigdict['approach']) == 'patches':
                    # mean extraction with patches
                    sig = sig_processing.extract_patches(
                        videoFileName, str(self.sigdict['patches']), str(self.sigdict['type']))

                ## 4. sig windowing
                windowed_sig, timesES = sig_windowing(sig, int(self.sigdict['winSize']), 1, fps)

                end_video = time.time()

                # -- loop on methods
                for m in self.methods:
                    try:
                        start_method = time.time()
                        if verb:
                            app = str(self.sigdict['approach'])
                            print(f'## method: {str(m)} ({app}) {self.sigdict["landmarks_list"][ldmk_idx]}')

                        ## 5. PRE FILTERING
                        filtered_windowed_sig = windowed_sig

                        # -- color threshold - applied only with patches
                        #if str(self.sigdict['approach']) == 'patches':
                        #    filtered_windowed_sig = apply_filter(windowed_sig, rgb_filter_th,
                        #        params={'RGB_LOW_TH':  np.int32(self.bvpdict['color_low_threshold']),
                        #                'RGB_HIGH_TH': np.int32(self.bvpdict['color_high_threshold'])})

                        # -- custom filters
                        prefilter_list = ast.literal_eval(self.methodsdict[m]['pre_filtering'])
                        if len(prefilter_list) > 0:
                            for f in prefilter_list:
                                # if verb:
                                #     print("  pre-filter: %s" % f)
                                fdict = dict(parser[f].items())
                                if fdict['path'] != 'None':
                                    # custom path
                                    spec = util.spec_from_file_location(fdict['name'], fdict['path'])
                                    mod = util.module_from_spec(spec)
                                    spec.loader.exec_module(mod)
                                    method_to_call = getattr(mod, fdict['name'])
                                else:
                                    # package path
                                    module = import_module('pyVHR.BVP.filters')
                                    method_to_call = getattr(module, fdict['name'])
                                filtered_windowed_sig = apply_filter(filtered_windowed_sig, method_to_call, fps=fps, params=ast.literal_eval(fdict['params']))

                        ## 6. BVP extraction
                        if self.methodsdict[m]['path'] != 'None':
                            # custom path
                            spec = util.spec_from_file_location(self.methodsdict[m]['name'], self.methodsdict[m]['path'])
                            mod = util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            method_to_call = getattr(mod, self.methodsdict[m]['name'])
                        else:
                            # package path
                            module = import_module('pyVHR.BVP.methods')
                            method_to_call = getattr(module, self.methodsdict[m]['name'])
                        bvps_win = RGB_sig_to_BVP(filtered_windowed_sig, fps,
                                            device_type=self.methodsdict[m]['device_type'], 
                                            method=method_to_call, 
                                            params=ast.literal_eval(self.methodsdict[m]['params']))

                        ## 7. POST FILTERING
                        postfilter_list = ast.literal_eval(self.methodsdict[m]['post_filtering'])
                        if len(postfilter_list) > 0:
                            for f in postfilter_list:
                                # if verb:
                                #     print("  post-filter: %s" % f)
                                fdict = dict(parser[f].items())
                                if fdict['path'] != 'None':
                                    # custom path
                                    spec = util.spec_from_file_location(
                                        fdict['name'], fdict['path'])
                                    mod = util.module_from_spec(spec)
                                    spec.loader.exec_module(mod)
                                    method_to_call = getattr(mod, fdict['name'])
                                else:
                                    # package path
                                    module = import_module('pyVHR.BVP.filters')
                                    method_to_call = getattr(module, fdict['name'])
                                
                                bvps_win = apply_filter(bvps_win, method_to_call, fps=fps, params=ast.literal_eval(fdict['params']))

                        ## 8. BPM extraction
                        MAD = []
                        if self.bpmdict['estimate'] == 'holistic' or self.bpmdict['estimate'] == 'median':
                            if eval(self.sigdict['cuda']):
                                bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=float(
                                    self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                            else:
                                bpmES = BVP_to_BPM(bvps_win, fps, minHz=float(
                                    self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                            
                            if self.bpmdict['estimate'] == 'median':
                                # median BPM from multiple estimators BPM
                                bpmES, MAD = BPM_median(bpmES)

                        elif self.bpmdict['estimate'] == 'clustering':
                            # if eval(self.sigdict['cuda']):
                            #     bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=float(
                            #         self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                            # else:
                            #bpmES = BPM_clustering(sig_processing, bvps_win, winSizeGT, movement_thrs=[15, 15, 15], fps=fps, opt_factor=0.5)
                            ma = MotionAnalysis(sig_processing, winSizeGT, fps)
                            mthrs = self.bpmdict['movement_thrs']
                            mthrs = mthrs.replace('[', '')
                            mthrs = mthrs.replace(']', '')
                            movement_thsrs = [float(i) for i in mthrs.split(",")]
                            bpmES = BPM_clustering(ma, bvps_win, fps, winSizeGT, movement_thrs=movement_thrs, opt_factor=0.5)

                        end_method = time.time() 
                        time_requirement = (end_method-start_method) + (end_video-start_video) + (end_process-start_process)
                    
                    except Exception as e:
                        print("error: ", e) 
                        RMSE, MAE, MAX, PCC, CCC, SNR, MAD = [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]
                        bpmGT, bpmES, timesGT, timesES = np.nan, np.nan, np.nan, np.nan
                        m = ''
                        videoFileName = dataset.getVideoFilename(v)
                        time_requirement = np.nan
                        continue

                    ## 9. error metrics
                    RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps_win, fps, bpmES, bpmGT, timesES, timesGT)

                    # -- save results
                    res.newDataSerie()
                    res.addData('dataset', str(self.datasetdict['dataset']))
                    res.addData('method', str(m))
                    res.addData('videoIdx', v)
                    res.addData('RMSE', RMSE)
                    res.addData('MAE', MAE)
                    res.addData('MAX', MAX)
                    res.addData('PCC', PCC)
                    res.addData('CCC', CCC)
                    res.addData('SNR', SNR)
                    res.addData('MAD', MAD)
                    res.addData('bpmGT', bpmGT)
                    res.addData('bpmES', bpmES)
                    res.addData('timeGT', timesGT)
                    res.addData('timeES', timesES)
                    res.addData('TIME_REQUIREMENT', time_requirement)
                    res.addData('videoFilename', videoFileName)
                    res.addData('landmarks', self.sigdict['landmarks_list'][ldmk_idx])
                    res.addDataSerie()
                    
                    self.res = res
                    # res.dataFrame.to_hdf("C:/Users/erolland/Documents/pyVHR/results/test_landmarks/h5/LGI_PGGI_each_landmark_test.h5", key='tmp', mode='a')
            
                    if verb:
                        printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)

        return res

    def parse_cfg(self, configFilename):
        """ parses the given configuration file for loading the test's parameters.
        
        Args:
            configFilename: configuation file (.cfg) name of path .

        """

        self.parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        self.parser.optionxform = str
        if not self.parser.read(configFilename):
            raise FileNotFoundError(configFilename)

        # load paramas
        self.datasetdict = dict(self.parser['DATASET'].items())
        self.sigdict = dict(self.parser['SIG'].items())
        self.bvpdict = dict(self.parser['BVP'].items())
        self.bpmdict = dict(self.parser['BPM'].items())

        # CHANGED
        if "test_ldmks" in self.sigdict.keys():
            self.test_ldmks = self.sigdict['test_ldmks']
        else: self.test_ldmks = False

        # video idx list extraction
        if isinstance(ast.literal_eval(self.datasetdict['videoIdx']), list):
            self.videoIdx = [int(v) for v in ast.literal_eval(
                self.datasetdict['videoIdx'])]

        # load parameters for each methods
        self.methodsdict = {}
        self.methods = ast.literal_eval(self.bvpdict['methods'])
        for x in self.methods:
            self.methodsdict[x] = dict(self.parser[x].items())

    def __merge(self, dict1, dict2):
        for key in dict2:
            if key not in dict1:
                dict1[key] = dict2[key]

    def __verbose(self, verb):
        if verb == 'a':
            print("** Run the test with the following config:")
            print("      dataset: " + self.datasetdict['dataset'].upper())
            print("      methods: " + str(self.methods))

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
        D['sigFilename'] = ''     # GT signal filename
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
        D['TIME_REQUIREMENT'] = ''
        D['landmarks'] = ''
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



############################ STATS.PY #########################################

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as ss
import scikit_posthocs as sp
from autorank import autorank, plot_stats, create_report
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase, Colorbar
import itertools
import math


class StatAnalysisLdmk:
  """ 
  Statistic analyses for multiple datasets and multiple landmarks
  """

  def __init__(self, filepath, join_data=False, remove_outliers=False):
    """
    Args:
      filepath:
        - The path to the file contaning the results to test
      join_data: 
        - 'True'  - If filepath is a folder, join the dataframes contained in 
                    the folder (to be used when wanting to merge multiple results from the same pipeline on the same dataset)
        - 'False' - (default) To be used if you want to test the same pipeline (eventually with multiple methods) on multiple datasets
      remove_outliers:
        - 'True'  -  Remove outliers from data prior to statistical testing
        - 'False' - (default) no outlier removal
    """

    if os.path.isdir(filepath):
        self.multidataset = True
        self.path = filepath + "/"
        self.datasetsList = os.listdir(filepath)
    elif os.path.isfile(filepath):
        self.multidataset = False
        self.datasetsList = [filepath]
        self.path = ""
    else:
        raise "Error: filepath is wrong!"

    self.join_data = join_data
    self.available_metrics = ['MAE', 'RMSE', 'PCC', 'CCC', 'SNR']
    self.remove_outliers = remove_outliers

    # -- get data
    self.__getLandmarks()
    self.metricSort = {'MAE': 'min', 'RMSE': 'min', 'PCC': 'max', 'CCC': 'max', 'SNR': 'max'}
    self.scale = {'MAE': 'log', 'RMSE': 'log', 'PCC': 'linear', 'CCC': 'linear', 'SNR': 'linear'}

    self.use_stats_pipeline = False

  def __any_equal(self, mylist):
    equal = []
    for a, b in itertools.combinations(mylist, 2):
        equal.append(a == b)
    return np.any(equal)

  def run_stats(self, landmarks=None, metric='CCC', approach='frequentist', print_report=True):
    """
    Runs the statistical testing procedure by automatically selecting the appropriate test for the available data.

    Args:
        landmarks:
            - The landmarks to analyze
        metric:
            - 'MAE' - Mean Absolute Error
            - 'RMSE' - Root Mean Squared Error
            - 'PCC' - Pearson's Correlation Coefficient
            - 'CCC' - Concordance Correlation Coefficient
            - 'SNR' - Signal to Noise Ratio
        approach:
            - 'frequentist' - (default) Use frequentist hypotesis tests for the analysis
            - 'bayesian' - Use bayesian hypotesis tests for the analysis
        print_report:
            - 'True' - (default) print a report of the hypotesis testing procedure
            - 'False' - Doesn't print any report

    Returns:
        Y_df: A pandas DataFrame containing the data on which the statistical analysis has been performed
        fig : A matplotlib figure displaying the outcome of the statistical analysis (an empty figure if the wilcoxon test has been chosen)
    """
    metric = metric.upper()
    assert metric in self.available_metrics, 'Error! Available metrics are ' + str(self.available_metrics)

    # -- Landmark(s) 
    if landmarks is not None:
        if not set(landmarks).issubset(set(self.landmarks)):
            raise ValueError("Some method is wrong!")
        else:
            self.landmarks = landmarks

    assert approach == 'frequentist' or approach == 'bayesian', "Approach should be 'frequentist' or bayesian, not " + str(approach)

    # -- set metric
    self.metric = metric
    self.mag = self.metricSort[metric]

    # -- get data from dataset(s)
    if self.multidataset:
        Y = self.__getData()
    else:
        Y = self.__getDataMono()
    self.ndataset = Y.shape[0]

    if metric == 'MAE' or metric == 'RMSE':
        order = 'ascending'
    else:
        order = 'descending'

    lmk_names = ['-'.join(x) for x in self.landmarks]
    Y_df = pd.DataFrame(Y, columns=lmk_names)

    results = autorank(Y_df, alpha=0.05, order=order, verbose=False, approach=approach)
    self.stat_result = results
    self.use_stats_pipeline = True

    if approach == 'bayesian':
        res_df = results.rankdf.iloc[:, [0, 1, 4, 5, 8]]
        print(res_df)

    if print_report:
        print(' ')
        create_report(results)
        print(' ')

    fig = plt.figure(figsize=(12, 5))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    _, ax = self.computeCD(approach=approach, ax=ax)

    return Y_df, fig

  def SignificancePlot(self, landmarks=None, metric='MAE'):
    """
    Returns a significance plot of the results of hypotesis testing
    """

    # -- Landmars(s) 
    if landmarks == None:
      landmarks = self.landmarks
    else:
      ## TODO
      if not any([x in lm for lm in self.landmarks for x in landmarks]):
        raise ValueError("Some landmarks is wrong!")
      else:
        self.landmarks = landmarks

    # -- set metric
    self.metric = metric
    self.mag = self.metricSort[metric]

    # -- get data from dataset(s)
    if self.multidataset:
      Y = self.__getData()
    else:
      Y = self.__getDataMono()

    # -- Significance plot, a heatmap of p values
    lmk_names = ['-'.join(x) for x in self.landmarks]
    if self.__any_equal(lmk_names):
      lmk_names = self.landmarks
    Ypd = pd.DataFrame(Y, columns=lmk_names)
    ph = sp.posthoc_nemenyi_friedman(Ypd)
    cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
                    'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.85, 0.35, 0.04, 0.3]}

    fig = plt.figure(figsize=(10, 7))
    ax, cbar = sp.sign_plot(ph, cbar=True, **heatmap_args)
    ax.set_title('p-vals')
    return fig

  def computeCD(self, ax=None, avranks=None, numDatasets=None, alpha='0.05', display=True, approach='frequentist'):
    """
    Returns critical difference and critical difference diagram for Nemenyi post-hoc test if the frequentist approach has been chosen
    Returns a Plot of the results of bayesian significance testing otherwise
    """
    cd = self.stat_result.cd
    if display and approach == 'frequentist':
      stats_fig = plot_stats(self.stat_result, allow_insignificant=True, ax=ax)
    elif display and approach == 'bayesian':
      stats_fig = self.plot_bayesian_res(self.stat_result)
    return cd, stats_fig

  def plot_bayesian_res(self, stat_result):
    """
    Plots the results of bayesian significance testing
    """
    dm = stat_result.decision_matrix.copy()
    cmap = ['1', '#fb6a4a', '#08306b', '#4292c6']  # , '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
                    'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.85, 0.35, 0.04, 0.3]}
    dm[dm == 'inconclusive'] = 0
    dm[dm == 'smaller'] = 1
    dm[dm == 'larger'] = 2
    np.fill_diagonal(dm.values, -1)

    pl, ax = plt.subplots()
    ax.imshow(dm.values.astype(int), cmap=ListedColormap(cmap))
    labels = list(dm.columns)
    # Major ticks
    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_yticks(np.arange(0, len(labels), 1))
    # Labels for major ticks
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_yticklabels(labels)
    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.set_title('Metric: ' + self.metric)
    cbar_ax = ax.figure.add_axes([0.85, 0.35, 0.04, 0.3])
    cbar = ColorbarBase(cbar_ax, cmap=ListedColormap(cmap), boundaries=[0, 1, 2, 3, 4])
    cbar.set_ticks(np.linspace(0.5, 3.5, 4))
    cbar.set_ticklabels(['None', 'equivalent', 'smaller', 'larger'])
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('0.5')
    cbar.ax.tick_params(size=0)
    return pl

  def displayBoxPlot(self, landmarks=None, metric='MAE', scale=None, title=True):
    """
    Shows the distribution of populations with box-plots 
    """
    metric = metric.upper()

    # -- Landmark(s) 
    if landmarks is None:
        landmarks = self.landmarks
    else:
      if not any([x in lm for lm in self.landmarks for x in landmarks]):
        raise ValueError("Some landmarks is wrong!")
      else:
        self.landmarks = landmarks

    # -- set metric
    self.metric = metric
    self.mag = self.metricSort[metric]
    if scale == None:
      scale = self.scale[metric]

    # collect data
    Y = self.__getData()

    # -- display box plot
    fig = self.boxPlot(landmarks, metric, Y, scale=scale, title=title)
    return fig

  def boxPlot(self, landmarks, metric, Y, scale, title):
    """
    Creates the box plot 
    """

    #  Y = mat(n-datasets,k-landmarks)   

    k = len(landmarks)

    offset = 50
    fig = go.Figure()
    landmarks = ['-'.join(x) for x in landmarks]

    for landmark in landmarks:
        yd = np.array(Y[landmark])
        name = landmark
        # -- set color for box
        if metric == 'MAE' or metric == 'RMSE' or metric == 'TIME_REQUIREMENT' or metric == 'SNR':
            med = np.median(yd)
            col = str(min(200, 5 * int(med) + offset))
        if metric == 'CC' or metric == 'PCC' or metric == 'CCC':
            med = 1 - np.abs(np.median(yd))
            col = str(int(200 * med) + offset)

        # -- add box 
        fig.add_trace(go.Box(
            y=yd,
            name=name,
            boxpoints='all',
            jitter=.7,
            # whiskerwidth=0.2,
            fillcolor="rgba(" + col + "," + col + "," + col + ",0.5)",
            line_color="rgba(0,0,255,0.5)",
            marker_size=2,
            line_width=2)
        )

    # if title:
    #     tit = "Metric: " + metric
    #     top = 40
    # else:
    #     tit = ''
    #     top = 10

    # fig.update_layout(
    #     title=tit,
    #     yaxis_type=scale,
    #     xaxis_type="category",
    #     yaxis=dict(
    #         autorange=True,
    #         showgrid=True,
    #         zeroline=True,
    #         # dtick=gwidth,
    #         gridcolor='rgb(255,255,255)',
    #         gridwidth=.1,
    #         zerolinewidth=2,
    #         titlefont=dict(size=30)
    #     ),
    #     font=dict(
    #         family="monospace",
    #         size=16,
    #         color='rgb(20,20,20)'
    #     ),
    #     margin=dict(
    #         l=20,
    #         r=10,
    #         b=20,
    #         t=top,
    #     ),
    #     paper_bgcolor='rgb(250, 250, 250)',
    #     plot_bgcolor='rgb(243, 243, 243)',
    #     showlegend=False
    # )

    # fig.show()
    return fig

  def saveStatsData(self, landmarks=None, metric='MAE', outfilename='statsData.csv'):
    """
    Saves statistics of data on disk 
    """
    Y = self.getStatsData(landmarks=landmarks, metric=metric, printTable=False)
    np.savetxt(outfilename, Y)

  def getStatsData(self, landmarks=None, metric='MAE', printTable=True):
    """
    Computes statistics of data 
    """
    # -- Landmark(s)
    if landmarks == None:
        landmarks = self.landmarks
    else:
        if not any([x in lm for lm in self.landmarks for x in landmarks]):
            raise ValueError("Some landmarks is wrong!")
        else:
            self.landmarks = landmarks

    # -- set metric
    self.metric = metric
    self.mag = self.metricSort[metric]

    # -- get data from dataset(s)
    #    return Y = mat(n-datasets,k-landmarks)   
    if self.multidataset:
        Y = self.__getData()
    else:
        Y = self.__getDataMono()

    # -- add median and IQR
    I = ss.iqr(Y, axis=0)
    M = np.median(Y, axis=0)
    Y = np.vstack((Y, M))
    Y = np.vstack((Y, I))

    lmk_names = ['-'.join(x) for x in self.landmarks]
    dataseNames = self.datasetNames
    dataseNames.append('Median')
    dataseNames.append('IQR')
    df = pd.DataFrame(Y, columns=lmk_names, index=dataseNames)
    if printTable:
        display(df)

    return Y, df

  def __remove_outliers(self, df, factor=3.5):
    """
    Removes the outliers. A data point is considered an outlier if
    lies outside factor times the inter-quartile range of the data distribution 
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - factor * IQR)) | (df > (Q3 + factor * IQR))).any(axis=1)]
    return df_out

  def get_data(self, metric):
    self.metric = metric
    self.mag = self.metricSort[metric]
    return self.__getData()

  def __getData(self):

    mag = self.mag
    metric = self.metric
    landmarks = self.landmarks
    landmark_names = ['-'.join(x) for x in self.landmarks]

    # -- loop on datasets
    Y = dict.fromkeys(['-'.join(x) for x in self.landmarks])
    for i, frame in enumerate(self.dataFrame):

      # -- loop on landmarks
      for lmk_idx, landmark in enumerate(landmarks):
          y = []
          vals = frame[frame['landmarks'].apply(lambda x: x == landmark)][metric]
          for v in vals:
            if not math.isnan(v):
              y.append(v.item(0))

          # add to dict
          if Y[landmark_names[lmk_idx]] is None:
            Y[landmark_names[lmk_idx]] = y
          else:
            Y[landmark_names[lmk_idx]] += y

    return Y

  def __getLandmarks(self):

    lmks = []
    dataFrame = []
    N = len(self.datasetsList)

    # -- load dataframes
    self.datasetNames = []
    for file in self.datasetsList:
        filename = self.path + file
        self.datasetNames.append(file)
        data = pd.read_hdf(filename)
        # unique landmarks values
        lmks.append(unique_list_values(data, 'landmarks'))
        dataFrame.append(data)

    if not self.join_data:
        # -- landmarks names intersection among datasets
        landmarks = lmks[0]
        if N > 1:
            for m in range(1, N - 1):
                landmarks.intersection(lmks[m])
        landmarks = list(landmarks)
    else:
        print("landmarks", landmarks)
        if sorted(list(set(landmarks))) != sorted(landmarks):
            raise ("Found multiple landmarks with the same name... Please ensure using different names for each landmark when 'join_data=True'")
    
    self.landmarks = landmarks
    self.dataFrame = dataFrame


# Get unique list values in df column
def unique_list_values(df, column_name):
    seen_values = set()

    for row in df[column_name]:
        if tuple(row) not in seen_values:
            seen_values.add(tuple(row))

    return [list(x) for x in seen_values]