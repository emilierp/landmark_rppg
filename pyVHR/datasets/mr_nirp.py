import xml.etree.ElementTree as ET
import numpy as np
from os import path
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal
import scipy
import os
import glob
import re
import cv2


class MR_NIRP(Dataset):
    """
    MR_NIRP Dataset

    .. MR_NIRP dataset structure:
    .. ---------------------------
    ..    datasetDIR/
    ..    |
    ..    |-- vidDIR1/
    ..    |   |-- videoFile1.avi
    ..    |
    ..    |...
    ..    |
    ..    |-- vidDIRM/
    ..        |-- videoFile1.avi
    """
    name = 'MR_NIRP'
    signalGT = 'BVP'          # GT signal type
    numLevels = 2             # depth of the filesystem collecting video and BVP files
    numSubjects = 1           # number of subjects
    video_EXT = 'avi'         # extension of the video files
    frameRate = 30            # vieo frame rate
    VIDEO_SUBSTRING = 'Subject'  # substring contained in the filename
    SIG_EXT = '.mat'           # extension of the BVP files
    SIG_SUBSTRING = 'pulseOx'   # substring contained in the filename
    SIG_SampleRate = 60       # sample rate of the BVP files

    def readSigfile(self, filename):

        bvp_elements =  scipy.io.loadmat(filename)['pulseOxRecord'][0]
        if bvp_elements[0].ndim == 0:
            bvp = bvp_elements
        else:
            bvp = [bvp_elements[i][0][0] for i in range(len(bvp_elements))]
        data = np.array(bvp)
        
        # n_bvp_samples = len(bvp)
        # last_bvp_time = int((n_bvp_samples*1000)/self.SIG_SampleRate)
        # print("bvp: ",n_bvp_samples, last_bvp_time)

        # vid_npy_filename = path.join(path.dirname(
        #     filename), 'pulseOx.npy')
        # vid_npy = np.load(vid_npy_filename, allow_pickle=True)
        # last_vid_time = int((vid_npy.shape[-1]*1000)/self.frameRate)
        # print("last vid time",last_vid_time)

        # diff = ((last_bvp_time - last_vid_time)/1000)
        # print("diff", diff)

        # assert diff >= 0, 'Unusable data.'

        # #print("Skipping %.2f seconds..." % diff)

        # diff_samples = round(diff*self.SIG_SampleRate)
        # print("Diff samples", diff_samples)

        # data = np.array(bvp[diff_samples:])
        # print("data:",data.shape)

        return BVPsignal(data, self.SIG_SampleRate)
    

    def loadFilenames(self):
        """
        Load dataset file names and directories of frames: 
        define vars videoFilenames and BVPFilenames
        """

        # # check if video files exist or create them from images
        # for root, dirs, files in os.walk(self.videodataDIR):
        #     for d in dirs:
        #         dirname = os.path.join(root, d)
        #         if not dirname.endswith('RGB_corrected'):
        #              continue
        #         if not glob.glob(dirname +'/*.avi'):
        #             #create videofile from images
        #             frames = self.__loadFrames(dirname)
        #             fps = self.frameRate
        #             width = frames[0].shape[1]
        #             height = frames[0].shape[0]
        #             #fourcc = cv2.VideoWriter_fourcc(*'MPNG')
        #             writer = cv2.VideoWriter(dirname + '/' + dirname.split('\\')[-2] + '.avi', 0, fps, (width, height))
        #             for frame in frames:
        #                 writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #             writer.release()
        
        # -- loop on the dir struct of the dataset getting filenames
        for root, dirs, files in os.walk(self.videodataDIR):
            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)

                # -- select video
                if filename.endswith(self.video_EXT) and (name.find(self.VIDEO_SUBSTRING) >= 0):
                    self.videoFilenames.append(filename)

        # -- loop on the dir struct of the dataset getting BVP filenames
        for root, dirs, files in os.walk(self.BVPdataDIR):
            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)
                # -- select signal
                if filename.endswith(self.SIG_EXT) and (name.find(self.SIG_SUBSTRING) >= 0):
                    self.sigFilenames.append(filename)

        # -- number of videos
        self.numVideos = len(self.videoFilenames)

    def __sort_nicely(self, l): 
        """ Sort the given list in the way that humans expect. 
        """ 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        l.sort( key=alphanum_key )
        return l

    def __loadFrames(self, directorypath):
        # -- get filenames within dir
        f_names = self.__sort_nicely(os.listdir(directorypath))
        frames = []
        for n in range(len(f_names)):
            filename = os.path.join(directorypath, f_names[n])
            frames.append(cv2.imread(filename)[:, :, ::-1])
        
        frames = np.array(frames)
        return frames

