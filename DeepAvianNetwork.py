# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\alamd\Desktop\UI\guitrial.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from pydub import AudioSegment
import scipy.io.wavfile 
import numpy as np
import pandas as pd 
import glob
import os
from pathlib import Path
from tqdm import tqdm 
import csv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
import librosa, librosa.display
import matplotlib
#matplotlib.use('Agg') # No pictures displayed 
import librosa
import librosa.display
import librosa
import matplotlib.colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import  QFileDialog


from numpy import expand_dims
from keras.preprocessing.image import img_to_array, load_img
import umap
import umap.plot
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fclusterdata
import shutil 
from scipy.signal import find_peaks
import scipy
import umap
from sklearn.preprocessing import RobustScaler
import umap.plot
import umap


from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from sklearn.model_selection import train_test_split
from numpy import expand_dims


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.keras import regularizers


class Ui_MainWindow(object):
    
    def _open_file_dialog(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit.setText('{}'.format(directory)) 
        
    def _open_file_dialog_CI(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_CI.setText('{}'.format(directory)) 
    
    def _open_file_dialog_SM(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_SM.setText('{}'.format(directory)) 
        
  #  def _open_file_dialog_SSM(self):
   #     directory = str(QtWidgets.QFileDialog.getExistingDirectory())
    #    self.FolderBrowse_lineEdit_SM.setText('{}'.format(directory)) 
        
    def _open_file_dialog_SSM(self):
        directory = QtWidgets.QFileDialog.getOpenFileNames()
        self.FolderBrowse_lineEdit_SM.setText('{}'.format(directory))     
    
    def _open_file_dialog_CMI(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_CMI.setText('{}'.format(directory)) 
    
    def _open_file_dialog_SC(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_SC.setText('{}'.format(directory)) 
        
    def _open_file_dialog_MC(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_MC.setText('{}'.format(directory)) 

    def _open_file_dialog_NN(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_NN.setText('{}'.format(directory)) 

    def _open_file_dialog_NN2(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_NN2.setText('{}'.format(directory)) 
        
    def _open_file_dialog_NN3(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_NN3.setText('{}'.format(directory)) 
        
    def _open_file_dialog_NNM(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_NNM.setText('{}'.format(directory)) 

    def _open_file_dialog_NNM2(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_NN2.setText('{}'.format(directory)) 
        
    def _open_file_dialog_NNM3(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.FolderBrowse_lineEdit_NNM3.setText('{}'.format(directory)) 
    
    
    
    def Segment(self):        
        From = int(self.FromDays_lineEdit.text())
        To = int(self.ToDays_lineEdit.text())

        if self.LowerThreshold.text() == "":
            self.LowerThreshold.setText("10000")
        
        foldername= self.FolderBrowse_lineEdit.text() + '/'
        
        header = (['Filename','SegmentFilename', 'Onset', 'Offset','Duration (ms)'])    
    

        Times= []
        
        Start=From
        End= To+1
        
        
        for i in tqdm(range(Start, End)):
            
            Times= []
            foldername= self.FolderBrowse_lineEdit.text() + '/'

            dayname= str(i)
            folderday= os.path.join(foldername, dayname)
            waves= '*wav'
            folderday1= os.path.join(folderday, waves)
            filenames = sorted(glob.glob(folderday1))
            audiofile = filenames
            path = (folderday)
            output_folder= Path(foldername+dayname)
            output_dir = os.path.join(output_folder, 'Segments/')
            output_csv = os.path.join(foldername,dayname,'SegmentTimes.csv')

            with open(output_csv, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(header)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            for f in tqdm(audiofile):
                
                sr, x = scipy.io.wavfile.read(f)
        
                times = np.arange(len(x))/float(sr)
                
                fx= x/float(1)
        
                chunks = np.array_split(fx, (len(x)/10))
        
                variance=[]
        
                for i in chunks:
                    var= np.var(i)
                    variance.append(var)
        
        
                gaussianvariance=gaussian_filter(variance, sigma=10)
                thresholdmin= float(self.LowerThreshold.text())

                threshold= thresholdmin+7000
                
                gaussianvariance[gaussianvariance > threshold] = threshold
                gaussianvariance[gaussianvariance < thresholdmin] = thresholdmin
        
                
                chunks2 = librosa.util.frame(gaussianvariance, frame_length=10, hop_length=20,axis=0)
                
                mean=[]
                for i in chunks2:
                    m= np.mean(i)
                    mean.append(m)
        
                
                diff= np.diff(mean)
                peaks, _ = find_peaks(diff,distance=10, height=((np.percentile(diff, 97))))
                if len(peaks) > 0:
            
                    peaks= peaks - 1
        
                    
                    diff2= np.diff(mean)
                    diff2[diff2 > 0] = 0
                    
                    mins, _ =find_peaks(diff2*-1,distance=10, height=((np.percentile(diff2, 97))))
                    mins=np.append(mins,(peaks[-1]+1))
            
            
            
                    mins1= np.array([mins[mins > i].min() for i in peaks])
                    mins1= mins1 + 1
            
            
            
                    
                    peaks= peaks * 20
                    mins1= mins1 * 20
                    
           
                    
                    gaussianvariance=np.repeat(gaussianvariance, 10)
                    variance=np.repeat(variance, 10)
            
                    maybe=[]
                    for i in range(len(peaks)):
                        maybe.append(peaks[i]*10)
                        maybe.append(mins1[i]*10)
                        
            
                    
                    lengthinseconds=len(x)/44100
                    lengthofeachelement= lengthinseconds/len(x)
                    yvalues=np.arange(0, len(gaussianvariance),1)
                    timesm= np.arange(0, lengthinseconds,lengthofeachelement )
                    timesofvariance= yvalues
                    listofframe=[]
                    for i in range(len(timesofvariance)):
                        A=timesofvariance[i]     
                        B=timesm[A]    
                        listofframe.append(B)
            
                    sound_times=[]
                    #first = 0
                    #sound_times.append(first)
                    for i in range(len(maybe)):
                        A=maybe[i]            
                        B=listofframe[A]    
                        sound_times.append(B)
                    
                    onset_offset= pd.DataFrame(sound_times)
                    onset_offset.columns= ['Onset']    
                    onset_values = pd.DataFrame(onset_offset['Onset'].iloc[0::2].values)                           
                    offset_values= pd.DataFrame(onset_offset['Onset'].iloc[1::2].values)
                    onset_offset_times = pd.concat([onset_values,offset_values], axis =1)
                    onset_offset_times.columns= ['Onset', 'Offset']
                    onset_offset_times = onset_offset_times.dropna()
                       
                    seg_lims = onset_offset_times.values.tolist()
                    seglimits = np.asarray(seg_lims)
                    fdc= (len(seglimits[:]))
                    segmentlimits=pd.DataFrame(np.array(seglimits), columns=['Start', 'Stop'])
                    newseglimitsStart=[]
                    newseglimitsStop=[]
                    for index, row in segmentlimits.iterrows():
                        Duration=row.Stop-row.Start
                        if Duration > 0.04:
                            newseglimitsStart.append(row.Start)
                            newseglimitsStop.append(row.Stop)
                    seglimits=np.vstack((newseglimitsStart,newseglimitsStop)).T
                    if fdc > 0 :
                        sl= pd.DataFrame(np.array(seglimits), columns=['Start', 'Stop'])
                        times= [None] * 5
                        for index, row in sl.iterrows():
                            #SegmentStart = (row.Start *1000)-20
                            #SegmentStop = (row.Stop * 1000)+25
                            SegmentStart = (row.Start * 1000)
                            if SegmentStart < 0:
                                SegmentStart= np.float64(0)
                            SegmentStop = (row.Stop * 1000)
                            SegmentStart2 = row.Start
                            SegmentStop2 = row.Stop
                            SegmentStart1 = SegmentStart
                            SegmentStart1 = SegmentStart1.clip(min=0) 
                            SegmentStart1 = SegmentStart1
                            SegmentStart1 = SegmentStart1/10
                            SegmentedSong = AudioSegment.from_file(f)
                            SegmentedSong = SegmentedSong[SegmentStart:SegmentStop]
                            i= index
                            SegmentSongName = (output_dir + f[len(path):-4] + "_{}.wav".format(i))
                            SegmentedSong.export(SegmentSongName, format="wav")
                            sr, x = scipy.io.wavfile.read(SegmentSongName)
                            Duration= SegmentStop2-SegmentStart2
            
                            SegmentStart1 = SegmentStart1.clip(min=0)  
                            times[0] = f
                            times[1] = SegmentSongName
                            times[2] = SegmentStart2
                            times[3] = SegmentStop2
                            times[4] = Duration
                 
                            with open(output_csv, 'a', newline='') as v:
                                writer = csv.writer(v)
                                writer.writerows([times])
                            Times.append(times)

    def ViewMotifSyll(self):
        filename = QFileDialog.getOpenFileName()
        f = filename[0]


        
        sr, x = scipy.io.wavfile.read(f)

        
        fx= x/float(1)

        chunks = np.array_split(fx, (len(x)/10))

        variance=[]

        for i in chunks:
            var= np.var(i)
            variance.append(var)        


        gaussianvariance=gaussian_filter(variance, sigma=10)

        
        chunks2 = librosa.util.frame(gaussianvariance, frame_length=10, hop_length=20,axis=0)
        
        mean=[]
        for i in chunks2:
            m= np.mean(i)
            mean.append(m)
        
        gaussianvariance2=gaussian_filter(variance, sigma=10)
        chunks3 = librosa.util.frame(gaussianvariance2, frame_length=10, hop_length=20,axis=0)
        mean2=[]
        
        for i in chunks3:
            m= np.mean(i)
            mean2.append(m)
        
        
        plt.plot(mean2)     
        #plt.plot(peaks, gaussianvariance2[peaks], "x")
        
        #for xc in peaks:
        #    plt.axvline(x=xc, color='red')
            
        #for xc in mins1:
        #    plt.axvline(x=xc, color='orange')
        
        plt.show()


    def CreateImages(self):
        matplotlib.use('Agg')
        From = int(self.FromDays_lineEdit_CI.text())
        To = int(self.ToDays_lineEdit_CI.text())
        
        
        foldername= self.FolderBrowse_lineEdit_CI.text() + '/'
        
        norm = matplotlib.colors.Normalize(-1,1)
        colors = [[norm(-1.0), "black"],
                  [norm(-0.5), "black"],
                  [norm( 0.0), "midnightblue"],
                  [norm( 0.2), "royalblue"],
                  [norm( 0.5), "white"],
                  [norm( 1.0), "red"]]
        
        
        newcmp = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        
        Start=From
        End= To+1
        
        for i in tqdm(range(Start, End)):
            
            foldername2= foldername+str(i)
            foldername2= foldername2+str('/Segments')
            entries = Path(foldername2)
            output_dir = os.path.join(entries, 'Images/')
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
                
            for entry in tqdm(os.scandir(entries)):
                if entry.name.endswith(".wav"):
                    x, fs = librosa.load(entry)
                    Duration= librosa.get_duration(y=x, sr=44100)
                    if Duration > 0: 
                    #if Duration > 0.04 and Duration < 0.8:                        
                        plt.figure(figsize=(2.44, 2.44), dpi=100)
                        plt.axis('off') # no axis
                        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
                        fft= np.abs(librosa.stft(y=x, n_fft= 512, win_length=256, hop_length=8))
                        if len(fft[1]) > 1500:
                            fft=fft[:257,:1500]
                        else:
                            fft=fft 
                        X_sample = librosa.amplitude_to_db(fft)
                        result=np.asarray(np.min(X_sample))
                        result = np.full((257,1500),result)
                        result[:,:X_sample.shape[1]] = X_sample
                        librosa.display.specshow(result, sr = fs, fmax=80000,  cmap=newcmp)
                        file, ext = os.path.splitext(entry.name)
                        plt.savefig(output_dir + file + '.jpg'.format(output_dir), bbox_inches=None, pad_inches=0)
                        plt.close()
            
    def ViewPeaks(self):
        filename = QFileDialog.getOpenFileName()
        f = filename[0]
        sr, x = scipy.io.wavfile.read(f)
        
        fx= x/float(1)

        chunks = np.array_split(fx, (len(x)/10))

        variance=[]

        for i in chunks:
            var= np.var(i)
            variance.append(var)
        
        gaussianvariance=gaussian_filter(variance, sigma=10)

        
        chunks2 = librosa.util.frame(gaussianvariance, frame_length=10, hop_length=20,axis=0)
        
        mean=[]
        for i in chunks2:
            m= np.mean(i)
            mean.append(m)

        
        diff= np.diff(mean)
        distance=40
        
        topnumber= float(self.UpperBound_lineEdit_SM.text())
        #99.5
        
        peaks, _ = find_peaks(diff,distance=distance, height=((np.percentile(diff, topnumber))))
        
        gaussianvariance2=gaussian_filter(variance, sigma=10)
        chunks3 = librosa.util.frame(gaussianvariance2, frame_length=10, hop_length=20,axis=0)
        mean2=[]
        
        for i in chunks3:
            m= np.mean(i)
            mean2.append(m)
        
        

        #peaks= peaks 
        
        plt.plot(mean2)     
        plt.plot(peaks, gaussianvariance2[peaks], "x")
        plt.show()
        
    def ViewMotif(self):
        filename = QFileDialog.getOpenFileName()
        f = filename[0]
        distance1 = int(self.End_SongFile_lineEdit_SM.text())
        
        peaksminus = int(self.Beginning_SongFile_lineEdit_SM.text())

        
        sr, x = scipy.io.wavfile.read(f)

        
        fx= x/float(1)

        chunks = np.array_split(fx, (len(x)/10))

        variance=[]

        for i in chunks:
            var= np.var(i)
            variance.append(var)        


        gaussianvariance=gaussian_filter(variance, sigma=10)

        
        chunks2 = librosa.util.frame(gaussianvariance, frame_length=10, hop_length=20,axis=0)
        
        mean=[]
        for i in chunks2:
            m= np.mean(i)
            mean.append(m)

        
        diff= np.diff(mean)
        distance=40
        
        topnumber= float(self.LowerThreshold.text())
        #99.5
        
        peaks, _ = find_peaks(diff,distance=distance, height=((np.percentile(diff, topnumber))))
        
        gaussianvariance2=gaussian_filter(variance, sigma=10)
        chunks3 = librosa.util.frame(gaussianvariance2, frame_length=10, hop_length=20,axis=0)
        mean2=[]
        
        for i in chunks3:
            m= np.mean(i)
            mean2.append(m)
        
        
        mins1= peaks + distance1
        #plt.plot(mean2)       
        #plt.plot(mins1, gaussianvariance2[mins1], "x")
        
        
        peaks= peaks - peaksminus
        
        plt.plot(mean2)     
        plt.plot(peaks, gaussianvariance2[peaks], "x")
        
        for xc in peaks:
            plt.axvline(x=xc, color='red')
            
        for xc in mins1:
            plt.axvline(x=xc, color='orange')
        
        plt.show()



    def MotifSegment(self):   
        From = int(self.FromDays_lineEdit_SM.text())
        To = int(self.ToDays_lineEdit_SM.text())
        
        if self.UpperBound_lineEdit_SM.text() == "":
            self.UpperBound_lineEdit_SM.setText("99.5")

        foldername= self.FolderBrowse_lineEdit_SM.text() + '/'
        
        header = (['Filename','SegmentFilename', 'Onset', 'Offset','Duration'])    
        #output_csv = os.path.join(foldername,'SegmentTimes.csv')
        
        
        #with open(output_csv, 'w') as file:
            #writer = csv.writer(file)
            #writer.writerow(header)
        
        Times= []
        
        Start=From
        End= To+1
        
        for i in tqdm(range(Start, End)):
            Times= []
            dayname= str(i)
        
            folderday= os.path.join(foldername, dayname)
            waves= '*wav'
            folderday1= os.path.join(folderday, waves)
            filenames = sorted(glob.glob(folderday1))
            audiofile = filenames
            path = (folderday)
            output_dir = foldername+dayname + '/Segments/Motif/'
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            output_csv = os.path.join(output_dir,'SegmentTimes.csv')
            with open(output_csv, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                
            for f in tqdm(audiofile):

                sr, x = scipy.io.wavfile.read(f)
        
                times = np.arange(len(x))/float(sr)
                
                fx= x/float(1)
        
                chunks = np.array_split(fx, (len(x)/10))
        
                variance=[]
        
                for i in chunks:
                    var= np.var(i)
                    variance.append(var)
        
                begvarmean= np.mean(variance[:160]) 
                
        
        
                gaussianvariance=gaussian_filter(variance, sigma=10)

                chunks2 = librosa.util.frame(gaussianvariance, frame_length=10, hop_length=20,axis=0)
                
                mean=[]
                for i in chunks2:
                    m= np.mean(i)
                    mean.append(m)
        
                
                diff= np.diff(mean)
                distance=40
                topnumber= float(self.UpperBound_lineEdit_SM.text())
                peaks, _ = find_peaks(diff,distance=distance, height=((np.percentile(diff, topnumber))))
                
                gaussianvariance2=gaussian_filter(variance, sigma=10)
                chunks3 = librosa.util.frame(gaussianvariance2, frame_length=10, hop_length=20,axis=0)
                mean2=[]
                for i in chunks3:
                    m= np.mean(i)
                    mean2.append(m)
                

                
                if begvarmean > 100000:
                    peaks= np.pad(peaks, 1, 'constant', constant_values=(1))
                    peaks=peaks[:-1]
        
                if len(peaks) > 0:
                    
                    distance1 = int(self.End_SongFile_lineEdit_SM.text())
                    
                    mins1= peaks + distance1

                    peaksminus = int(self.Beginning_SongFile_lineEdit_SM.text())
                            

                    peaks= peaks - peaksminus

                    peaks= peaks * 20
                    mins1= mins1 * 20
                    
           
                    
                    gaussianvariance=np.repeat(gaussianvariance, 10)
                    variance=np.repeat(variance, 10)
            
                    maybe=[]
                    for i in range(len(peaks)):
                        maybe.append(peaks[i]*10)
                        maybe.append(mins1[i]*10)
                        
            
                    
                    lengthinseconds=len(x)/44100
                    lengthofeachelement= lengthinseconds/len(x)
                    yvalues=np.arange(0, len(gaussianvariance),1)
                    timesm= np.arange(0, lengthinseconds,lengthofeachelement )
                    timesofvariance= yvalues
                    listofframe=[]
                    for i in range(len(timesofvariance)):
                        A=timesofvariance[i]     
                        B=timesm[A]    
                        listofframe.append(B)
            
                    sound_times=[]
                    #first = 0
                    #sound_times.append(first)
                    for i in range(len(maybe)):
                        A=maybe[i]
                        if A< len(listofframe):
                            B=listofframe[A]    
                            sound_times.append(B)
                    
                    onset_offset= pd.DataFrame(sound_times)
                    onset_offset.columns= ['Onset']    
                    onset_values = pd.DataFrame(onset_offset['Onset'].iloc[0::2].values)                           
                    offset_values= pd.DataFrame(onset_offset['Onset'].iloc[1::2].values)
                    onset_offset_times = pd.concat([onset_values,offset_values], axis =1)
                    onset_offset_times.columns= ['Onset', 'Offset']
                    onset_offset_times = onset_offset_times.dropna()
                       
                    seg_lims = onset_offset_times.values.tolist()
                    seglimits = np.asarray(seg_lims)
                    fdc= (len(seglimits[:]))
                    segmentlimits=pd.DataFrame(np.array(seglimits), columns=['Start', 'Stop'])
                    newseglimitsStart=[]
                    newseglimitsStop=[]
                    for index, row in segmentlimits.iterrows():
                        Duration=row.Stop-row.Start
                        if Duration > 0.03:
                            newseglimitsStart.append(row.Start)
                            newseglimitsStop.append(row.Stop)
                    seglimits=np.vstack((newseglimitsStart,newseglimitsStop)).T
                    if fdc > 0 :
                        sl= pd.DataFrame(np.array(seglimits), columns=['Start', 'Stop'])
                        times= [None] * 5
                        for index, row in sl.iterrows():
                            #SegmentStart = (row.Start *1000)-20
                            #SegmentStop = (row.Stop * 1000)+25
                            SegmentStart = (row.Start * 1000)
                            if SegmentStart < 0:
                                SegmentStart= np.float64(0)
                            SegmentStop = (row.Stop * 1000)+15
                            SegmentStart2 = row.Start
                            SegmentStop3 = row.Stop
                            SegmentStart1 = SegmentStart
                            SegmentStart1 = SegmentStart1.clip(min=0) 
                            SegmentStart1 = SegmentStart1
                            SegmentStart1 = SegmentStart1/10
                            SegmentedSong = AudioSegment.from_file(f)
                            SegmentedSong = SegmentedSong[SegmentStart:SegmentStop]
                            i= index
                            SegmentSongName = (output_dir + f[len(path)+1:-4] + "_{}.wav".format(i))
                            SegmentedSong.export(SegmentSongName, format="wav")
                            sr, x = scipy.io.wavfile.read(SegmentSongName)
                            Duration= (SegmentStop3-SegmentStart2)*1000
            
                            SegmentStart1 = SegmentStart1.clip(min=0)  
                            times[0] = f
                            times[1] = SegmentSongName
                            times[2] = SegmentStart2
                            times[3] = SegmentStop3
                            times[4] = Duration
                 
                            with open(output_csv, 'a', newline='') as v:
                                writer = csv.writer(v)
                                writer.writerows([times])
                            Times.append(times)            
                
                
    def CreateMotifImages(self):
        matplotlib.use('Agg')

        From = int(self.FromDays_lineEdit_CMI.text())
        To = int(self.ToDays_lineEdit_CMI.text())
        
        
        foldername= self.FolderBrowse_lineEdit_CMI.text() + '/'
        
        norm = matplotlib.colors.Normalize(-1,1)
        colors = [[norm(-1.0), "black"],
                  [norm(-0.5), "black"],
                  [norm( 0.0), "midnightblue"],
                  [norm( 0.2), "royalblue"],
                  [norm( 0.5), "white"],
                  [norm( 1.0), "red"]]
        
        
        newcmp = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        
        Start=From
        End= To+1
        
        for i in tqdm(range(Start, End)):
            
            foldername2= foldername+str(i)
            foldername2= foldername2+str('/Segments/Motif')
            entries = Path(foldername2)
            output_dir = os.path.join(entries, 'Images/')
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
                
            for entry in tqdm(os.scandir(entries)):
                if entry.name.endswith(".wav"):
                    x, fs = librosa.load(entry)
                    Duration= librosa.get_duration(y=x, sr=44100)
                    if Duration > 0: 
                    #if Duration > 0.04 and Duration < 0.8:                        
                        plt.figure(figsize=(2.44, 2.44), dpi=100)
                        plt.axis('off') # no axis
                        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
                        fft= np.abs(librosa.stft(y=x, n_fft= 512, win_length=256, hop_length=8))
                        if len(fft[1]) > 3000:
                            fft=fft[:257,:3000]
                        else:
                            fft=fft 
                        X_sample = librosa.amplitude_to_db(fft)
                        result=np.asarray(np.min(X_sample))
                        result = np.full((257,3000),result)
                        result[:,:X_sample.shape[1]] = X_sample
                        librosa.display.specshow(result, sr = fs, fmax=80000,  cmap=newcmp)
                        file, ext = os.path.splitext(entry.name)
                        plt.savefig(output_dir + file + '.jpg'.format(output_dir), bbox_inches=None, pad_inches=0)
                        plt.close()
                            
    
    def SyllableCluster(self):
       
        image_width = 28
        image_height = 28
        channels = 3
        
        
        folder = self.FolderBrowse_lineEdit_SC.text() + '/'
        
        onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        wavfolder=  folder.split('Images')
        folder_name = wavfolder[0]
        extension = '.wav'
        pattern = os.path.join(folder_name, "*{}".format(extension))
        
        Segment = sorted(glob.glob(pattern))
        
        
        wavpathname= folder
        pathname = folder
        birdletter= self.BirdLetter_lineEdit_SC.text()
            
                
        fw= []
        dataset= []
        jpgfilenames=[]
        for f in tqdm(Segment):
            Features = []
        
            Filename = f
            
            x, sr = librosa.load(f)
            x= x/x.max()
            Duration = librosa.get_duration(y=x, sr=sr)
        
            spec=librosa.stft(x, win_length=256, hop_length=20)
            rms= librosa.feature.rms(S=spec,hop_length=20)
            peaks, _ = find_peaks(rms.flatten(), prominence=(np.percentile(rms,20)))
            if len(peaks) > 0:
            
                
            
                segmentjpg= Filename
                segmentjpg= segmentjpg[:-3]
                segmentjpg= segmentjpg +'jpg'
                segmentjpg= segmentjpg.split('Segments')
                segmentjpg= segmentjpg[0]+ 'Segments/Images' + segmentjpg[1]
                jpgfilenames.append(segmentjpg)
                        
                hop_length = 20
                frame_length = 1024
                
                zx= np.pad(x, (512, 512), 'constant')
                
                frames = librosa.util.frame(zx, frame_length=1024, hop_length=20,axis=0)
                    
                Features = []
               
                Filename = f
                 
                Goodness=[]
                for i in range(len(frames)):
                    spectrum = np.fft.fft(frames[i])
                    rceps = np.fft.ifft(np.log(np.abs(spectrum))).real
                    end= int((len(rceps)/2))
                    g= np.max(rceps[24:end]).tolist()
                    Goodness.append(g)
                    
                
                flatness = librosa.feature.spectral_flatness(x, win_length=256, hop_length=20)
                flatness= flatness.T.tolist()
                
        
                freqs, psd = scipy.signal.welch(x)
                gf = np.max(psd)
                psd=psd.tolist()
                
                gd= gf *((np.mean(rms))*((np.percentile(Goodness, 90)- np.mean(Goodness)))/((np.mean(flatness))))
                            
                #if gd > .02:
                
                power= spec **2
                logD=np.log(spec)
                gmean = np.exp(np.mean(logD, axis=0, keepdims=True))
                amean = np.mean(logD, axis=0, keepdims=True)
                evalues=((gmean/amean).real)
                Entropy= evalues.tolist()
                Entropy=Entropy[0]
                
                rms= rms.T[peaks[0]:]
        
                rms=rms.T
                rms= rms.tolist()
                
                mf=librosa.feature.spectral_rolloff(x, sr=sr, roll_percent=0.5,win_length=256, hop_length=20 ).tolist()
                
            
                Features.append('Filename')
                Features.append('Goodness')
                Features.append('Entropy')
                Features.append('RMS')
                Features.append('PSD')
                Features.append('MeanFrequency')
        
                
                Features[0] = Filename
                Features[1] = Goodness
                Features[2] = Entropy
                Features[3] = rms
                Features[4] = psd
                Features[5] = mf
        
                
                fw.append(Features)
        
        fww= []
        dataset= []
        
        for f in tqdm(jpgfilenames):
            Featuress = []
        
            Filename = f
            img = load_img(Filename)  # this is a PIL image
            img.thumbnail((image_width, image_height))
            
            
            # convert to numpy array
            x = img_to_array(img)  
            # expand dimension to one sample
            samples = expand_dims(x, 0)
            x = x / 255
            Image=x.flatten()
        
            dataset.append(Image)
            
            Featuress.append('Filename')
            Featuress.append('Image')
        
            
            Featuress[0] = Filename
            Featuress[1] = Image
        
            
            fww.append(Featuress)
            
            
        darray= np.array(dataset)
        d_mapper = umap.UMAP(random_state=42).fit(darray)
        transformer = RobustScaler().fit(darray)
        darraytrans=transformer.transform(darray)
        
        
        goodness=[]
        for i in range(len(fw)):
            s=fw[i]
            g=s[1]
            goodness.append(g)
        
        y=np.array([np.array(xi) for xi in goodness])
        max_len = max([len(x) for x in y])
        output = [np.pad(x, (0, max_len - len(x)), 'constant') for x in y]
        garray= np.array(output)
        
        transformer = RobustScaler().fit(garray)
        garraytrans=transformer.transform(garray)
         
        
        entropy=[]
        for i in range(len(fw)):
            s=fw[i]
            e=s[2]
            entropy.append(e)
        
        y=np.array([np.array(xi) for xi in entropy])
        max_len = max([len(x) for x in y])
        output = [np.pad(x, (0, max_len - len(x)), 'constant') for x in y]
        earray= np.array(output)
        
        transformer = RobustScaler().fit(earray)
        earraytrans=transformer.transform(earray)
        
        rms=[]
        for i in range(len(fw)):
            s=fw[i]
            r=s[3]
            r=r[0]
            rms.append(r)
        
        y=np.array([np.array(xi) for xi in rms])
        max_len = max([len(x) for x in y])
        medianlen= np.median([len(x) for x in y])
        output = [np.pad(x, (0, max_len - len(x)), 'constant') for x in y]
        rarray= np.array(output)
        rarray= rarray[:,:int(medianlen)]
        transformer = RobustScaler().fit(rarray)
        rarraytrans=transformer.transform(rarray)
        
        
        gg=[]
        for i in range(len(fw)):
            s=fw[i]
            z=s[4]
            gg.append(z)
        
        y=np.array([np.array(xi) for xi in gg])
        max_len = max([len(x) for x in y])
        output = [np.pad(x, (0, max_len - len(x)), 'constant') for x in y]
        ggarray= np.array(output)
        
        transformer = RobustScaler().fit(ggarray)
        ggarraytrans=transformer.transform(ggarray)
        
        mf=[]
        for i in range(len(fw)):
            s=fw[i]
            z=s[5]
            zz= z[0]
            mf.append(zz)
        
        y=np.array([np.array(xi) for xi in mf])
        max_len = max([len(x) for x in y])
        output = [np.pad(x, (0, max_len - len(x)), 'constant') for x in y]
        mfarray= np.array(output)
        
        transformer = RobustScaler().fit(mfarray)
        mfarraytrans=transformer.transform(mfarray)
        
        
        
        
        d_mapper = umap.UMAP(random_state=42).fit(darraytrans)
        g_mapper = umap.UMAP(random_state=42).fit(garraytrans)
        e_mapper = umap.UMAP(random_state=42).fit(earraytrans)
        r_mapper = umap.UMAP(random_state=42).fit(rarraytrans)
        gg_mapper = umap.UMAP(random_state=42).fit(ggarraytrans)
        mf_mapper = umap.UMAP(random_state=42).fit(mfarraytrans)
        
        
        
        pgezggunion_mapper = (g_mapper+ e_mapper +  r_mapper+ gg_mapper + mf_mapper +d_mapper) 
        umap.plot.points(pgezggunion_mapper)
        
        points = umap.plot._get_embedding(pgezggunion_mapper)
        
        #points = umap.plot._get_embedding(d_mapper)
        
        
        names=[]
        for i in range(len(fw)):
            s=fw[i]
            n=s[0]
            names.append(n)
            
        thresh = .4
        clusters = (fclusterdata(points, thresh, criterion="distance"))-1
        
        
        labels= clusters
        
        plt.scatter(*np.transpose(points), c=clusters)
        plt.axis("equal")
        title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
        plt.title(title)
        plt.show()
        
        clusters=labels
        
        
        
        names_k= np.vstack((names,clusters)).T
        
        names_kpd= pd.DataFrame(names_k)
        names_kpd.columns= (['Filename', 'Cluster'])
        
        
        for i in range(-1,max(labels)+1):
            path = pathname + '{}'.format(i)
            if not os.path.exists(path):
                os.mkdir(path)
        
        entries = Path(wavpathname)
        entries1 = Path(pathname)
        
        names_kpd['Cluster'] = names_kpd.Cluster.astype(int)
        
        names_kpd=names_kpd.groupby(['Cluster']).filter(lambda x: len(x) > 10)
        
        
        clusters= np.max(names_kpd.Cluster.astype(int))
        
        for _, row in names_kpd.iterrows():
            a= row['Filename']
            splitnamestochange= a.split(birdletter)
            splitnamestochangepointed= splitnamestochange[-1].split('w')
            src= [pathname + birdletter,splitnamestochangepointed[0]]
            changedfilename= ''.join(src)
            changedfilename= changedfilename+ 'jpg'
            output_dir = os.path.join(entries1, str(row[1]))
            file_path = Path(changedfilename)                    
            if os.path.exists(file_path):
                shutil.move(changedfilename, output_dir)
        
        
        for i in range(np.min(clusters),(np.max(clusters)+1)):
            path= pathname +'{}'.format(i)
            entries = Path(path)    
            segmentfiles= []
            segmentfilenames= []    
            for entry in tqdm(os.scandir(entries)):
                if entry.name.endswith(".jpg"):
                    a= entry.name
                    splitnamestochange= a.split('j')
                    src= [wavpathname,splitnamestochange[0],'wav']
                    changedfilename= ''.join(src)
                    segmentfiles.append(changedfilename)
                    segmentfilenames.append(splitnamestochange[0])
   
    def MotifCluster(self):
        image_width = 28
        image_height = 28
        channels = 3
        
        
        folder = self.FolderBrowse_lineEdit_MC.text() + '/'
        onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        dataset= []
        for _file in tqdm(onlyfiles):
            img = load_img((folder + "/" + _file))  # this is a PIL image
            img.thumbnail((image_width, image_height))
            
            
            # convert to numpy array
            x = img_to_array(img)  
            # expand dimension to one sample
            samples = expand_dims(x, 0)
            x = x / 255
            x=x.flatten()
        
            dataset.append(x)
        
        
        

        folder_name = folder
        extension = '.jpg'
        pattern = os.path.join(folder_name, "*{}".format(extension))
        
        Segment2 = sorted(glob.glob(pattern))
        
        wavpathname= folder
        pathname = folder
        birdletter= self.BirdLetter_lineEdit_MC.text()
        
        
        fw= []
        dataset= []
        
        for f in tqdm(Segment):
            Features = []
        
            Filename = f
            img = load_img(Filename)  # this is a PIL image
            img.thumbnail((image_width, image_height))
            
            
            # convert to numpy array
            x = img_to_array(img)  
            # expand dimension to one sample
            samples = expand_dims(x, 0)
            x = x / 255
            Image=x.flatten()
        
            dataset.append(Image)
            
            Features.append('Filename')
            Features.append('Image')
        
            
            Features[0] = Filename
            Features[1] = Image
        
            
            fw.append(Features)
            
            
        darray= np.array(dataset)
        d_mapper = umap.UMAP(random_state=42).fit(darray)
        umap.plot.points(d_mapper)
        
        points = umap.plot._get_embedding(d_mapper)
        
        
        names=[]
        for i in range(len(fw)):
            s=fw[i]
            n=s[0]
            names.append(n)
            
        thresh = .6
        clusters = (fclusterdata(points, thresh, criterion="distance"))-1
        
        
        labels= clusters
        
        plt.scatter(*np.transpose(points), c=clusters)
        plt.axis("equal")
        title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
        plt.title(title)
        plt.show()
        
        clusters=labels
        
        
        
        names_k= np.vstack((names,clusters)).T
        
        names_kpd= pd.DataFrame(names_k)
        names_kpd.columns= (['Filename', 'Cluster'])
        
        
        for i in range(0,max(labels)+1):
            path = pathname + '{}'.format(i)
            if not os.path.exists(path):
                os.mkdir(path)
        
        entries = Path(wavpathname)
        entries1 = Path(pathname)
        
        names_kpd['Cluster'] = names_kpd.Cluster.astype(int)
        
        names_kpd=names_kpd.groupby(['Cluster']).filter(lambda x: len(x) > 1)
        
        
        clusters= np.max(names_kpd.Cluster.astype(int))
        
        for _, row in names_kpd.iterrows():
            a= row['Filename']
            splitnamestochange= a.split(birdletter)
            splitnamestochangepointed= splitnamestochange[-1].split('w')
            src= [pathname + birdletter,splitnamestochangepointed[0]]
            changedfilename= ''.join(src)
            output_dir = os.path.join(entries1, str(row[1]))
            file_path = Path(changedfilename)                    
            if os.path.exists(file_path):
                shutil.move(changedfilename, output_dir)
        
        
        for i in range(np.min(clusters),(np.max(clusters)+1)):
            path= pathname +'{}'.format(i)
            entries = Path(path)    
            segmentfiles= []
            segmentfilenames= []    
            for entry in tqdm(os.scandir(entries)):
                if entry.name.endswith(".jpg"):
                    a= entry.name
                    splitnamestochange= a.split('j')
                    src= [wavpathname,splitnamestochange[0],'wav']
                    changedfilename= ''.join(src)
                    segmentfiles.append(changedfilename)
                    segmentfilenames.append(splitnamestochange[0])
    
    def PrepSyll(self):
                
        train_files = []
        y_train= []
        
        sfolders= int(self.folders_lineEdit_NN.text())#"FromDAYS"
        
        
        for i in range(0,(sfolders + 1)):
        
            syll = str(i)
            entries = Path(self.FolderBrowse_lineEdit_NN.text() + '/' + syll)
        
            script_dir = os.path.dirname(entries)
            output_dir = os.path.join(entries)#, 'Images/')
        
            folder = output_dir
        
            onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            for _file in onlyfiles:
                train_files.append(_file)
                clusters= i
                y_train.append(clusters)
                
        image_width = 28
        image_height = 28
        channels = 3
        flatdim= image_width*image_height*channels
        
        lentrainfiles= len(train_files)
        dataset = np.ndarray(shape=(lentrainfiles, image_height, image_width, channels),
                             dtype=np.float32)
        
        # example of horizontal shift image augmentation
        
        i=0
        v= 0    
        label= []
        
        for c in range(0,(sfolders+1)):
            syll = str(c)
            entries = Path(self.FolderBrowse_lineEdit_NN.text() + '/' + syll)
        
            script_dir = os.path.dirname(entries)
            output_dir = os.path.join(entries)#, 'Images/')
        
            folder = output_dir
            onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            for _file in tqdm(onlyfiles):
        # load the image
                img = load_img(folder + "/" + _file)  # this is a PIL image
                img.thumbnail((image_width, image_height))
                
                # convert to numpy array
                x = img_to_array(img)  
                x = x / 255
                dataset[i] = x
                i += 1
                a= y_train[v]
                a= [a]
                v+= 1
                label.extend(a)
        
        
        #Splitting 
        X_train, X_test, y_train, y_test = train_test_split(dataset, y_train, test_size=0.3, random_state=31)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=30)
        



        
        
        num_classes = sfolders + 1
        classes= num_classes
        
        def euclidean_distance(vects):
            x, y = vects
            sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, K.epsilon()))
        
        
        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)
        
        
        def contrastive_loss(y_true, y_pred):
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            margin = 1
            sqaure_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
        
        def trip_loss(y_true, y_pred):
            import tensorflow as tf
            return tf.contrib.losses.metric_learning.triplet_semihard_loss(y_true, y_pred) 
        
        def create_pairs(x, digit_indices):
            list1=list(range(num_classes))
            list2= list(range(num_classes))
            
            difpairs= [[a, b] for a in list1
                      for b in list2 if a != b] 
            likpairs=[[a, b] for a in list1
                      for b in list2 if a == b]
            
            pairs=[]
            labels=[]
            for i in range(len(likpairs)):
                y=likpairs[i][0]
                n=likpairs[i][1]
                likepairs = [[a, b] for a in digit_indices[y]  
                          for b in digit_indices[n] if a != b] 
                for v in range(len(likepairs)):
                    tryy=likepairs[v]
                    pairs+=[[x[tryy[0]], x[tryy[1]]]]
                    labels +=[1] 
            for i in range(len(difpairs)):
                y=difpairs[i][0]
                n=difpairs[i][1]
                differentpairs = [[a, b] for a in digit_indices[y]  
                          for b in digit_indices[n] if a != b]
                for v in range(len(differentpairs)):
                    tryy=differentpairs[v]
                    pairs+=[[x[tryy[0]], x[tryy[1]]]]
                    labels +=[0] 
            
            return np.array(pairs), np.array(labels)
        
        
        def create_base_network(input_shape):
            '''Base network to be shared (eq. to feature extraction).
            '''
            inputShape = (28,28, 3)
            chanDim = -1
            
            # define the model input
            inputs = tensorflow.keras.layers.Input(shape=inputShape)
                    
                    # Convolution Layer 1
            x = Conv2D(32, (3, 3), padding="same", activation="elu")(inputs)
            x = BatchNormalization(axis=chanDim)(x)
                    
                    # Convolution Layer 2
            x = Conv2D(32, (3, 3), padding="same", strides=(2, 2),activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
            
            x = Conv2D(32, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 3
            x = Conv2D(32, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
                    
                    # Convolution Layer 4
            x = Conv2D(32, (3, 3), strides=(2, 2), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
                    
            x = Conv2D(32, (3, 3), strides=(1, 1), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 5
            x = Conv2D(32, (3, 3), strides=(2, 2), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
            
            x = Conv2D(32, (3, 3), strides=(1, 1), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 6
            x = Conv2D(64, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
            
            x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
        
            x = Dense(1024, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
            
            x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
        
                    # Convolution Layer 7
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
                   
            x = Conv2D(128, (3, 3), padding="same",strides=(1, 1), activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 8
            x = Conv2D(128, (3, 3), strides=(2, 2), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
                  
                    # Convolution Layer 9
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
                    
                    # Convolution Layer 10
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
                    
                    # Convolution Layer 11
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = Flatten()(x)
            x = Dropout(0.2)(x)
              
            		# first (and only) set of FC => RELU layers
            x = Dense(2048, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
            x = Dense(1024, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
            x = Dense(512, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
               
            		# softmax classifier
            x = Dense(classes,activation="sigmoid",kernel_regularizer=regularizers.l2(0.01))(x)
              
            		# create the model
            model = Model(inputs, x)
            model.summary()
            return model
        
        def compute_accuracy(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.
            '''
            pred = y_pred.ravel() < 0.5
            return np.mean(pred == y_true)
        
        
        def accuracy(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.
            '''
            return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
        
        
        # the data, split between train and test sets
        input_shape = X_train.shape[1:]
        y_train1= np.asarray(y_train)
        y_test1= np.asarray(y_test)
        y_val1= np.asarray(y_val)
        
        from sklearn.utils.class_weight import compute_class_weight
        from keras.utils import np_utils  
        import os.path as path
        
        filename = self.FolderBrowse_lineEdit_NN3.text() +'/'+'newfile1.dat'
        # create training+test positive and negative pairs
        digit_indices = [np.where(y_train1 == i)[0] for i in range(num_classes)]
        tr_pairs, tr_y = create_pairs(X_train, digit_indices)
        tr = np.memmap(filename, dtype='float32', mode='w+', shape=(len(tr_y),2,28,28,3))
        for i in range(len(tr_y)):
            tr[i]=tr_pairs[i]
        del tr_pairs
        
        filename1 = self.FolderBrowse_lineEdit_NN3.text() +'/'+'newfile2.dat'
        digit_indices = [np.where(y_test1 == i)[0] for i in range(num_classes)]
        te_pairs, te_y = create_pairs(X_test, digit_indices)
        te = np.memmap(filename1, dtype='float32', mode='w+', shape=(len(te_y),2,28,28,3))
        for i in range(len(te_y)):
            te[i]=te_pairs[i]
        del te_pairs
        
        filename2 = self.FolderBrowse_lineEdit_NN3.text() +'/'+'newfile3.dat'
        digit_indices = [np.where(y_val1 == i)[0] for i in range(num_classes)]
        tv_pairs, tv_y = create_pairs(X_val, digit_indices)
        tv = np.memmap(filename2, dtype='float32', mode='w+', shape=(len(tv_y),2,28,28,3))
        for i in range(len(tv_y)):
            tv[i]=tv_pairs[i]
        del tv_pairs 
        
        # network definition
        #needs tensoflow1.14
        base_network = create_base_network(input_shape)
        
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        print(input_a)
        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        
        model = Model([input_a, input_b], distance)
        
        y= y_test + y_train
        nb_classes= num_classes
        ycat= np.float16(np_utils.to_categorical(y, nb_classes))
        y_flat = np.argmax(ycat, axis=1)
        
        class_weights = compute_class_weight(class_weight= 'balanced', classes= np.unique(y_flat),
                                            y= y_flat)
        
        
        class_weights = dict(zip(np.unique(y_flat), class_weights))
        
        es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
        
        import tensorflow as tf
        
        te_y= np.float32(te_y)
        tr_y= np.float32(tr_y)
        tv_y=np.float32(tv_y)
        te_y=tf.cast(te_y, tf.float32)
        
        
        
        # train
        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
        H = model.fit([tr[:, 0], tr[:, 1]], tr_y, shuffle=True, class_weight= class_weights,
                  batch_size=512,
                  epochs=20, validation_data=([tv[:, 0], tv[:, 1]], tv_y), callbacks= [es])
         
        
        # compute final accuracy on training and test sets
        y_pred = model.predict([tv[:, 0], tv[:, 1]])
        tr_acc = compute_accuracy(tv_y, y_pred)
        y_pred = model.predict([te[:, 0], te[:, 1]])
        te_acc = compute_accuracy(te_y, y_pred)
        
        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
        




        Start= int(self.FromDays_lineEdit_NN.text())
        End= int(self.ToDays_lineEdit_NN.text()) 
        End= End + 1
        

        
        for day in tqdm(range(Start, End)):
            foldername= self.FolderBrowse_lineEdit_NN2.text() + '/'
            dayname= str(day)
            foldername= foldername+str(day)
            foldername= foldername+str('/Segments/Images/')
            entries = Path(foldername)
            
            train_files = []
            y_train= []
            
            for i in range(1):
            
                syll = str(i)
                #entries = Path('D:/Songs/Multilesions/FINAL/Y611/244/Segments/Motif/Images/')
                script_dir = os.path.dirname(entries)
                output_dir = os.path.join(entries)#, 'Images/')
            
                folder = output_dir
            
                onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                for _file in onlyfiles:
                    train_files.append(_file)
                    clusters= i
                    y_train.append(clusters)
                    
            image_width = 28
            image_height = 28
            
            channels = 3
            lentrainfiles= len(train_files)
            
            datfilename= self.FolderBrowse_lineEdit_NN3.text() + '/'
            datfilenameday= datfilename+str(day)
            
            datfilenamedayr=os.path.join(datfilenameday+".mat")
            #filename = r'C:/Users/RobertsLab/Desktop/55.dat'
            dataset = np.memmap(datfilenamedayr, dtype='float32', mode='w+', shape=(lentrainfiles, image_height, image_width, channels))
            
            # example of horizontal shift image augmentation
           
            
            i=0
            v= 0    
            label= []
            
            for c in tqdm(range(1)):
                syll = str(c)
                #entries = Path('D:/Songs/Multilesions/FINAL/Y611/244/Segments/Motif/Images/')
            
                script_dir = os.path.dirname(entries)
                output_dir = os.path.join(entries)#, 'Images/')
            
                folder = output_dir
                onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                for _file in tqdm(onlyfiles):
            # load the image
                    img = load_img(folder + "/" + _file)  # this is a PIL image
                    img.thumbnail((image_width, image_height))
                    
                    # convert to numpy array
                    x = img_to_array(img)  
                    # expand dimension to one sample
                    samples = expand_dims(x, 0)
                    # create image data augmentation generator
                    datagen = ImageDataGenerator(width_shift_range=[-20,20])
                    # prepare iterator
                    it = datagen.flow(samples, batch_size=1)
                    # generate samples and plot
                    x = x / 255
                    dataset[i] = x
                    i += 1
                    a= y_train[v]
                    a= [a]
                    v+= 1
                    label.extend(a)
            
        
        
            entry= entries
            digit_indices = [np.where(y_train1 == i)[0] for i in range(num_classes)]
            if not os.path.isdir(entries):
                os.makedirs(entries)
            
            y_new1= np.asarray(label)
            digit_new = [np.where(y_new1 == i)[0] for i in range(1)]
            
            for zz in range(sfolders+1):
            
                def XpairsSyll1(x):
                    '''Positive and negative pair creation.
                    Alternates between positive and negative pairs.
                    '''
                    pairs = []
                    z1= digit_indices[zz][0]
                    for r in range(len(y_new1)):
                        z2= digit_new[0][r]
                        pairs += [[X_train[z1], x[z2]]]
                    return np.array(pairs)
                
                tx_pairs = XpairsSyll1(dataset)
                y_prob = model.predict([tx_pairs[:, 0], tx_pairs[:, 1]])
                percentile= np.percentile(y_prob, 99)
                pred = y_prob.ravel() < 0.1
                
                files= []
                for ii in range(len(y_prob)):
                    if pred[ii] == True:
                        file = onlyfiles[ii]
                        files.append(file)  
                        
                for iii in range(sfolders):
                    old = '{}'.format(iii)
                    path = os.path.join(entries, old)
                    if not os.path.exists(path):
                        os.mkdir(path)
                
                
                df = pd.DataFrame(files, columns= ['Filename'])
                
                #script_dir = os.path.join(entries, 'Images')
                foldarname= str(zz)
                output_dir = os.path.join(entries, foldarname)
                
                
                for _, row in df.iterrows():
                    
                    b=  row['Filename']
                    a = os.path.join(entry, b)
                    if os.path.exists(a):
                        shutil.move(a, output_dir)


    def PrepMotif(self):
                
        train_files = []
        y_train= []
        
        sfolders= int(self.folders_lineEdit_NNM.text())#"FromDAYS"
        
        
        for i in range(0,(sfolders + 1)):
        
            syll = str(i)
            entries = Path(self.FolderBrowse_lineEdit_NNM.text() + '/' + syll)
        
            script_dir = os.path.dirname(entries)
            output_dir = os.path.join(entries)#, 'Images/')
        
            folder = output_dir
        
            onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            for _file in onlyfiles:
                train_files.append(_file)
                clusters= i
                y_train.append(clusters)
                
        image_width = 28
        image_height = 28
        channels = 3
        flatdim= image_width*image_height*channels
        
        lentrainfiles= len(train_files)
        dataset = np.ndarray(shape=(lentrainfiles, image_height, image_width, channels),
                             dtype=np.float32)
        
        # example of horizontal shift image augmentation
        
        i=0
        v= 0    
        label= []
        
        for c in range(0,(sfolders+1)):
            syll = str(c)
            entries = Path(self.FolderBrowse_lineEdit_NNM.text() + '/' + syll)
        
            script_dir = os.path.dirname(entries)
            output_dir = os.path.join(entries)#, 'Images/')
        
            folder = output_dir
            onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            for _file in tqdm(onlyfiles):
        # load the image
                img = load_img(folder + "/" + _file)  # this is a PIL image
                img.thumbnail((image_width, image_height))
                
                # convert to numpy array
                x = img_to_array(img)  
                x = x / 255
                dataset[i] = x
                i += 1
                a= y_train[v]
                a= [a]
                v+= 1
                label.extend(a)
        
        
        #Splitting 
        X_train, X_test, y_train, y_test = train_test_split(dataset, y_train, test_size=0.3, random_state=31)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=30)
        



        
        
        num_classes = sfolders + 1
        classes= num_classes
        
        def euclidean_distance(vects):
            x, y = vects
            sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, K.epsilon()))
        
        
        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)
        
        
        def contrastive_loss(y_true, y_pred):
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            margin = 1
            sqaure_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
        
        def trip_loss(y_true, y_pred):
            import tensorflow as tf
            return tf.contrib.losses.metric_learning.triplet_semihard_loss(y_true, y_pred) 
        
        def create_pairs(x, digit_indices):
            list1=list(range(num_classes))
            list2= list(range(num_classes))
            
            difpairs= [[a, b] for a in list1
                      for b in list2 if a != b] 
            likpairs=[[a, b] for a in list1
                      for b in list2 if a == b]
            
            pairs=[]
            labels=[]
            for i in range(len(likpairs)):
                y=likpairs[i][0]
                n=likpairs[i][1]
                likepairs = [[a, b] for a in digit_indices[y]  
                          for b in digit_indices[n] if a != b] 
                for v in range(len(likepairs)):
                    tryy=likepairs[v]
                    pairs+=[[x[tryy[0]], x[tryy[1]]]]
                    labels +=[1] 
            for i in range(len(difpairs)):
                y=difpairs[i][0]
                n=difpairs[i][1]
                differentpairs = [[a, b] for a in digit_indices[y]  
                          for b in digit_indices[n] if a != b]
                for v in range(len(differentpairs)):
                    tryy=differentpairs[v]
                    pairs+=[[x[tryy[0]], x[tryy[1]]]]
                    labels +=[0] 
            
            return np.array(pairs), np.array(labels)
        
        
        def create_base_network(input_shape):
            '''Base network to be shared (eq. to feature extraction).
            '''
            inputShape = (28,28, 3)
            chanDim = -1
            
            # define the model input
            inputs = tensorflow.keras.layers.Input(shape=inputShape)
                    
                    # Convolution Layer 1
            x = Conv2D(32, (3, 3), padding="same", activation="elu")(inputs)
            x = BatchNormalization(axis=chanDim)(x)
                    
                    # Convolution Layer 2
            x = Conv2D(32, (3, 3), padding="same", strides=(2, 2),activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
            
            x = Conv2D(32, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 3
            x = Conv2D(32, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
                    
                    # Convolution Layer 4
            x = Conv2D(32, (3, 3), strides=(2, 2), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
                    
            x = Conv2D(32, (3, 3), strides=(1, 1), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 5
            x = Conv2D(32, (3, 3), strides=(2, 2), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
            
            x = Conv2D(32, (3, 3), strides=(1, 1), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 6
            x = Conv2D(64, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
            
            x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
        
            x = Dense(1024, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
            
            x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
        
                    # Convolution Layer 7
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
                   
            x = Conv2D(128, (3, 3), padding="same",strides=(1, 1), activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
                    # Convolution Layer 8
            x = Conv2D(128, (3, 3), strides=(2, 2), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
                  
                    # Convolution Layer 9
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
                    
                    # Convolution Layer 10
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
                    
                    # Convolution Layer 11
            x = Conv2D(128, (3, 3), padding="same",activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = Flatten()(x)
            x = Dropout(0.2)(x)
              
            		# first (and only) set of FC => RELU layers
            x = Dense(2048, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
            x = Dense(1024, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
        
            x = Dense(512, activation="elu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.2)(x)
               
            		# softmax classifier
            x = Dense(classes,activation="sigmoid",kernel_regularizer=regularizers.l2(0.01))(x)
              
            		# create the model
            model = Model(inputs, x)
            model.summary()
            return model
        
        def compute_accuracy(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.
            '''
            pred = y_pred.ravel() < 0.5
            return np.mean(pred == y_true)
        
        
        def accuracy(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.
            '''
            return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
        
        
        # the data, split between train and test sets
        input_shape = X_train.shape[1:]
        y_train1= np.asarray(y_train)
        y_test1= np.asarray(y_test)
        y_val1= np.asarray(y_val)
        
        from sklearn.utils.class_weight import compute_class_weight
        from keras.utils import np_utils  
        import os.path as path
        
        filename = self.FolderBrowse_lineEdit_NNM3.text() +'/'+'newfile1.dat'
        # create training+test positive and negative pairs
        digit_indices = [np.where(y_train1 == i)[0] for i in range(num_classes)]
        tr_pairs, tr_y = create_pairs(X_train, digit_indices)
        tr = np.memmap(filename, dtype='float32', mode='w+', shape=(len(tr_y),2,28,28,3))
        for i in range(len(tr_y)):
            tr[i]=tr_pairs[i]
        del tr_pairs
        
        filename1 = self.FolderBrowse_lineEdit_NNM3.text() +'/'+'newfile2.dat'
        digit_indices = [np.where(y_test1 == i)[0] for i in range(num_classes)]
        te_pairs, te_y = create_pairs(X_test, digit_indices)
        te = np.memmap(filename1, dtype='float32', mode='w+', shape=(len(te_y),2,28,28,3))
        for i in range(len(te_y)):
            te[i]=te_pairs[i]
        del te_pairs
        
        filename2 = self.FolderBrowse_lineEdit_NNM3.text() +'/'+'newfile3.dat'
        digit_indices = [np.where(y_val1 == i)[0] for i in range(num_classes)]
        tv_pairs, tv_y = create_pairs(X_val, digit_indices)
        tv = np.memmap(filename2, dtype='float32', mode='w+', shape=(len(tv_y),2,28,28,3))
        for i in range(len(tv_y)):
            tv[i]=tv_pairs[i]
        del tv_pairs 
        
        # network definition
        #needs tensoflow1.14
        base_network = create_base_network(input_shape)
        
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        print(input_a)
        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        
        model = Model([input_a, input_b], distance)
        
        y= y_test + y_train
        nb_classes= num_classes
        ycat= np.float16(np_utils.to_categorical(y, nb_classes))
        y_flat = np.argmax(ycat, axis=1)
        
        class_weights = compute_class_weight(class_weight= 'balanced', classes= np.unique(y_flat),
                                            y= y_flat)
        
        
        class_weights = dict(zip(np.unique(y_flat), class_weights))
        
        es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
        
        import tensorflow as tf
        
        te_y= np.float32(te_y)
        tr_y= np.float32(tr_y)
        tv_y=np.float32(tv_y)
        te_y=tf.cast(te_y, tf.float32)
        
        
        
        # train
        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
        H = model.fit([tr[:, 0], tr[:, 1]], tr_y, shuffle=True, class_weight= class_weights,
                  batch_size=512,
                  epochs=20, validation_data=([tv[:, 0], tv[:, 1]], tv_y), callbacks= [es])
         
        
        # compute final accuracy on training and test sets
        y_pred = model.predict([tv[:, 0], tv[:, 1]])
        tr_acc = compute_accuracy(tv_y, y_pred)
        y_pred = model.predict([te[:, 0], te[:, 1]])
        te_acc = compute_accuracy(te_y, y_pred)
        
        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
        




        Start= int(self.FromDays_lineEdit_NNM.text())
        End= int(self.ToDays_lineEdit_NNM.text()) 
        End= End + 1
        

        
        for day in tqdm(range(Start, End)):
            foldername= self.FolderBrowse_lineEdit_NNM2.text() + '/'
            dayname= str(day)
            foldername= foldername+str(day)
            foldername= foldername+str('/Segments/Motif/Images/')
            entries = Path(foldername)
            
            train_files = []
            y_train= []
            
            for i in range(1):
            
                syll = str(i)
                #entries = Path('D:/Songs/Multilesions/FINAL/Y611/244/Segments/Motif/Images/')
                script_dir = os.path.dirname(entries)
                output_dir = os.path.join(entries)#, 'Images/')
            
                folder = output_dir
            
                onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                for _file in onlyfiles:
                    train_files.append(_file)
                    clusters= i
                    y_train.append(clusters)
                    
            image_width = 28
            image_height = 28
            
            channels = 3
            lentrainfiles= len(train_files)
            
            datfilename= self.FolderBrowse_lineEdit_NNM3.text() + '/'
            datfilenameday= datfilename+str(day)
            
            datfilenamedayr=os.path.join(datfilenameday+".mat")
            #filename = r'C:/Users/RobertsLab/Desktop/55.dat'
            dataset = np.memmap(datfilenamedayr, dtype='float32', mode='w+', shape=(lentrainfiles, image_height, image_width, channels))
            
            # example of horizontal shift image augmentation
           
            
            i=0
            v= 0    
            label= []
            
            for c in tqdm(range(1)):
                syll = str(c)
                #entries = Path('D:/Songs/Multilesions/FINAL/Y611/244/Segments/Motif/Images/')
            
                script_dir = os.path.dirname(entries)
                output_dir = os.path.join(entries)#, 'Images/')
            
                folder = output_dir
                onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                for _file in tqdm(onlyfiles):
            # load the image
                    img = load_img(folder + "/" + _file)  # this is a PIL image
                    img.thumbnail((image_width, image_height))
                    
                    # convert to numpy array
                    x = img_to_array(img)  
                    # expand dimension to one sample
                    samples = expand_dims(x, 0)
                    # create image data augmentation generator
                    datagen = ImageDataGenerator(width_shift_range=[-20,20])
                    # prepare iterator
                    it = datagen.flow(samples, batch_size=1)
                    # generate samples and plot
                    x = x / 255
                    dataset[i] = x
                    i += 1
                    a= y_train[v]
                    a= [a]
                    v+= 1
                    label.extend(a)
            
        
        
            entry= entries
            digit_indices = [np.where(y_train1 == i)[0] for i in range(num_classes)]
            if not os.path.isdir(entries):
                os.makedirs(entries)
            
            y_new1= np.asarray(label)
            digit_new = [np.where(y_new1 == i)[0] for i in range(1)]
            
            for zz in range(sfolders+1):
            
                def XpairsSyll1(x):
                    '''Positive and negative pair creation.
                    Alternates between positive and negative pairs.
                    '''
                    pairs = []
                    z1= digit_indices[zz][0]
                    for r in range(len(y_new1)):
                        z2= digit_new[0][r]
                        pairs += [[X_train[z1], x[z2]]]
                    return np.array(pairs)
                
                tx_pairs = XpairsSyll1(dataset)
                y_prob = model.predict([tx_pairs[:, 0], tx_pairs[:, 1]])
                percentile= np.percentile(y_prob, 99)
                pred = y_prob.ravel() < 0.1
                
                files= []
                for ii in range(len(y_prob)):
                    if pred[ii] == True:
                        file = onlyfiles[ii]
                        files.append(file)  
                        
                for iii in range(sfolders):
                    old = '{}'.format(iii)
                    path = os.path.join(entries, old)
                    if not os.path.exists(path):
                        os.mkdir(path)
                
                
                df = pd.DataFrame(files, columns= ['Filename'])
                
                #script_dir = os.path.join(entries, 'Images')
                foldarname= str(zz)
                output_dir = os.path.join(entries, foldarname)
                
                
                for _, row in df.iterrows():
                    
                    b=  row['Filename']
                    a = os.path.join(entry, b)
                    if os.path.exists(a):
                        shutil.move(a, output_dir)
                        
            
    
    
    
    

                
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(40, 30, 721, 511))
        self.tabWidget.setObjectName("tabWidget")
        self.Segment_tab = QtWidgets.QWidget()
        self.Segment_tab.setObjectName("Segment_tab")
        
        
        
        self.BirdPath_label = QtWidgets.QLabel(self.Segment_tab)
        self.BirdPath_label.setGeometry(QtCore.QRect(20, 30, 111, 41))
        self.BirdPath_label.setObjectName("BirdPath_label")
        
        
        self.FolderBrowse_lineEdit = QtWidgets.QLineEdit(self.Segment_tab)
        self.FolderBrowse_lineEdit.setGeometry(QtCore.QRect(20, 70, 211, 31))
        self.FolderBrowse_lineEdit.setObjectName("FolderBrowse_lineEdit")
        
        
        self.FolderBrowse_pushButton = QtWidgets.QPushButton(self.Segment_tab)
        self.FolderBrowse_pushButton.setGeometry(QtCore.QRect(270, 70, 75, 23))
        self.FolderBrowse_pushButton.setObjectName("FolderBrowse_pushButton")
        self.FolderBrowse_pushButton.clicked.connect(self._open_file_dialog)
        
        self.FromDays_lineEdit = QtWidgets.QLineEdit(self.Segment_tab)
        self.FromDays_lineEdit.setGeometry(QtCore.QRect(30, 160, 61, 31))
        self.FromDays_lineEdit.setObjectName("FromDays_lineEdit")
        
        
        self.ToDays_lineEdit = QtWidgets.QLineEdit(self.Segment_tab)
        self.ToDays_lineEdit.setGeometry(QtCore.QRect(160, 160, 61, 31))
        self.ToDays_lineEdit.setObjectName("ToDays_lineEdit")
        
        
        self.From_label = QtWidgets.QLabel(self.Segment_tab)
        self.From_label.setGeometry(QtCore.QRect(40, 140, 47, 13))
        self.From_label.setObjectName("From_label")
        
        
        self.To_label = QtWidgets.QLabel(self.Segment_tab)
        self.To_label.setGeometry(QtCore.QRect(170, 140, 47, 13))
        self.To_label.setObjectName("To_label")
        
        
        self.Segment_pushButton = QtWidgets.QPushButton(self.Segment_tab)
        self.Segment_pushButton.setGeometry(QtCore.QRect(60, 270, 75, 23))
        self.Segment_pushButton.setObjectName("Segment_pushButton")
        self.Segment_pushButton.clicked.connect(self.Segment)
        
        self.CreateGraphView_pushButton = QtWidgets.QPushButton(self.Segment_tab)
        self.CreateGraphView_pushButton.setGeometry(QtCore.QRect(430, 70, 101, 23))
        self.CreateGraphView_pushButton.setObjectName("CreateGraph_pushButton")
        self.CreateGraphView_pushButton.clicked.connect(self.ViewMotifSyll)
        
        self.LowerThreshold = QtWidgets.QLineEdit(self.Segment_tab)
        self.LowerThreshold.setGeometry(QtCore.QRect(430, 170, 61, 31))
        self.LowerThreshold.setObjectName("LowerThreshold")
        
        
        
        self.tabWidget.addTab(self.Segment_tab, "")
        self.CreateImages_tab = QtWidgets.QWidget()
        self.CreateImages_tab.setObjectName("CreateImages_tab")
        
        
        self.To_label_CI = QtWidgets.QLabel(self.CreateImages_tab)
        self.To_label_CI.setGeometry(QtCore.QRect(170, 140, 47, 13))
        self.To_label_CI.setObjectName("To_label_CI")
        
        
        self.BirdPath_label_CI = QtWidgets.QLabel(self.CreateImages_tab)
        self.BirdPath_label_CI.setGeometry(QtCore.QRect(20, 30, 111, 41))
        self.BirdPath_label_CI.setObjectName("BirdPath_label_CI")
        
        
        self.FolderBrowse_lineEdit_CI = QtWidgets.QLineEdit(self.CreateImages_tab)
        self.FolderBrowse_lineEdit_CI.setGeometry(QtCore.QRect(20, 70, 211, 31))
        self.FolderBrowse_lineEdit_CI.setObjectName("FolderBrowse_lineEdit_CI")
        
        
        self.From_label_CI = QtWidgets.QLabel(self.CreateImages_tab)
        self.From_label_CI.setGeometry(QtCore.QRect(40, 140, 47, 13))
        self.From_label_CI.setObjectName("From_label_CI")
        
        
        self.FolderBrowse_pushButton_CI = QtWidgets.QPushButton(self.CreateImages_tab)
        self.FolderBrowse_pushButton_CI.setGeometry(QtCore.QRect(270, 70, 75, 23))
        self.FolderBrowse_pushButton_CI.setObjectName("FolderBrowse_pushButton_CI")
        self.FolderBrowse_pushButton_CI.clicked.connect(self._open_file_dialog_CI)
        
        
        self.FromDays_lineEdit_CI = QtWidgets.QLineEdit(self.CreateImages_tab)
        self.FromDays_lineEdit_CI.setGeometry(QtCore.QRect(30, 160, 61, 31))
        self.FromDays_lineEdit_CI.setObjectName("FromDays_lineEdit_CI")
        
        
        self.ToDays_lineEdit_CI = QtWidgets.QLineEdit(self.CreateImages_tab)
        self.ToDays_lineEdit_CI.setGeometry(QtCore.QRect(160, 160, 61, 31))
        self.ToDays_lineEdit_CI.setObjectName("ToDays_lineEdit_CI")
        
        
        self.CreateImages_pushButton = QtWidgets.QPushButton(self.CreateImages_tab)
        self.CreateImages_pushButton.setGeometry(QtCore.QRect(60, 270, 101, 23))
        self.CreateImages_pushButton.setObjectName("CreateImages_pushButton")
        self.CreateImages_pushButton.clicked.connect(self.CreateImages)
        
        
        self.tabWidget.addTab(self.CreateImages_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        
        self.tabWidget.addTab(self.CreateImages_tab, "")
        self.SegmentMotif_tab = QtWidgets.QWidget()
        self.SegmentMotif_tab.setObjectName("SegmentMotif_tab")
        
        self.UpperBound_Label_SM = QtWidgets.QLabel(self.SegmentMotif_tab)
        self.UpperBound_Label_SM.setGeometry(QtCore.QRect(20, 30, 111, 41))
        self.UpperBound_Label_SM.setObjectName("UpperBound_Label_SM")
        
        self.UpperBound_lineEdit_SM = QtWidgets.QLineEdit(self.SegmentMotif_tab)
        self.UpperBound_lineEdit_SM.setGeometry(QtCore.QRect(30, 70, 61, 31))
        self.UpperBound_lineEdit_SM.setObjectName("UpperBound_lineEdit_SM")
        
        self.ViewPeak_pushButton_SM = QtWidgets.QPushButton(self.SegmentMotif_tab)
        self.ViewPeak_pushButton_SM.setGeometry(QtCore.QRect(160, 70, 61, 31))
        self.ViewPeak_pushButton_SM.setObjectName("ViewPeak_pushButton_SM")
        self.ViewPeak_pushButton_SM.clicked.connect(self.ViewPeaks)
        
        self.Beginning_SongFile = QtWidgets.QLabel(self.SegmentMotif_tab)
        self.Beginning_SongFile.setGeometry(QtCore.QRect(40, 140, 47, 13))
        self.Beginning_SongFile.setObjectName("Beginning_SongFile")
        
        
        self.End_SongFile = QtWidgets.QLabel(self.SegmentMotif_tab)
        self.End_SongFile.setGeometry(QtCore.QRect(170, 140, 47, 13))
        self.End_SongFile.setObjectName("End_SongFile")
        
        self.Beginning_SongFile_lineEdit_SM = QtWidgets.QLineEdit(self.SegmentMotif_tab)
        self.Beginning_SongFile_lineEdit_SM.setGeometry(QtCore.QRect(30, 160, 61, 31))
        self.Beginning_SongFile_lineEdit_SM.setObjectName("Beginning_SongFile_lineEdit_SM")
        
        
        self.End_SongFile_lineEdit_SM = QtWidgets.QLineEdit(self.SegmentMotif_tab)
        self.End_SongFile_lineEdit_SM.setGeometry(QtCore.QRect(160, 160, 61, 31))
        self.End_SongFile_lineEdit_SM.setObjectName("End_SongFile_lineEdit_SM")
        
        
        self.CreateGraph_pushButton = QtWidgets.QPushButton(self.SegmentMotif_tab)
        self.CreateGraph_pushButton.setGeometry(QtCore.QRect(30, 230, 101, 23))
        self.CreateGraph_pushButton.setObjectName("CreateGraph_pushButton")
        self.CreateGraph_pushButton.clicked.connect(self.ViewMotif)
        
        
        
        
        self.To_label_SM = QtWidgets.QLabel(self.SegmentMotif_tab)
        self.To_label_SM.setGeometry(QtCore.QRect(470, 140, 47, 13))
        self.To_label_SM.setObjectName("To_label_SM")
        
        
        self.BirdPath_label_SM = QtWidgets.QLabel(self.SegmentMotif_tab)
        self.BirdPath_label_SM.setGeometry(QtCore.QRect(320, 30, 111, 41))
        self.BirdPath_label_SM.setObjectName("BirdPath_label_SM")
        
        
        self.FolderBrowse_lineEdit_SM = QtWidgets.QLineEdit(self.SegmentMotif_tab)
        self.FolderBrowse_lineEdit_SM.setGeometry(QtCore.QRect(320, 70, 211, 31))
        self.FolderBrowse_lineEdit_SM.setObjectName("FolderBrowse_lineEdit_SM")
        
        
        self.From_label_SM = QtWidgets.QLabel(self.SegmentMotif_tab)
        self.From_label_SM.setGeometry(QtCore.QRect(340, 140, 47, 13))
        self.From_label_SM.setObjectName("From_label_SM")
        
        
        self.FolderBrowse_pushButton_SM = QtWidgets.QPushButton(self.SegmentMotif_tab)
        self.FolderBrowse_pushButton_SM.setGeometry(QtCore.QRect(570, 70, 75, 23))
        self.FolderBrowse_pushButton_SM.setObjectName("FolderBrowse_pushButton_SM")
        self.FolderBrowse_pushButton_SM.clicked.connect(self._open_file_dialog_SM)
        
        
        self.FromDays_lineEdit_SM = QtWidgets.QLineEdit(self.SegmentMotif_tab)
        self.FromDays_lineEdit_SM.setGeometry(QtCore.QRect(330, 160, 61, 31))
        self.FromDays_lineEdit_SM.setObjectName("FromDays_lineEdit_SM")
        
        
        self.ToDays_lineEdit_SM = QtWidgets.QLineEdit(self.SegmentMotif_tab)
        self.ToDays_lineEdit_SM.setGeometry(QtCore.QRect(460, 160, 61, 31))
        self.ToDays_lineEdit_SM.setObjectName("ToDays_lineEdit_SM")
        
        
        self.SegmentMotif_pushButton = QtWidgets.QPushButton(self.SegmentMotif_tab)
        self.SegmentMotif_pushButton.setGeometry(QtCore.QRect(360, 270, 101, 23))
        self.SegmentMotif_pushButton.setObjectName("CreateImages_pushButton")
        self.SegmentMotif_pushButton.clicked.connect(self.MotifSegment)
        
        
        self.tabWidget.addTab(self.SegmentMotif_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        

        self.tabWidget.addTab(self.SegmentMotif_tab, "")
        self.CreateMotifImages_tab = QtWidgets.QWidget()
        self.CreateMotifImages_tab.setObjectName("CreateMotifImages_tab")
        
        
        self.To_label_CMI = QtWidgets.QLabel(self.CreateMotifImages_tab)
        self.To_label_CMI.setGeometry(QtCore.QRect(170, 140, 47, 13))
        self.To_label_CMI.setObjectName("To_label_CMI")
        
        
        self.BirdPath_label_CMI = QtWidgets.QLabel(self.CreateMotifImages_tab)
        self.BirdPath_label_CMI.setGeometry(QtCore.QRect(20, 30, 111, 41))
        self.BirdPath_label_CMI.setObjectName("BirdPath_label_CMI")
        
        
        self.FolderBrowse_lineEdit_CMI = QtWidgets.QLineEdit(self.CreateMotifImages_tab)
        self.FolderBrowse_lineEdit_CMI.setGeometry(QtCore.QRect(20, 70, 211, 31))
        self.FolderBrowse_lineEdit_CMI.setObjectName("FolderBrowse_lineEdit_CMI")
        
        
        self.From_label_CMI = QtWidgets.QLabel(self.CreateMotifImages_tab)
        self.From_label_CMI.setGeometry(QtCore.QRect(40, 140, 47, 13))
        self.From_label_CMI.setObjectName("From_label_CMI")
        
        
        self.FolderBrowse_pushButton_CMI = QtWidgets.QPushButton(self.CreateMotifImages_tab)
        self.FolderBrowse_pushButton_CMI.setGeometry(QtCore.QRect(270, 70, 75, 23))
        self.FolderBrowse_pushButton_CMI.setObjectName("FolderBrowse_pushButton_CMI")
        self.FolderBrowse_pushButton_CMI.clicked.connect(self._open_file_dialog_CMI)
        
        
        self.FromDays_lineEdit_CMI = QtWidgets.QLineEdit(self.CreateMotifImages_tab)
        self.FromDays_lineEdit_CMI.setGeometry(QtCore.QRect(30, 160, 61, 31))
        self.FromDays_lineEdit_CMI.setObjectName("FromDays_lineEdit_CMI")
        
        
        self.ToDays_lineEdit_CMI = QtWidgets.QLineEdit(self.CreateMotifImages_tab)
        self.ToDays_lineEdit_CMI.setGeometry(QtCore.QRect(160, 160, 61, 31))
        self.ToDays_lineEdit_CMI.setObjectName("ToDays_lineEdit_CMI")
        
        
        self.CreateMotifImages_pushButton = QtWidgets.QPushButton(self.CreateMotifImages_tab)
        self.CreateMotifImages_pushButton.setGeometry(QtCore.QRect(60, 270, 131, 23))
        self.CreateMotifImages_pushButton.setObjectName("CreateMotifImages_pushButton")
        self.CreateMotifImages_pushButton.clicked.connect(self.CreateMotifImages)
        
        
        self.tabWidget.addTab(self.CreateMotifImages_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        
        
        self.tabWidget.addTab(self.CreateMotifImages_tab, "")
        self.SyllableCluster_tab = QtWidgets.QWidget()
        self.SyllableCluster_tab.setObjectName("SyllableCluster_tab")

        self.FolderBrowse_lineEdit_SC = QtWidgets.QLineEdit(self.SyllableCluster_tab)
        self.FolderBrowse_lineEdit_SC.setGeometry(QtCore.QRect(20, 70, 211, 31))
        self.FolderBrowse_lineEdit_SC.setObjectName("FolderBrowse_lineEdit_SC")


        self.BirdPath_label_SC = QtWidgets.QLabel(self.SyllableCluster_tab)
        self.BirdPath_label_SC.setGeometry(QtCore.QRect(20, 30, 111, 41))
        self.BirdPath_label_SC.setObjectName("BirdPath_label_SC")

        self.FolderBrowse_pushButton_SC = QtWidgets.QPushButton(self.SyllableCluster_tab)
        self.FolderBrowse_pushButton_SC.setGeometry(QtCore.QRect(270, 70, 75, 23))
        self.FolderBrowse_pushButton_SC.setObjectName("FolderBrowse_pushButton_SC")
        self.FolderBrowse_pushButton_SC.clicked.connect(self._open_file_dialog_SC)
        
        self.Bird_Letter_SC = QtWidgets.QLabel(self.SyllableCluster_tab)
        self.Bird_Letter_SC.setGeometry(QtCore.QRect(40, 140, 100, 13))
        self.Bird_Letter_SC.setObjectName("Bird_Letter_SC")
        
        self.BirdLetter_lineEdit_SC = QtWidgets.QLineEdit(self.SyllableCluster_tab)
        self.BirdLetter_lineEdit_SC.setGeometry(QtCore.QRect(50, 160, 61, 31))
        self.BirdLetter_lineEdit_SC.setObjectName("BirdLetter_lineEdit_SC")
        
        self.SyllableCluster_pushButton = QtWidgets.QPushButton(self.SyllableCluster_tab)
        self.SyllableCluster_pushButton.setGeometry(QtCore.QRect(60, 220, 131, 23))
        self.SyllableCluster_pushButton.setObjectName("SyllableCluster_pushButton")
        self.SyllableCluster_pushButton.clicked.connect(self.SyllableCluster)


        self.tabWidget.addTab(self.SyllableCluster_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        
        
        self.tabWidget.addTab(self.SyllableCluster_tab, "")
        self.MotifCluster_tab = QtWidgets.QWidget()
        self.MotifCluster_tab.setObjectName("MotifCluster_tab")

        self.FolderBrowse_lineEdit_MC = QtWidgets.QLineEdit(self.MotifCluster_tab)
        self.FolderBrowse_lineEdit_MC.setGeometry(QtCore.QRect(20, 70, 211, 31))
        self.FolderBrowse_lineEdit_MC.setObjectName("FolderBrowse_lineEdit_MC")


        self.BirdPath_label_MC = QtWidgets.QLabel(self.MotifCluster_tab)
        self.BirdPath_label_MC.setGeometry(QtCore.QRect(20, 30, 111, 41))
        self.BirdPath_label_MC.setObjectName("BirdPath_label_MC")

        self.FolderBrowse_pushButton_MC = QtWidgets.QPushButton(self.MotifCluster_tab)
        self.FolderBrowse_pushButton_MC.setGeometry(QtCore.QRect(270, 70, 75, 23))
        self.FolderBrowse_pushButton_MC.setObjectName("FolderBrowse_pushButton_MC")
        self.FolderBrowse_pushButton_MC.clicked.connect(self._open_file_dialog_MC)
        
        self.Bird_Letter_MC = QtWidgets.QLabel(self.MotifCluster_tab)
        self.Bird_Letter_MC.setGeometry(QtCore.QRect(40, 140, 100, 13))
        self.Bird_Letter_MC.setObjectName("Bird_Letter_MC")
        
        self.BirdLetter_lineEdit_MC = QtWidgets.QLineEdit(self.MotifCluster_tab)
        self.BirdLetter_lineEdit_MC.setGeometry(QtCore.QRect(50, 160, 61, 31))
        self.BirdLetter_lineEdit_MC.setObjectName("BirdLetter_lineEdit_MC")
        
        self.MotifCluster_pushButton = QtWidgets.QPushButton(self.MotifCluster_tab)
        self.MotifCluster_pushButton.setGeometry(QtCore.QRect(60, 220, 131, 23))
        self.MotifCluster_pushButton.setObjectName("MotifCluster_pushButton")
        self.MotifCluster_pushButton.clicked.connect(self.MotifCluster)


        self.tabWidget.addTab(self.MotifCluster_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)



        self.tabWidget.addTab(self.MotifCluster_tab, "")
        self.SyllNN_tab = QtWidgets.QWidget()
        self.SyllNN_tab.setObjectName("SyllNN_tab")

        self.FolderBrowse_lineEdit_NN = QtWidgets.QLineEdit(self.SyllNN_tab)
        self.FolderBrowse_lineEdit_NN.setGeometry(QtCore.QRect(20, 70, 211, 31))
        self.FolderBrowse_lineEdit_NN.setObjectName("FolderBrowse_lineEdit_NN")

        self.ClusterPath_label_NN = QtWidgets.QLabel(self.SyllNN_tab)
        self.ClusterPath_label_NN.setGeometry(QtCore.QRect(20, 40, 211, 31))
        self.ClusterPath_label_NN.setObjectName("ClusterPath_label_NN")
        
        self.FolderBrowse_lineEdit_NN2 = QtWidgets.QLineEdit(self.SyllNN_tab)
        self.FolderBrowse_lineEdit_NN2.setGeometry(QtCore.QRect(20, 140, 211, 31))
        self.FolderBrowse_lineEdit_NN2.setObjectName("FolderBrowse_lineEdit_NN2")

        self.Root_label_NN = QtWidgets.QLabel(self.SyllNN_tab)
        self.Root_label_NN.setGeometry(QtCore.QRect(20, 110, 211, 31))
        self.Root_label_NN.setObjectName("Root_label_NN")
        
        self.FolderBrowse_lineEdit_NN3 = QtWidgets.QLineEdit(self.SyllNN_tab)
        self.FolderBrowse_lineEdit_NN3.setGeometry(QtCore.QRect(20, 210, 211, 31))
        self.FolderBrowse_lineEdit_NN3.setObjectName("FolderBrowse_lineEdit_NN3")        

        self.Temp_label_NN = QtWidgets.QLabel(self.SyllNN_tab)
        self.Temp_label_NN.setGeometry(QtCore.QRect(20, 180, 211, 31))
        self.Temp_label_NN.setObjectName("Temp_label_NN")
        
        self.BirdPath_label_NN = QtWidgets.QLabel(self.SyllNN_tab)
        self.BirdPath_label_NN.setGeometry(QtCore.QRect(420, 50, 111, 41))
        self.BirdPath_label_NN.setObjectName("BirdPath_label_NN")
        
        self.To_label_NN = QtWidgets.QLabel(self.SyllNN_tab)
        self.To_label_NN.setGeometry(QtCore.QRect(560, 140, 47, 13))
        self.To_label_NN.setObjectName("To_label_NN")
        
        
        self.FromDays_lineEdit_NN = QtWidgets.QLineEdit(self.SyllNN_tab)
        self.FromDays_lineEdit_NN.setGeometry(QtCore.QRect(430, 160, 61, 31))
        self.FromDays_lineEdit_NN.setObjectName("FromDays_lineEdit_NN")

        self.From_label_NN = QtWidgets.QLabel(self.SyllNN_tab)
        self.From_label_NN.setGeometry(QtCore.QRect(430, 140, 47, 13))
        self.From_label_NN.setObjectName("From_label_NN")
        
        
        self.ToDays_lineEdit_NN = QtWidgets.QLineEdit(self.SyllNN_tab)
        self.ToDays_lineEdit_NN.setGeometry(QtCore.QRect(560, 160, 61, 31))
        self.ToDays_lineEdit_NN.setObjectName("ToDays_lineEdit_NN")
        
        self.folders_lineEdit_NN = QtWidgets.QLineEdit(self.SyllNN_tab)
        self.folders_lineEdit_NN.setGeometry(QtCore.QRect(500, 60, 61, 31))
        self.folders_lineEdit_NN.setObjectName("ToDays_lineEdit_NN")
        
        self.FolderBrowse_pushButton_NN = QtWidgets.QPushButton(self.SyllNN_tab)
        self.FolderBrowse_pushButton_NN.setGeometry(QtCore.QRect(270, 70, 75, 23))
        self.FolderBrowse_pushButton_NN.setObjectName("FolderBrowse_pushButton_NN")
        self.FolderBrowse_pushButton_NN.clicked.connect(self._open_file_dialog_NN)
        
        self.FolderBrowse_pushButton_NN2 = QtWidgets.QPushButton(self.SyllNN_tab)
        self.FolderBrowse_pushButton_NN2.setGeometry(QtCore.QRect(270, 140, 75, 23))
        self.FolderBrowse_pushButton_NN2.setObjectName("FolderBrowse_pushButton_NN2")
        self.FolderBrowse_pushButton_NN2.clicked.connect(self._open_file_dialog_NN2)
        
        self.FolderBrowse_pushButton_NN3 = QtWidgets.QPushButton(self.SyllNN_tab)
        self.FolderBrowse_pushButton_NN3.setGeometry(QtCore.QRect(270, 210, 75, 23))
        self.FolderBrowse_pushButton_NN3.setObjectName("FolderBrowse_pushButton_NN3")
        self.FolderBrowse_pushButton_NN3.clicked.connect(self._open_file_dialog_NN3)
        
        self.SyllNN_pushButton = QtWidgets.QPushButton(self.SyllNN_tab)
        self.SyllNN_pushButton.setGeometry(QtCore.QRect(60, 250, 131, 23))
        self.SyllNN_pushButton.setObjectName("SyllNN_pushButton")
        self.SyllNN_pushButton.clicked.connect(self.PrepSyll)
        
        self.tabWidget.addTab(self.SyllNN_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        
        self.tabWidget.addTab(self.SyllNN_tab, "")
        self.MotifNN_tab = QtWidgets.QWidget()
        self.MotifNN_tab.setObjectName("MotifNN_tab")

        self.FolderBrowse_lineEdit_NNM = QtWidgets.QLineEdit(self.MotifNN_tab)
        self.FolderBrowse_lineEdit_NNM.setGeometry(QtCore.QRect(20, 70, 211, 31))
        self.FolderBrowse_lineEdit_NNM.setObjectName("FolderBrowse_lineEdit_NNM")

        self.ClusterPath_label_NNM = QtWidgets.QLabel(self.MotifNN_tab)
        self.ClusterPath_label_NNM.setGeometry(QtCore.QRect(20, 40, 211, 31))
        self.ClusterPath_label_NNM.setObjectName("ClusterPath_label_NNM")
        
        self.FolderBrowse_lineEdit_NNM2 = QtWidgets.QLineEdit(self.MotifNN_tab)
        self.FolderBrowse_lineEdit_NNM2.setGeometry(QtCore.QRect(20, 140, 211, 31))
        self.FolderBrowse_lineEdit_NNM2.setObjectName("FolderBrowse_lineEdit_NNM2")

        self.Root_label_NNM = QtWidgets.QLabel(self.MotifNN_tab)
        self.Root_label_NNM.setGeometry(QtCore.QRect(20, 110, 211, 31))
        self.Root_label_NNM.setObjectName("Root_label_NNM")
        
        self.FolderBrowse_lineEdit_NNM3 = QtWidgets.QLineEdit(self.MotifNN_tab)
        self.FolderBrowse_lineEdit_NNM3.setGeometry(QtCore.QRect(20, 210, 211, 31))
        self.FolderBrowse_lineEdit_NNM3.setObjectName("FolderBrowse_lineEdit_NNM3")        

        self.Temp_label_NNM = QtWidgets.QLabel(self.MotifNN_tab)
        self.Temp_label_NNM.setGeometry(QtCore.QRect(20, 180, 211, 31))
        self.Temp_label_NNM.setObjectName("Temp_label_NNM")
        
        self.BirdPath_label_NNM = QtWidgets.QLabel(self.MotifNN_tab)
        self.BirdPath_label_NNM.setGeometry(QtCore.QRect(420, 50, 111, 41))
        self.BirdPath_label_NNM.setObjectName("BirdPath_label_NNM")
        
        self.To_label_NNM = QtWidgets.QLabel(self.MotifNN_tab)
        self.To_label_NNM.setGeometry(QtCore.QRect(560, 140, 47, 13))
        self.To_label_NNM.setObjectName("To_label_NNM")
        
        
        self.FromDays_lineEdit_NNM = QtWidgets.QLineEdit(self.MotifNN_tab)
        self.FromDays_lineEdit_NNM.setGeometry(QtCore.QRect(430, 160, 61, 31))
        self.FromDays_lineEdit_NNM.setObjectName("FromDays_lineEdit_NNM")

        self.From_label_NNM = QtWidgets.QLabel(self.MotifNN_tab)
        self.From_label_NNM.setGeometry(QtCore.QRect(430, 140, 47, 13))
        self.From_label_NNM.setObjectName("From_label_NNM")
        
        
        self.ToDays_lineEdit_NNM = QtWidgets.QLineEdit(self.MotifNN_tab)
        self.ToDays_lineEdit_NNM.setGeometry(QtCore.QRect(560, 160, 61, 31))
        self.ToDays_lineEdit_NNM.setObjectName("ToDays_lineEdit_NNM")
        
        self.folders_lineEdit_NNM = QtWidgets.QLineEdit(self.MotifNN_tab)
        self.folders_lineEdit_NNM.setGeometry(QtCore.QRect(500, 60, 61, 31))
        self.folders_lineEdit_NNM.setObjectName("ToDays_lineEdit_NNM")
        
        self.FolderBrowse_pushButton_NNM = QtWidgets.QPushButton(self.MotifNN_tab)
        self.FolderBrowse_pushButton_NNM.setGeometry(QtCore.QRect(270, 70, 75, 23))
        self.FolderBrowse_pushButton_NNM.setObjectName("FolderBrowse_pushButton_NNM")
        self.FolderBrowse_pushButton_NNM.clicked.connect(self._open_file_dialog_NNM)
        
        self.FolderBrowse_pushButton_NNM2 = QtWidgets.QPushButton(self.MotifNN_tab)
        self.FolderBrowse_pushButton_NNM2.setGeometry(QtCore.QRect(270, 140, 75, 23))
        self.FolderBrowse_pushButton_NNM2.setObjectName("FolderBrowse_pushButton_NNM2")
        self.FolderBrowse_pushButton_NNM2.clicked.connect(self._open_file_dialog_NNM2)
        
        self.FolderBrowse_pushButton_NNM3 = QtWidgets.QPushButton(self.MotifNN_tab)
        self.FolderBrowse_pushButton_NNM3.setGeometry(QtCore.QRect(270, 210, 75, 23))
        self.FolderBrowse_pushButton_NNM3.setObjectName("FolderBrowse_pushButton_NNM3")
        self.FolderBrowse_pushButton_NNM3.clicked.connect(self._open_file_dialog_NNM3)
        
        self.MotifNN_pushButton = QtWidgets.QPushButton(self.MotifNN_tab)
        self.MotifNN_pushButton.setGeometry(QtCore.QRect(60, 250, 131, 23))
        self.MotifNN_pushButton.setObjectName("MotifNN_pushButton")
        self.MotifNN_pushButton.clicked.connect(self.PrepMotif)
        
        self.tabWidget.addTab(self.MotifNN_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BirdPath_label.setText(_translate("MainWindow", "Folder Path"))
        self.FolderBrowse_pushButton.setText(_translate("MainWindow", "Browse"))
        self.From_label.setText(_translate("MainWindow", "From Day"))
        self.To_label.setText(_translate("MainWindow", "To Day"))
        self.Segment_pushButton.setText(_translate("MainWindow", "Segment"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Segment_tab), _translate("MainWindow", "Segment"))
        self.CreateGraphView_pushButton.setText(_translate("MainWindow", "View"))
        self.LowerThreshold.setPlaceholderText("10000")

        self.To_label_CI.setText(_translate("MainWindow", "To Day"))
        self.BirdPath_label_CI.setText(_translate("MainWindow", "Folder Path"))
        self.From_label_CI.setText(_translate("MainWindow", "From Day"))
        self.FolderBrowse_pushButton_CI.setText(_translate("MainWindow", "Browse"))
        self.CreateImages_pushButton.setText(_translate("MainWindow", "Create Images"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.CreateImages_tab), _translate("MainWindow", "Create Images"))


        self.UpperBound_Label_SM.setText(_translate("MainWindow", "Upper Bound"))
        self.UpperBound_lineEdit_SM.setPlaceholderText("99.5")
        self.ViewPeak_pushButton_SM.setText(_translate("MainWindow", "Peaks"))
        self.End_SongFile.setText(_translate("MainWindow", "End"))
        self.End_SongFile_lineEdit_SM.setText(_translate("MainWindow", "1"))
        self.Beginning_SongFile.setText(_translate("MainWindow", "Beginning"))
        self.Beginning_SongFile_lineEdit_SM.setText(_translate("MainWindow", "1"))
        self.CreateGraph_pushButton.setText(_translate("MainWindow", "Bounds"))
        
        
        self.To_label_SM.setText(_translate("MainWindow", "To Day"))
        self.BirdPath_label_SM.setText(_translate("MainWindow", "Folder Path"))
        self.From_label_SM.setText(_translate("MainWindow", "From Day"))
        self.FolderBrowse_pushButton_SM.setText(_translate("MainWindow", "Browse"))
        self.SegmentMotif_pushButton.setText(_translate("MainWindow", "Segment Motifs"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SegmentMotif_tab), _translate("MainWindow", "Segment Motifs"))


        self.To_label_CMI.setText(_translate("MainWindow", "To Day"))
        self.BirdPath_label_CMI.setText(_translate("MainWindow", "Folder Path"))
        self.From_label_CMI.setText(_translate("MainWindow", "From Day"))
        self.FolderBrowse_pushButton_CMI.setText(_translate("MainWindow", "Browse"))
        self.CreateMotifImages_pushButton.setText(_translate("MainWindow", "Create Motif Images"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.CreateMotifImages_tab), _translate("MainWindow", "Create Motif Images"))

        self.BirdPath_label_SC.setText(_translate("MainWindow", "Folder Path"))
        self.Bird_Letter_SC.setText(_translate("MainWindow", "First Letter of Bird"))
        self.FolderBrowse_pushButton_SC.setText(_translate("MainWindow", "Browse"))
        self.SyllableCluster_pushButton.setText(_translate("MainWindow", "Cluster"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SyllableCluster_tab), _translate("MainWindow", "Syllable Cluster"))

        self.BirdPath_label_MC.setText(_translate("MainWindow", "Folder Path"))
        self.Bird_Letter_MC.setText(_translate("MainWindow", "First Letter of Bird"))
        self.FolderBrowse_pushButton_MC.setText(_translate("MainWindow", "Browse"))
        self.MotifCluster_pushButton.setText(_translate("MainWindow", "Cluster"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.MotifCluster_tab), _translate("MainWindow", "Motif Cluster"))
        
        self.ClusterPath_label_NN.setText(_translate("MainWindow", "Image Path"))
        self.Root_label_NN.setText(_translate("MainWindow", "Bird Path"))
        self.Temp_label_NN.setText(_translate("MainWindow", "Temp Path"))

        self.BirdPath_label_NN.setText(_translate("MainWindow", "# of Folders"))
        self.From_label_NN.setText(_translate("MainWindow", "From"))
        self.To_label_NN.setText(_translate("MainWindow", "To"))
        self.FolderBrowse_pushButton_NN.setText(_translate("MainWindow", "Browse"))
        self.FolderBrowse_pushButton_NN2.setText(_translate("MainWindow", "Browse"))
        self.FolderBrowse_pushButton_NN3.setText(_translate("MainWindow", "Browse"))
        self.SyllNN_pushButton.setText(_translate("MainWindow", "DL Cluster"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SyllNN_tab), _translate("MainWindow", "Syllable DL"))


        self.ClusterPath_label_NNM.setText(_translate("MainWindow", "Image Path"))
        self.Root_label_NNM.setText(_translate("MainWindow", "Bird Path"))
        self.Temp_label_NNM.setText(_translate("MainWindow", "Temp Path"))
        self.BirdPath_label_NNM.setText(_translate("MainWindow", "# of Folders"))
        self.From_label_NNM.setText(_translate("MainWindow", "From"))
        self.To_label_NNM.setText(_translate("MainWindow", "To"))
        self.FolderBrowse_pushButton_NNM.setText(_translate("MainWindow", "Browse"))
        self.FolderBrowse_pushButton_NNM2.setText(_translate("MainWindow", "Browse"))
        self.FolderBrowse_pushButton_NNM3.setText(_translate("MainWindow", "Browse"))
        self.MotifNN_pushButton.setText(_translate("MainWindow", "DL Cluster"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.MotifNN_tab), _translate("MainWindow", "Motif DL"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

