# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:11:37 2020

@author: sheng
"""
import numpy as np
import os
import wave
from scipy import signal
import matplotlib.pyplot as plt
home = "./more_sample_data/"

# %% Read in raw data

# Get the file name and path of the file
def get_file_name(naudio, nfile):
    path = get_dir(naudio)
    file = str(naudio)+'_'+ str(nfile)+ '_filtered' + ".wav"
    path = path + "/"+file
    return path, file

def get_dir(naudio):
    naudio_str = str(naudio)
    subpath = naudio_str
    path = home + subpath + '/bandpass_filtered'
    return path

# Read the wave data of a file and stire ub an array
def read_wave(file_path):
    f = wave.open(file_path,"rb")
    params = f.getparams()
    _, _, framerate, nframes = params[:4]
    #Reads and returns nframes of audio, as a string of bytes. 
    str_data = f.readframes(nframes)
    #close the stream
    f.close()
    #turn the wave's data to array
    wave_data = np.fromstring(str_data, dtype = np.short)
    #transpose the data
    wave_data = wave_data.T
    #calculate the time bar
    time = np.arange(0, nframes) * (1.0/framerate)
    return wave_data, time, framerate

# Convert wave data to spectrum data
def get_spectrum(filepath):
    wave_data, time, framerate = read_wave(filepath)
    f,t,Sxx = signal.spectrogram(wave_data,fs=framerate,window=signal.get_window('hann',500))   #Using the hanning window to get the spectrum
    Sxx =[*Sxx[0:18],*Sxx[66:70]]       #Combine the spectrum at low frequency and high frequency
    return Sxx


# Read in wave data 
def extract_features(directory):
    '''
    Scan all the files and separate them in to gunshot and none-gunshot. In which:
    x_Gunshot,x_nGunshot: data in time domain
    s_Gunshot,s_nGunshot: data in spectrum
    As_Gunshot: Spectrum gunshot in Africa
    dataset: Spectrum gunshot in online dataset
    '''
    x_nGunshot = []
    x_Gunshot = []
    s_nGunshot = []
    s_Gunshot = []
    As_Gunshot = []
    s_online = []
    
    noise_path = os.path.join(directory, '\\0')
    gunshot_path = os.path.join(directory, '\\1')
    online_path = os.path.join(directory, '\\2')
    
    # Read noise files
    for file in os.listdir(noise_path):
        noise_data,_,_ = read_wave(noise_path)
        x_nGunshot.append(noise_data)
        Sxx = get_spectrum(noise_path)
        s_nGunshot.append(Sxx)
        
    # Read African gunshot files
    for file in os.listdir(gunshot_path):
        gunshot_data,_,_ = read_wave(gunshot_path)
        x_Gunshot.append(gunshot_data)
        Sxx = get_spectrum(gunshot_path)
        s_nGunshot.append(Sxx)
        As_Gunshot.append(Sxx)
        
    # Read online gunshot files
    for file in os.listdir(online_path):
        online_data,_,_ = read_wave(online_path)
        x_Gunshot.append(online_data)
        Sxx = get_spectrum(online_path)
        s_Gunshot.append(Sxx)
        s_online.append(Sxx)
    
    return x_Gunshot,s_Gunshot,x_nGunshot,s_nGunshot,s_online,As_Gunshot
            

def get_highest(wave_data, window_len, sample_rate):
    i = 0
    maximum = 0
    while (i+window_len*sample_rate) < len(wave_data):
        test1 = wave_data[i:int(i+window_len*sample_rate)]
        test = map(abs, test1)
        test = sum(test)
        if test > maximum :
            maximum = test
            store = test1
            t = i
        i += 10
    return maximum,store,t
    

# =============================================================================
# 
# if __name__ == "__main__":
#     wave_data, time, framerate = read_wave('Split/2/bandpass_filtered/gunshot_filtered.wav')
#     maximum,highest,t= get_highest(wave_data,0.1,framerate)
#     print(t)
#     plt.subplot(2,1,1)
#     plt.plot(range(0,len(wave_data)),wave_data)
#     plt.subplot(2,1,2)
#     plt.plot(range(0,len(highest)),highest)
#     plt.show()
#     
#     x_Gunshot,s_Gunshot,x_nGunshot,s_nGunshot = extract_features()
# =============================================================================
