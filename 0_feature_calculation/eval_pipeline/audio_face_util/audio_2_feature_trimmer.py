# -*- coding: utf-8 -*-
"""
Updated on 2022年  2月 17日 木曜日 20:42:00 JST

Author : Takeshi Saga

"""

import os
import numpy as np
import csv
import pprint
import re
import shutil

DEBUG = 0

wav_dir = './src_audio'
#feature_dir = './tmp_audio_out'
feature_dir = '.\\tmp_audio_out'

ENERGY      = False
F0          = True
INTENSITY   = True
FORMANT     = False
JITTER      = False

def main():
    
    try:
        file_names = os.listdir(wav_dir)
        #print(file_names)
        file_names = sorted(file_names, key=lambda x:int(re.search(r"[0-9]+", x).group(0)))
        #pprint.pprint(file_names)
    except:
        file_names = os.listdir(wav_dir)
    
    prosodic_feature_list = []
    
    feature_name_list = ['ID']
    
    if ENERGY:
        feature_name_list.extend(['Energy'])
    if F0:
        feature_name_list.extend(['F0_MEAN','F0_MAX','F0_MIN','F0_RANGE','F0_SD'])
    if INTENSITY:
        feature_name_list.extend(['Int_MEAN','Int_MAX','Int_MIN','Int_RANGE','Int_SD'])
    if FORMANT:
        feature_name_list.extend([
            'F1_freq_MEAN','F2_freq_MEAN','F3_freq_MEAN',
            'F1_SD','F2_SD','F3_SD',
            'F1_band_MEAN','F2_band_MEAN','F3_band_MEAN',
            'F2/F1_MEAN','F3/F1_MEAN',
            'F2/F1_SD','F3/F1_SD'])
    if JITTER:
        feature_name_list.extend(['Jitter','Shimmer','Unvoiced','Breaks'])
    
    prosodic_feature_list.append(feature_name_list)

    
    for index in range(len(file_names)):
        
        x = file_names[index]
        base_name,ext = x.split('.')
        
        #print('Now processing : {}'.format(x))
        #print(os.path.join(feature_dir, ' '+base_name+'_energy.csv'))
        
        #print(os.listdir(feature_dir))
        
        
        #Energy
        if ENERGY:
            with open(os.path.join(feature_dir, ' '+base_name+'_energy.csv'), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                tmp = [y for y in reader]
                energy = float(tmp[0][0])
                if DEBUG:
                    print("Energy = {:.3f}".format(energy))
        
        #F0(Pitch)
        if F0:
            with open(os.path.join(feature_dir, ' '+base_name+'_pitch.csv'), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                tmp = []
                for y in reader:
                    tmp.append([])
                    for z in y:
                        if not z=='':
                            tmp[-1].append(z)
                
                f0_list = []
                for i in range(len(tmp)):
                    if 'frames' in tmp[i]:
                        j = 0
                        while True:
                            if 'frequency' in tmp[i+j]:
                                frame_num = tmp[i][1]
                                frame_num = frame_num[1:-2]
                                
                                #To avoid exception
                                if frame_num == '':
                                    break
                                
                                f0_list.append([int(frame_num), float(tmp[i+j][2])])
                                break
                            j += 1
                            
                #f0_list : [frame_num, frequency]
                            
                tmp = [y[1] for y in f0_list if(not y[1] == 0) ]
                f0_mean = np.mean(np.asarray(tmp), axis=0)
                f0_max = max(tmp)
                f0_min = min(tmp)
                f0_range = f0_max - f0_min
                f0_sd = np.std(tmp)
                if DEBUG:
                    print("F0_MEAN = {:.3f}".format(f0_mean))
                    print("F0_MAX = {:.3f}".format(f0_max))
                    print("F0_MIN = {:.3f}".format(f0_min))
                    print("F0_Range = {:.3f}".format(f0_range))
                    print("F0_SD = {:.3f}".format(f0_sd))
                            
        #Intensity
        if INTENSITY:
            with open(os.path.join(feature_dir, ' '+base_name+'_intensity.csv'), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                tmp = []
                for y in reader:
                    tmp.append([])
                    for z in y:
                        if not z=='':
                            tmp[-1].append(z)
                #pprint.pprint(tmp[0:50])
                
                intensity_list = []
                for i in range(len(tmp)):
                    if 'z' in tmp[i]:
                        if len(tmp[i])==5:
                            frame_num = tmp[i][2]
                            frame_num = frame_num[1:-1]
                            
                            #To avoid exception
                            if frame_num == '':
                                continue
                            
                            intensity_list.append([int(frame_num),float(tmp[i][4])])
                
                #intensity_list : [frame_num, intensity]
                
                tmp = [y[1] for y in intensity_list if(not y[1] == -300)]
                intensity_mean = np.mean(np.asarray(tmp), axis=0)
                intensity_max = max(tmp)
                intensity_min = min(tmp)
                intensity_range = intensity_max - intensity_min
                intensity_sd = np.std(tmp)
                if DEBUG:
                    print("Intensity_MEAN = {:.3f}".format(intensity_mean))
                    print("Intensity_MAX = {:.3f}".format(intensity_max))
                    print("Intensity_MIN = {:.3f}".format(intensity_min))
                    print("Intensity_Range = {:.3f}".format(intensity_range))
                    print("Intensity_SD = {:.3f}".format(intensity_sd))
        
        #Formant
        if FORMANT:
            with open(os.path.join(feature_dir, ' '+base_name+'_formant.csv'), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                tmp = []
                for y in reader:
                    tmp.append([])
                    for z in y:
                        if not z=='':
                            tmp[-1].append(z)
            
                formant_list = []
                for i in range(len(tmp)):
                    if 'numberOfFormants' in tmp[i]:
                        n_frm = int(tmp[i][2])
                        features = []
                        for j in range(n_frm):
                            features.append(float(tmp[i+3*(j+1)][2]))
                            features.append(float(tmp[i+3*(j+1)+1][2]))
                        for j in range(3-n_frm):
                            features.append(np.nan)
                            features.append(np.nan)
                        formant_list.append(features)
                
                # formant_list : [f1_freq, f1_band, f2_freq, f2_band, f3_freq, f3_band]
                f2_on_f1 = []
                f3_on_f1 = []
                for i in range(len(formant_list)):
                    try:
                        f2_on_f1.append(formant_list[i][2]/formant_list[i][0])
                    except:
                        f2_on_f1.append(np.nan)
                    try:
                        f3_on_f1.append(formant_list[i][4]/formant_list[i][0])
                    except:
                        f3_on_f1.append(np.nan)            
                
                #tmp = [y[1] for y in formant_list if(not y[1] == -300)]
                tmp = formant_list
                f1_freq_mean, f1_band_mean, f2_freq_mean, f2_band_mean, f3_freq_mean, f3_band_mean = np.nanmean(np.asarray(tmp), axis=0)
                f1_sd, _, f2_sd, _, f3_sd, _ = np.nanstd(tmp, axis=0)
                f2_on_f1_mean = np.nanmean(f2_on_f1, axis=0)
                f2_on_f1_sd = np.nanstd(f2_on_f1, axis=0)
                f3_on_f1_mean = np.nanmean(f3_on_f1, axis=0)
                f3_on_f1_sd = np.nanstd(f3_on_f1, axis=0)
                
                if DEBUG:
                    print("F1_freq_MEAN = {:.3f}".format(f1_freq_mean))
                    print("F2_freq_MEAN = {:.3f}".format(f2_freq_mean))
                    print("F3_freq_MEAN = {:.3f}".format(f3_freq_mean))
                    print("F1_band_MEAN = {:.3f}".format(f1_band_mean))
                    print("F2_band_MEAN = {:.3f}".format(f2_band_mean))
                    print("F3_band_MEAN = {:.3f}".format(f3_band_mean))
                    print("F1_SD = {:.3f}".format(f1_sd))
                    print("F2_SD = {:.3f}".format(f2_sd))
                    print("F3_SD = {:.3f}".format(f3_sd))
                    print("F2/F1_MEAN = {:.3f}".format(f2_on_f1_mean))
                    print("F3/F1_MEAN = {:.3f}".format(f3_on_f1_mean))
                    print("F2/F1_SD = {:.3f}".format(f2_on_f1_sd))
                    print("F3/F1_SD = {:.3f}".format(f3_on_f1_sd))
        
        
        #Jitter/Shimmer/Unvoiced/breaks
        if JITTER:
            with open(os.path.join(feature_dir, ' '+base_name+'_voice_report.csv'), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                tmp = []
                for y in reader:
                    tmp.append([])
                    for z in y:
                        if not z=='':
                            tmp[-1].append(z)
                #pprint.pprint(tmp[:50])
                for i in range(len(tmp)):
                    if 'Jitter:' in tmp[i]:
                        jitter = tmp[i+1][2]
                        jitter = float(jitter[:-1])
                    if 'Shimmer:' in tmp[i]:
                        shimmer = tmp[i+1][2]
                        shimmer = float(shimmer[:-1])
                    if 'Voicing:' in tmp[i]:
                        unvoiced = tmp[i+1][5]
                        unvoiced= float(unvoiced[:-1])
                        breaks = tmp[i+3][4]
                        breaks = float(breaks[:-1])
                
                if DEBUG:
                    print("Jitter = {:.3f}".format(jitter))
                    print("Shimmer = {:.3f}".format(shimmer))
                    print("Unvoiced = {:.3f}".format(unvoiced))
                    print("Breaks = {:.3f}".format(breaks))
        
        """
        ['ID',
         'Energy',
         'F0_MEAN','F0_MAX','F0_MIN','F0_RANGE','F0_SD',
         'Int_MEAN','Int_MAX','Int_MIN','Int_RANGE','Int_SD',
         'F1_freq_MEAN','F2_freq_MEAN','F3_freq_MEAN',
         'F1_SD','F2_SD','F3_SD',
         'F1_band_MEAN','F2_band_MEAN','F3_band_MEAN',
         'F2/F1_MEAN','F3/F1_MEAN',
         'F2/F1_SD','F2/F1_SD',
         'Jitter','Shimmer','Unvoiced','Breaks']
        """
        
        prosodic_feature_list_row = [file_names[index].split('.')[0]]
        if ENERGY:
            prosodic_feature_list_row.extend([energy])
        if F0:
            prosodic_feature_list_row.extend([f0_mean, f0_max, f0_min, f0_range, f0_sd])
        if INTENSITY:
            prosodic_feature_list_row.extend([intensity_mean, intensity_max, intensity_min, intensity_range, intensity_sd])
        if FORMANT:
            prosodic_feature_list_row.extend([
                f1_freq_mean, f2_freq_mean, f3_freq_mean,
                f1_sd, f2_sd, f3_sd,
                f1_band_mean,f2_band_mean, f3_band_mean,
                f2_on_f1_mean, f3_on_f1_mean,
                f2_on_f1_sd, f3_on_f1_sd])
        if JITTER:
            prosodic_feature_list_row.extend([jitter, shimmer, unvoiced, breaks])
        prosodic_feature_list.append(prosodic_feature_list_row)
    
    with open('feature_audio.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(prosodic_feature_list)        
               
if __name__ == '__main__':
    main()