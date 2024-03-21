#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:23:32 2021

@author: takeshi-s
"""
from matplotlib import pyplot as plt
import pprint as pp
import pandas as pd
import numpy as np
import traceback
import shutil
import math
import time
import csv
import sys
import os

#dummy value. This will be alwayse over-rided in the parent source code.
DEBUG = None

def calc_ave_voice_int(audio_intensity):
            
    #audio_intensity : [frame_num, intensity]
    temp = [y[1] for y in audio_intensity if(not y[1] == -300)]
    
    intensity_mean = np.mean(np.asarray(temp), axis=0)
    intensity_max = max(temp)
    intensity_min = min(temp)
    intensity_range = intensity_max - intensity_min
    intensity_sd = np.std(temp)
    if DEBUG:
        print("Intensity_MEAN = {:.3f}".format(intensity_mean))
        print("Intensity_MAX = {:.3f}".format(intensity_max))
        print("Intensity_MIN = {:.3f}".format(intensity_min))
        print("Intensity_Range = {:.3f}".format(intensity_range))
        print("Intensity_SD = {:.3f}".format(intensity_sd))
        
    return [intensity_mean]

def calc_cv_voice_f0(audio_f0):
    
    cv_f0 = np.std(audio_f0)/np.mean(audio_f0)
    
    return [cv_f0]