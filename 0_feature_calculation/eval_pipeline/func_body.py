#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:34:42 2021

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

MAX_POINTS = 18 * 2

def calc_gesture_cv(body):
    
    #TODO!:
    #   consider 2d array index [[x, y], [x, y].
    #   currently, those index list (e.g. whole_column_names) is 1d array index and doesn't pick proper data from source data (this is bug)
        
    #body_column_array = np.asarray(body[0])
    #body_array = np.asarray(body[1:]).astype(float)
    
    upper_column_names = [
        0, 1, 2, 3, 4, 5, 6, 7, 
        14, 15, 16, 17
        ]
    arm_column_names = [
        2, 3, 4, 5, 6, 7
        ]
    whole_column_names = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17
        ]
    
    try:
        
        # Eliminate incomplete data, where complete means the number of maximum components in a row of the body
        max_points = 0
        for i in range(len(body)):
            if len(body[i]) > max_points:
                max_points = len(body[i])
        temp = []
        for i in range(len(body)):
            if len(body[i]) == max_points:
                temp.append(body[i])
                
        body_column_array = np.asarray(whole_column_names)
        body_array = np.asarray(temp, dtype=np.float32)
    
        
        # delete row with NaN
        #body_array = body_array[~np.isnan(body_array).any(axis=1)]
            
        upper_index_list = []
        for column_name in upper_column_names:
            #print(body_column_array)
            #print(column_name)
            #print(np.where(body_column_array == column_name))
            index = np.where(body_column_array == column_name)[0][0]
            upper_index_list.append(index)
            
        arm_index_list = []
        for column_name in arm_column_names:
            arm_index_list.append(np.where(body_column_array == column_name)[0][0])
        
        upper_column_array = body_column_array[upper_index_list]
        upper_array = body_array[:, upper_index_list]
    
        arm_column_array = body_column_array[arm_index_list]
        arm_array = body_array[:, arm_index_list]
        
        def xy2dist(array):
            
            dist_list = []
            for xy_row in array:
                dist_row = []
                for i in range(0, len(xy_row), 2):
                    dist = math.sqrt(xy_row[i]*xy_row[i] + xy_row[i+1]*xy_row[i+1])
                    dist_row.append(dist)
                dist_list.append(dist_row)
            dist_array = np.asarray(dist_list)
            
            return dist_array
        
        body_array = xy2dist(body_array)
        upper_array = xy2dist(upper_array)
        arm_array = xy2dist(arm_array)
        
        body_cv = np.std(body_array)/np.mean(body_array)
        upper_cv = np.std(upper_array)/np.mean(upper_array)
        arm_cv = np.std(arm_array)/np.mean(arm_array)
    
        if DEBUG:
            print('columns:', body_column_array)
            print('columns:', body_array)
            print('upper_index:', upper_index_list)
            print('arm_index:', arm_index_list)
            print('body_array:', body_array)
            print('upper_array:', upper_array)
            print('arm_array:', arm_array)
            print('body_cv:', body_cv)
            print('upper_cv:', upper_cv)
            print('arm_cv:', arm_cv)
    
    except Exception:
        
        print(traceback.print_exc())
        
        print('Body: calculation error. Replace feature values with 0.')
        
        body_cv = 0
        upper_cv = 0
        arm_cv = 0
        
    return [body_cv, upper_cv, arm_cv]
