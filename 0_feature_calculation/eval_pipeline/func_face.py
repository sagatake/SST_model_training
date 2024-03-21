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

def calc_mutual_smile(face_1, face_2):
    
    face_1_smile_freq_list = _calc_smile_flag(face_1)
    face_2_smile_freq_list = _calc_smile_flag(face_2)
    
    if len(face_1_smile_freq_list) > len(face_2_smile_freq_list):
        num_frames = len(face_2_smile_freq_list)
    else:
        num_frames = len(face_1_smile_freq_list)
    
    cnt = 0
    for i in range(num_frames):
        cnt += face_1_smile_freq_list[i]*face_2_smile_freq_list[i]
    
    feature = cnt / num_frames
    
    return [feature]

def calc_AU_stats(face):
    
    column_list = face[0]
    face = face[1:]
    
    features = []
    names = []
    
    face = np.asarray(face).astype(float)
    face = face.T.tolist()
    
    for column_index in range(len(column_list)):
        
        if ('AU' in column_list[column_index]) and ('_r' in column_list[column_index]):
            mean = np.mean(face[column_index])
            if np.mean(face[column_index]) == 0.0:
                cv = 0
            else:
                cv = np.std(face[column_index]) / np.mean(face[column_index])
            features.append(mean)
            features.append(cv)
            names.append(column_list[column_index]+'_mean')
            names.append(column_list[column_index]+'_cv')
    
    return [features, names]

def calc_smile_freq(face):
    
    """
    column_list = face[0]
    face = face[1:]

    num = 1
    #print(np.shape(face))
    
    num_frames = len(face[0])-1
    
    AU06_index = np.where(np.asarray(column_list)==' AU06_c')[0][0]
    AU12_index = np.where(np.asarray(column_list)==' AU12_c')[0][0]
    #print(pitch_index)
    
    face = np.asarray(face).astype(float)
    face = face.T.tolist()

    #calculate happiness
    num = 0
    for i in range(1, num_frames):
        if (face[AU06_index][i] == 1.0) and (face[AU12_index][i] == 1.0):
            num = num + face[AU06_index][i]*face[AU12_index][i]
    feature = num / (len(face[0])-1)
    """
    
    smile_flag_list = _calc_smile_flag(face)
    num_frames = len(smile_flag_list)
    feature = sum(smile_flag_list) / num_frames
        
    return [feature]

def calc_headpose(face):
        
    face_column_array = np.asarray(face[0])
    face_array = np.asarray(face[1:]).astype(float)

    headpose_column_names = [' pose_Rx', ' pose_Ry', ' pose_Rz']
    headpose_index_list = []
    for column_name in headpose_column_names:
        headpose_index_list.append(np.where(face_column_array == column_name)[0][0])
        
    headpose_array = face_array[:, headpose_index_list]
    
    headpose_mean = np.mean(headpose_array)
    headpose_cv = np.std(headpose_array)/np.mean(headpose_array)
    
    return [headpose_mean, headpose_cv]


def count_nods(face):
    
    pitch_index = 0
    frame_window_size = 10
    same_angle_threshold_degree = 5.0
    max_angle_threshold_degree = 30.0
        
    column_list = face[0]
    face = face[1:]

    face = np.asarray(face).astype(float)
    pitch = face[:, pitch_index] * 180 / math.pi
    
    #init counter
    n_nod = 0
    
    #init pitch queue
    q = _Queue()
    for i in range(frame_window_size):
        q.put(pitch[i])
    
    for frame in pitch[frame_window_size:]:
        if (abs(q.maximum() - q.minimum()) > max_angle_threshold_degree) \
            and (abs(q.queue[-1] - q.queue[0]) < same_angle_threshold_degree):
                n_nod += 1
        q.update(frame)
    
    return [n_nod]

def count_nods_kawato(face):
    
    LOCAL_DEBUG = False

    frame_window_size = 5
    same_angle_threshold_degree = 3
    #max_angle_threshold_degree = 30.0
    
    half_window_size = math.floor(frame_window_size/2)

    if LOCAL_DEBUG or DEBUG:
        print(same_angle_threshold_degree * math.pi / 180)
    
    
    column_list = face[0]
    face = face[1:]
    
    pitch_index = np.where(np.asarray(column_list)==' pose_Rx')[0][0]
    #print(pitch_index)
    
    face = np.asarray(face).astype(float)
    pitch = face[:, pitch_index] * 180 / math.pi
    #print(pitch[:50])
    
    #init counter
    n_nod = 0
    
    #init pitch queue
    pitch_q = _Queue()
    for i in range(frame_window_size):
        pitch_q.put(pitch[i])

    #init state queue #stable:0, extreme:1, transient:2
    state_q = _Queue()
    for i in range(frame_window_size):
        state_q.put(0)
        
    for frame in pitch[half_window_size+1:]:
        
        #print('{:.2f} deg'.format(abs(pitch_q.queue[-1] - frame)))
        
        # update state_q
        #if (pitch_q.maximum() - pitch_q.minimum()) < same_angle_threshold_degree:
        if abs(pitch_q.queue[-1] - frame) < same_angle_threshold_degree:
            state_q.update(0)
        elif (frame < pitch_q.minimum()) or (pitch_q.maximum() < frame):
            state_q.update(1)
        else:
            state_q.update(2)

        pitch_q.update(frame)
            
        # nod_evaluation
        if (state_q.queue[-1] != state_q.queue[-2]) and (state_q.queue[-1] == 0):
            
            # condition flag
            nod_detected = 0
            adj_extreme_differ = 0
            
            # condition1: if more than two extreme state were detected between stable states
            extreme_count = 0
            #stable_detected = 0
            for state in state_q.queue:
                if state == 0:
                    if 2 <= extreme_count:
                        nod_detected = 1
                    #stable_detected = 1
                    extreme_count = 0
                #elif (stable_detected == 1) and (state == 1):
                elif (state == 1):
                    extreme_count += 1

                if DEBUG or LOCAL_DEBUG:
                    print('state: ', state)
                    print('extreme_count: ', extreme_count)
                                    
            
            if DEBUG or LOCAL_DEBUG:
                print('nod_detected: ', nod_detected)
                
            # condition2: check whether all adjacent extreme frame differ with more than same_angle_threshold
            for i in range(1, frame_window_size):
                if (state_q.queue[i-1] == 1) and (state_q.queue[i] == 1):
                    if abs(abs(pitch_q.queue[i-1]) - abs(pitch_q.queue[i])) >= same_angle_threshold_degree:
                        adj_extreme_differ = 1
                    elif abs(abs(pitch_q.queue[i-1]) - abs(pitch_q.queue[i])) < same_angle_threshold_degree:
                        adj_extreme_differ = 0
                        
                if DEBUG or LOCAL_DEBUG:
                    print('pitch[i-1]: ', pitch_q.queue[i-1])
                    print('pitch[i]: ', pitch_q.queue[i])
                    print('pitch diff', abs(pitch_q.queue[i]-pitch_q.queue[i-1]))
                    print('adj_extreme_differ: ', adj_extreme_differ)


            if DEBUG or LOCAL_DEBUG:
                print('adj_extreme_differ: ', adj_extreme_differ)
            
            # final decision for nod detection
            if nod_detected and adj_extreme_differ:
                n_nod += 1
                
        if LOCAL_DEBUG:
            print('state_q: ', state_q.queue)
            print('pitch_q: ', pitch_q.queue)
            print('n_nod: ', n_nod)
            input('Type enter to increment')
    
    return [n_nod]

def _calc_smile_flag(face):
    
    column_list = face[0]
    face = face[1:]    
    
    AU06_index = np.where(np.asarray(column_list)==' AU06_c')[0][0]
    AU12_index = np.where(np.asarray(column_list)==' AU12_c')[0][0]
    #print(pitch_index)
    
    face = np.asarray(face).astype(float)
    face = face.T.tolist()

    num_frames = len(face[0])-1

    smile_flag_list = []
    #calculate happiness
    for i in range(1, num_frames):
        if (face[AU06_index][i] == 1.0) and (face[AU12_index][i] == 1.0):
            #smile_flag_list.append([1.0])
            smile_flag_list.append(1.0)
        else:
            #smile_flag_list.append([0.0])
            smile_flag_list.append(0.0)
    return smile_flag_list

    
class _Queue:          
          
    def __init__(self):
        self.queue = []
        
    def put(self, item):
        self.queue.append(item)
        
    def get(self):
        out = self.queue[0]
        self.queue = self.queue[1:]
        return out
    
    def update(self, item):
        self.put(item)
        _ = self.get()
    
    def minimum(self):
        return min(self.queue)
    
    def maximum(self):
        return max(self.queue)
