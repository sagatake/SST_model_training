#!/usr/bin/env python3
# coding: utf-8
"""
Created on Mon Mar  1 10:15:00 2022
Updated on Fri Jun 17 12:18:00 2022

@author: takeshi-s

It is better to use Windows machine since Openface on docker support multiprocessing only for Windows 

"""

import os
import csv
import sys
import cv2
import json
import time
import copy
import shutil
import platform
import datetime
import argparse
import threading
import subprocess
import numpy as np
import pandas as pd
import pprint as pp
import pickle as pkl
from tqdm import tqdm
from pydub import AudioSegment

#from eval_pipeline import audio_face_util

#from eval_pipeline.openpose_src import util as openpose_util
#from eval_pipeline.openpose_src.body import Body
#from eval_pipeline.openpose_src.hand import Hand

sys.path.append(os.path.dirname(__file__))
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data'))
#pp.pprint(sys.path)

import audio_face_util
import reset_util

from openpose_src import util as openpose_util
from openpose_src.body import Body
from openpose_src.hand import Hand
import torch

#print(dir(util))

TEXT = False
AUDIO = False
FACE = False
BODY = False

TEXT = True
AUDIO = True
FACE = True
BODY = True

NO_MUTUAL = True

if TEXT:
    import func_text
if AUDIO:
    import func_audio
if FACE:
    import func_face
if BODY:
    import func_body

VIS = False

modarity_flags = [TEXT, AUDIO, FACE, BODY]
num_modality = sum(modarity_flags)

FEATURE_EXTRACT = True
FEATURE_CALC = True
PREDICT = True
RECORD = True

SELECT_PREDICTION_TARGET = True
elimination_dict ={
    # "TELL":{"視線", "体の向きと距離", "明晰さ", "流暢さ"},
    # "LISTEN":{"体の向きと距離", "声の変化", "明晰さ", "流暢さ"},
    # "FAVOR":{"視線", "流暢さ"},
    # "REFUSE":{"表情", "声の変化", "明晰さ", "流暢さ"}
    "TELL":{},
    "LISTEN":{},
    "FAVOR":{},
    "REFUSE":{}
    }

with_FB_upper_thread = True
FB_score_thread = 4.0

# To speed up processing speed of openpose, skip frames with this number
#video_frame_skip_factor = 30
video_frame_skip_factor = 1
RESIZE_FACTOR = 5
#RESIZE_FACTOR = 3

DEBUG = False
SHORT_INSPECT = False
SHORT_THRESHOLD = 3

STRATEGY_BEST_WORST = False
STRATEGY_NM = True

RESCALE = True
#original_scale_range = [3, 5]
#target_scale_range = [1, 5]

original_scale_range = [3, 5]
target_scale_range = [1, 5]

COPY_FINALIZE_WAIT_TIME = 3

#Feature extraction
lookup_list_path = 'data/lookup.csv'
#
base_data_dir = 'data'
#
text_src_dir = 'src_text'
audio_src_dir = 'src_audio'
#face_src_dir = 'data\\src_face'
#body_src_dir = 'data\\src_body'
face_src_dir = 'src_video'
body_src_dir = 'src_video'
#
text_data_dir = os.path.join(base_data_dir, 'data_text')
audio_data_dir = os.path.join(base_data_dir, 'data_audio')
face_data_dir = os.path.join(base_data_dir, 'data_face')
body_data_dir = os.path.join(base_data_dir, 'data_body')
#
openpose_model_dir = 'eval_pipeline\\openpose_model'
#
praat_dir = 'eval_pipeline/audio_face_util'
praat_output_dir = 'eval_pipeline/audio_face_util/temp_audio_out'
audio_src_dir_from_praat_dir = '../../{}'.format(audio_src_dir)
praat_output_dir_from_praat_dir = 'temp_audio_out'
#praat_output_dir_list = praat_output_dir.split('\\')
praat_file = 'audio_2_extractor.praat'

#Feature calculation
#main_keyword = 'user'
main_keyword = 'user'
inter_keyword = 'agent'
all_keyword = 'all'
feature_file_path = 'data/features.csv'

#Score prediction
json_path = 'data/eval_results.json'

#Record
base_record_dir = 'record'
text_record_dir = 'record_text'
audio_record_dir = 'record_audio'
face_record_dir = 'record_face'
body_record_dir = 'record_body'

base_result_dir = 'result_for_viewer'



pf = platform.system()

results = {}

    
def pipeline(task_name = 'REFUSE', num_positive = 1, num_negative = 1):
    
    _prepare_dirs()
        
    global text_src_dir, audio_src_dir, face_src_dir, body_src_dir
    global text_data_dir, audio_data_dir, face_data_dir, body_data_dir

    feature_names = ['ID']

    for name in os.listdir(face_src_dir):
        if not(main_keyword in name):
            print(name)
            print(main_keyword)
            _add_name_tag(face_src_dir, main_keyword)
            break

    for name in os.listdir(audio_src_dir):
        if not(main_keyword in name):
            print(name)
            print(main_keyword)
            _add_name_tag(audio_src_dir, main_keyword)
            break
        
    s_time = time.time()    

    if FEATURE_EXTRACT:
        
        if TEXT:
            text_thread = text2csv()#Don't need
        if AUDIO:
            audio_thread = audio2csv()
        if FACE:
            face_thread = face2csv()
        if BODY:
            body_thread = body2csv()#TODO! 0.3sec per frame, about 60sec(30fps, from 2000 to 4000 frames) per file
        
        if DEBUG:
            if AUDIO:
                print(type(audio_thread))
            if FACE:
                print(type(face_thread))
            if BODY:
                print(type(body_thread))
        
        if TEXT:
            text_thread.start()
            time.sleep(0.01)
        if AUDIO:
            audio_thread.start()
            time.sleep(0.01)
        if FACE:
            face_thread.start()
            time.sleep(0.01)
        if BODY:
            body_thread.start()
            time.sleep(0.01)
        
        if TEXT:
            text_thread.join()
        if AUDIO:
            audio_thread.join()
        if FACE:
            face_thread.join()
        if BODY:
            body_thread.join()
    
    if FEATURE_CALC:


        
        if TEXT:
            text_main_IDs, text_main_subjects = _load_text_data(text_data_dir, keyword = main_keyword)
            text_all_IDs, text_all_subjects = _load_text_data(text_data_dir, keyword = all_keyword)
        if AUDIO:
            audio_IDs, audio_int_subjects = _load_audio_int_data(audio_data_dir, keyword = main_keyword)
            audio_IDs, audio_f0_subjects = _load_audio_f0_data(audio_data_dir, keyword = main_keyword)    
        if FACE or TEXT:
            face_main_IDs, face_main_subjects = _load_face_data(face_data_dir, keyword = main_keyword)
        if FACE:
            if NO_MUTUAL:
                face_inter_IDs, face_inter_subjects = face_main_IDs, face_main_subjects #TODO! dammy data
            else:
                face_inter_IDs, face_inter_subjects = _load_face_data(face_data_dir, keyword = inter_keyword)
        if BODY:
            body_IDs, body_subjects = _load_body_data(body_data_dir, keyword = main_keyword)
        
        if TEXT:
            text_main_IDs, text_features, text_feature_names = calc_text(
                text_main_IDs, text_main_subjects, 
                text_all_IDs, text_all_subjects, 
                face_main_IDs, face_main_subjects
                )
        if AUDIO:
            audio_IDs, audio_features, audio_feature_names = calc_audio(
                audio_IDs, audio_int_subjects, audio_f0_subjects
                )
        if FACE:
            face_main_IDs, face_features, face_feature_names = calc_face(
                face_main_IDs, face_main_subjects, 
                face_inter_IDs, face_inter_subjects
                )
        if BODY:
            body_IDs, body_features, body_feature_names = calc_body(
            body_IDs, body_subjects
            )
            
        ID_dict = {}

        if TEXT:
            for ID in text_main_IDs:
                if ID in ID_dict.keys():
                    ID_dict[ID] += 1
                else:
                    ID_dict[ID] = 1
            if DEBUG:
                print('########## TEXT ##########')
                print(ID_dict.items())
                #input('Enter:')

        if AUDIO:
            for ID in audio_IDs:
                if ID in ID_dict.keys():
                    ID_dict[ID] += 1
                else:
                    ID_dict[ID] = 1
            if DEBUG:
                print('########## AUDIO ##########')
                print(ID_dict.items())
                #input('Enter:')
        
        if FACE:
            for ID in face_main_IDs:
                if ID in ID_dict.keys():
                    ID_dict[ID] += 1
                else:
                    ID_dict[ID] = 1
            if DEBUG:
                print('########## FACE ##########')
                print(ID_dict.items())
                #input('Enter:')
        
        if BODY:
            for ID in body_IDs:
                if ID in ID_dict.keys():
                    ID_dict[ID] += 1
                else:
                    ID_dict[ID] = 1
            if DEBUG:
                print('########## BODY ##########')
                print(ID_dict.items())
                #input('Enter:')
                
        available_IDs = list(ID_dict.keys())
        
        for ID in ID_dict.keys():
            if ID_dict[ID] != num_modality:
                available_IDs.remove(ID)
         
        if DEBUG:
            if TEXT:
                pp.pprint(text_features)
                print(np.shape(text_features))
            if AUDIO:
                pp.pprint(audio_features)
                print(np.shape(audio_features))
            if FACE:
                pp.pprint(face_features)
                print(np.shape(face_features))
            if BODY:
                pp.pprint(body_features)
                print(np.shape(body_features))
        
        if TEXT:
            text_features =_filter_by_ID(available_IDs, text_features)
            text_features = np.asarray(text_features)
        if AUDIO:
            audio_features =_filter_by_ID(available_IDs, audio_features)
            audio_features = np.asarray(audio_features)
        if FACE:
            face_features =_filter_by_ID(available_IDs, face_features)
            face_features = np.asarray(face_features)
        if BODY:
            body_features =_filter_by_ID(available_IDs, body_features)
            body_features = np.asarray(body_features)

        available_IDs = np.asarray(available_IDs).reshape([-1, 1])
        
        if DEBUG:
            if TEXT:
                print(text_features.shape)
            if AUDIO:
                print(audio_features.shape)
            if FACE:
                print(face_features.shape)
            if BODY:
                print(body_features.shape)
        
        temp_pool = []
        temp_pool.append(available_IDs)
        if TEXT:
            temp_pool.append(text_features[:, 1:])
        if AUDIO:
            temp_pool.append(audio_features[:, 1:])
        if FACE:
            temp_pool.append(face_features[:, 1:])
        if BODY:
            temp_pool.append(body_features[:, 1:])
        final_features = np.concatenate(
            temp_pool,
            axis = 1
            )

        temp_pool = []
        if TEXT:
            temp_pool.append(text_feature_names)
        if AUDIO:
            temp_pool.append(audio_feature_names)
        if FACE:
            temp_pool.append(face_feature_names)
        if BODY:
            temp_pool.append(body_feature_names)
        
        for names in temp_pool:
            feature_names.extend(names)
        
        final_features = final_features.tolist()
        final_features.insert(0, feature_names)
        
        if DEBUG:
            pp.pprint(final_features)
        
        _write_csv(feature_file_path, final_features)
           
    if PREDICT:#important
        predicted_score = predict(task_name)
        
        if RESCALE:
            predicted_score = rescale(predicted_score)
        
        if SELECT_PREDICTION_TARGET:
            predicted_score = select_pred_target(predicted_score, task_name)
        
        if STRATEGY_BEST_WORST:
            feedbacks = generate_feedbacks(predicted_score, task_name)
        if STRATEGY_NM:
            feedbacks = generate_feedbacks(predicted_score, task_name, num_positive, num_negative)
            
        score = {'Score':predicted_score}
        results.update(score)
        results.update(feedbacks)
        pp.pprint(results)
            
        _write_json(results, json_path, encoding = 'shift-jis')

       # score_ = score.values()

        
   #     maxv = max(predicted_score.values())
    #    minv = min(predicted_score.values())

      #  print(maxv)
     #   print(minv)
      #  print(predicted_score)

      #  score=[x/maxv for x in score_]
      #  score_norm = {key: ((x - minv ) / (maxv - minv) )  for (key, x) in score.items() }
        
     #   pp.pprint(score_norm)
        

    
   
  #  print(predicted_score)
    
    
    if RECORD:
        _record_data(user_ID = None)
        print("Waiting for finalization... ({} sec.)".format(COPY_FINALIZE_WAIT_TIME))
        time.sleep(COPY_FINALIZE_WAIT_TIME)
        print("Done")

    e_time = time.time()
    print('Elapsed: ', e_time - s_time)
    
    
    #_remove_name_tag(text_src_dir, 'user')
    #_remove_name_tag(audio_src_dir, 'user')
    #_remove_name_tag(face_src_dir, 'user')

    return results

def select_pred_target(predicted_score, task_name):
    
    # SELECT_PREDICTION_TARGET = True
    # elimination_dict ={
    #     "TELL":{"視線", "体の向きと距離", "明晰さ", "流暢さ"},
    #     "LISTEN":{"体の向きと距離", "声の変化", "明晰さ", "流暢さ"},
    #     "FAVOR":{"視線", "流暢さ", "社会的妥当性"},
    #     "REFUSE":{"表情", "声の変化", "明晰さ", "流暢さ"}
    #     }
    filtered_score = {}
    for key in predicted_score.keys():
        if key in elimination_dict[task_name]:
            pass
        else:
            filtered_score[key] = predicted_score[key]
            
    if "社会的妥当性" in filtered_score.keys():
        if task_name == "TELL":
            task_specific_score = "相手の反応を気にした話しかた"
        if task_name == "LISTEN":
            task_specific_score = "積極的な聴きかた"
        if task_name == "FAVOR":
            task_specific_score = "相手に内容が伝わる頼みかた"
        if task_name == "REFUSE":
            task_specific_score = "相手との関係性を壊さないような話しかた"
        filtered_score[task_specific_score] = filtered_score["社会的妥当性"]
        filtered_score.pop("社会的妥当性")
    
    return filtered_score

def separate_text():
    #split source-text(with participant and trainer) into separated files
    
    print('Text : Text separate : Processing')
    
    src_text_dir = text_src_dir
    tgt_text_dir = text_data_dir
    src_file_names = os.listdir(src_text_dir)
    
    for src_file_name in src_file_names:
        
        if 'all' in src_file_name:
            continue
        if 'agent' in src_file_name:
            continue
        if 'user' in src_file_name:
            continue
        
        if DEBUG:
            print('Text file: ', src_file_name)
        whole_text_list = []
        agent_text_list = []
        user_text_list = []
        try:
            with open(os.path.join(src_text_dir, src_file_name), newline='\r\n', encoding='utf-8') as f:
                text = f.read()
                print(type(text))
                #print(text)
        except:
            pass
        try:
            with open(os.path.join(src_text_dir, src_file_name), newline='\r\n') as f:
                text = f.read()
                #print(type(text))
                #print(text)
        except:
            pass        
        for i, line in enumerate(text.split('\n')):
            #line = line.rstrip()
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            if line == '':
                continue
            #print(i, line)
            tag, utterance = line.split(':')
            
            if tag == 'Agent':
                agent_text_list.append(line)
            if tag == 'User':
                user_text_list.append(line)
            whole_text_list.append(line)
            
        whole_text_list = _add_delimiter(whole_text_list)
        agent_text_list = _add_delimiter(agent_text_list)
        user_text_list = _add_delimiter(user_text_list)
        
        if DEBUG:
            print('- All')
            pp.pprint(whole_text_list)
            print('- Agent')
            pp.pprint(agent_text_list)
            print('- User')
            pp.pprint(user_text_list)
        
        base, ext = src_file_name.split('.')
        tgt_file = base + '_all.' + ext
        with open(os.path.join(tgt_text_dir, tgt_file), 'w') as f:
            f.writelines(whole_text_list)
        tgt_file = base + '_agent.' + ext
        with open(os.path.join(tgt_text_dir, tgt_file), 'w') as f:
            f.writelines(agent_text_list)
        tgt_file = base + '_user.' + ext
        with open(os.path.join(tgt_text_dir, tgt_file), 'w') as f:
            f.writelines(user_text_list)
        
        print('Text : Text separate : Done')

def text2csv():
    thread = threading.Thread(target=separate_text)
    return thread
    
def audio2csv():
    
    #mp4, wav to csv with OpenFace, Praat

    extention = 'wav'
    wav_flags = []
    for file_name in os.listdir(audio_src_dir):
        if extention in file_name:
            wav_flags.append(1)
        else:
            wav_flags.append(0)
    if 0 in wav_flags:
        convert_audio()
        
    _ = renamer(audio_src_dir, lookup_list_path=None, output_file_path = lookup_list_path, ext=extention)
    
    thread = threading.Thread(target=run_praat)
        
    return thread
            
def face2csv():        

    thread = threading.Thread(target=audio_face_util.calc_visual, args=(os.getcwd(), face_src_dir))
                
    return thread

def body2csv():
    
    thread = threading.Thread(target=extract_body_feature)
    return thread
    
def extract_body_feature():
    
    #
    # https://github.com/Hzzone/pytorch-openpose
    #
    
    print('Body : OpenPose feature extract : Processing')
    
    s_time = time.time()
    
    body_estimation = Body(os.path.join(openpose_model_dir, 'body_pose_model.pth'))
    hand_estimation = Hand(os.path.join(openpose_model_dir, 'hand_pose_model.pth'))
    
    num_of_gpus = torch.cuda.device_count()
    print('Available GPU:', num_of_gpus)
    
    cuda_or_not = False
    try:
        body_estimation.model.cuda('cuda:0')
        print('Body model on cuda:0')
        cuda_or_not = True
    except:
        pass

    try:
        body_estimation.model.cuda('cuda:0')
        print('Body model on cuda:1')
        cuda_or_not = True
    except:
        pass
    
    if not cuda_or_not:
        print('Body model on cpu')
    
    #print(f"Torch device: {torch.cuda.get_device_name()}")
    
    wholeBody_body_index = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17
        ]
        
    #src_dir = 'src_body'
    src_body_dir = body_src_dir
    tgt_body_dir = body_data_dir
    
    print('RESIZE_FACTOR', RESIZE_FACTOR)
    print('body src', body_src_dir)
    print('body data', body_data_dir)
    
    file_names = os.listdir(src_body_dir)
    for file_name in file_names:
        cap = cv2.VideoCapture(os.path.join(src_body_dir, file_name))
        i = 0
        user_points = []
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Num frames:', num_frames)
        #input('Enter:')
        
        while cap.isOpened():
            
            try:
                #s_time = time.time()
                
                ret, oriImg = cap.read()

                #print(i)
                if i >= num_frames:
                    break
                
                if i%video_frame_skip_factor != 0:
                    i += 1
                    continue
                
                
                h, w, ch = oriImg.shape
                oriImg = cv2.resize(oriImg, (int(w/RESIZE_FACTOR), int(h/RESIZE_FACTOR)))
                
                candidate, subset = body_estimation(oriImg)
                
                target_data = pick_target_data(candidate, subset, wholeBody_body_index)
                
                #[[x0, y0], [x1, y1], ...] -> [x0, y0, x1, y1, ...]
                target_data = np.reshape(target_data, (-1, ), order = 'C')
                user_points.append(target_data)
                
                if DEBUG:
                    # print(oriImg.shape)            
                    # print('Candidate#####')
                    # pp.pprint(candidate)
                    # print('Subset#####')
                    # pp.pprint(subset)
                    print('Target data:', np.shape(target_data))
    
                
                if VIS:
                
                    canvas = copy.deepcopy(oriImg)
                    canvas = openpose_util.draw_bodypose(canvas, candidate, subset)
                
                    # detect hand
                    hands_list = openpose_util.handDetect(candidate, subset, oriImg)
                
                    all_hand_peaks = []
                    for x, y, w, is_left in hands_list:
                        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
                        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                        all_hand_peaks.append(peaks)
                
                    canvas = openpose_util.draw_handpose(canvas, all_hand_peaks)
                    
                
                    cv2.imshow('demo', canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
                #e_time = time.time()    
                
                #if DEBUG:
                    #input('Enter:')
                #    print('{:05}th frame: elapsed - {:.3f}sec'.format(i, e_time - s_time))
            
            except:
                pass
    
            i += 1
            
            # if DEBUG:
            #     if i > 3:
            #         break
                
        cap.release()
        cv2.destroyAllWindows()
        
        base, ext = file_name.split('.')
        tgt_path = os.path.join(tgt_body_dir, base + '.csv')
        _write_csv(tgt_path, user_points)

    print('Body : OpenPose feature extract : Done')
    
    e_time = time.time()
    print('Body : OpenPose feature extract : processing time : {:.3f} sec'.format(e_time - s_time))
        
def pick_target_data(candidate, subset, target_body_index):

    tgt_user_index = 0
    max_detected = 0
    for i, user_point_list in enumerate(subset):
        if user_point_list[-1] > max_detected:
            max_detected = user_point_list[-1]
            tgt_user_index = i
            # if DEBUG:
            #     print('#target id: ', tgt_user_index)
            #     print('#target #point: ', max_detected)
    
    target_data_index = subset[tgt_user_index][target_body_index].astype(np.int64)
    # if DEBUG:
    #     pp.pprint(target_data_index)
    target_data = []
    for data_i in target_data_index:
        
        #if data_i = -1, it means the point was not detected
        if data_i != -1:
            target_data.append(candidate[data_i].tolist())
        else:
            target_data.append([0, 0, -999, -999])
        
    target_data = np.asarray(target_data)
    #target_data = candidate[target_data_index]
    # if DEBUG:
    #     pp.pprint(target_data)
    target_data = target_data[:, 0:2]
    # if DEBUG:
    #     pp.pprint(target_data)
    
    return target_data

def predict(task_name):
    
    features, feature_labels = _feature_loader(feature_file_path)
    
    # 1/3 model save/load. model_names list
    model_save_dir = 'eval_pipeline\\trained_model'
    _check_dir(model_save_dir)
    file_name = 'model_names.sav'
    file_path = os.path.join(model_save_dir, file_name)
    model_names = pkl.load(open(file_path, 'rb'))
    
    # 2/3 model save/load. task specific normalizer
    model_name = _find_model_path(model_names, task_name)
    model_path = os.path.join(model_save_dir, model_name)
    normalizer = pkl.load(open(model_path, 'rb'))
    
    features = normalizer.transform(features)

    # 3/3 model save/load. task-label specific predicter
    label_names = []
    results = {}
    label_name_candidates = [x for x in model_names if 'predicter' in x]
    for candidate in label_name_candidates:
        candidate, ext = candidate.split('.')
        candidate = candidate.split('_')
        candidate = candidate[-1]
        if not(candidate in label_names):
            label_names.append(candidate)
    label_names = sorted(label_names)
    for label_name in label_names:
        model_name = _find_model_path(model_names, task_name, label_name)
        model_path = os.path.join(model_save_dir, model_name)
        model = pkl.load(open(model_path, 'rb'))
        result = model.predict(features)
        results[label_name] = result[0]
        #if DEBUG:
        #    print('{} : {} : {:.3f}'.format(task_name, label_name, result[0]))
    
    return results
        
def renamer(tgt_dir, lookup_list_path=None, ext=None, output_file_path='lookup.csv'):
    
    print('Renaming files in {} : Processing'.format(tgt_dir))
    
    if lookup_list_path == None:
        
        lookup_list = []
        
        input_list = []
        output_list = []
        
        for i,file_name in enumerate(os.listdir(tgt_dir)):
            
            base, ext = file_name.split('.')
            temp = [i+1, base]
            lookup_list.append(temp)
            
            input_list.append(os.path.join(tgt_dir, file_name))
            output_list.append(os.path.join(tgt_dir, str(i+1)+'.{}'.format(ext)))
        
        for i in range(len(input_list)):
            input_path = input_list[i]
            output_path = output_list[i]
            shutil.move(input_path, output_path)
        
        _write_csv(output_file_path, lookup_list)
            
    elif (lookup_list_path != None) and (ext != None):
        
        with open(lookup_list_path, 'r') as f:
            reader = csv.reader(f)
            lookup_list = [x for x in reader]
            
        input_list = []
        output_list = []
        
        for lookup in lookup_list:
            input_list.append(os.path.join(tgt_dir, lookup[1]+'.{}'.format(ext)))
            output_list.append(os.path.join(tgt_dir, lookup[0]+'.{}'.format(ext)))
            
        for i in range(len(input_list)):
            input_path = input_list[i]
            output_path = output_list[i]
            shutil.move(input_path, output_path)
    
            
    elif (lookup_list_path != None) and (ext == None):
        assert False, "'lookup_list' should be passed with 'ext'"
        
    print('Renaming files in {} : Done'.format(tgt_dir))
    
def post_renamer(tgt_dir, lookup_list_path = None, ext = None, mid_audio = False):
    
    print('Re-renaming : Processing')
    
    if (lookup_list_path != None) and (ext != None):
        
        with open(lookup_list_path, 'r') as f:
            reader = csv.reader(f)
            lookup_list = [x for x in reader if x!=[]]
            
        input_list = []
        output_list = []
        
        for lookup in lookup_list:
            print(lookup)
            if mid_audio:
                for data_type in ['formant', 'intensity', 'pitch']:
                    input_list.append(os.path.join(tgt_dir, ' '+lookup[0]+'_{}.{}'.format(data_type, ext)))
                    output_list.append(os.path.join(tgt_dir, lookup[1]+'_{}.{}'.format(data_type, ext)))
            else:
                input_list.append(os.path.join(tgt_dir, lookup[0]+'.{}'.format(ext)))
                output_list.append(os.path.join(tgt_dir, lookup[1]+'.{}'.format(ext)))
            #shutil.move(input_path, output_path)
        
        skipped = []
        for i in range(len(input_list)):
            try:
                input_path = input_list[i]
                output_path = output_list[i]
                print(input_path)
                print(output_path)
                shutil.move(input_path, output_path)
            except Exception as e:
                print(e)
                input()
                skipped.append([input_path, output_path])
        print('Skipped files : {}'.format(len(skipped)))
            
    else:
        print('Error: lookup_list_path and ext should be specified.')
            
    print('Re-renaming : Done')

def convert_audio(tgt_dir = 'src_audio'):
    
    print('Audio conversion : Processsing')
    
    tgt_file_names = os.listdir(tgt_dir)
        
    for i, src_file_name in enumerate(tgt_file_names):
        
        if not ('mp3' in src_file_name):
            continue
        
        try:
            
            base, ext = src_file_name.split('.')
            input_path = os.path.join(tgt_dir, src_file_name)
            input_path = os.path.abspath(input_path)
            #input_path = input_path.replace('\\','/')
            output_path = os.path.join(tgt_dir, base+'.wav')
            print('--- {:> 5.1f}%, Audio converting : {}'.format((i+1)/len(tgt_file_names)*100, 
                                                           input_path))
            tgt_audio = AudioSegment.from_file(input_path,
                                               format=ext)
            tgt_audio.export(output_path, format='wav')
            if input_path != output_path:
                os.remove(input_path)
        
        except Exception as e:
                        
            print(e)
        
        
    print('Audio conversion : Done')


def run_praat():
    
    praat_file_path = os.path.join(praat_dir, praat_file)
    #praat_output_path = os.path.join(praat_dir, praat_output_dir)

    #if praat_output_dir in os.listdir(praat_dir):
    if os.path.exists(praat_output_dir):
        shutil.rmtree(praat_output_dir)
        os.mkdir(praat_output_dir)
    else:
        os.mkdir(praat_output_dir)
        
    tgt_code = "input_directory$ = \"{}\"\n".format(audio_src_dir_from_praat_dir)
    _replace_path_in_code(praat_file_path, praat_file_path, 'MARK_INPUT', tgt_code)
    
    tgt_code = "output_directory$ = \"{}\"\n".format(praat_output_dir_from_praat_dir)
    _replace_path_in_code(praat_file_path, praat_file_path, 'MARK_OUTPUT', tgt_code)
    
    print('Audio : Praat feature extract : Processing')
    if pf == 'Windows':
        res = subprocess.call("cd {} && \
                               Praat.exe --run audio_2_extractor.praat && \
                               cd ..".format(praat_dir), 
                               shell=True)
    elif pf == 'Darwin':
        res = subprocess.call("cd {} && \
                               /Applications/Praat.app/Contents/MacOS/Praat --run audio_2_extractor.praat && \
                               cd ..".format(praat_dir), 
                               shell=True)
    elif pf == 'Linux':
        res = subprocess.call("cd {} && \
                               /usr/bin/praat --run audio_2_extractor.praat && \
                               cd ..".format(praat_dir), 
                               shell=True)
    if DEBUG:
        pp.pprint(res)

    print('Audio : Praat feature extract : Done')
    
    post_renamer(audio_src_dir, lookup_list_path = lookup_list_path, ext = 'wav', mid_audio = False)
    post_renamer(praat_output_dir, lookup_list_path = lookup_list_path, ext = 'csv', mid_audio = True)
    
    print('Audio : Finalize (Copy csv) : Processing')
    file_names = os.listdir(praat_output_dir)
    file_names = [x for x in file_names if '.csv' in x]
    for file_name in file_names:
        shutil.copyfile(os.path.join(praat_output_dir, file_name),
                    os.path.join(audio_data_dir, file_name))
    print('Audio : Finalize (Copy csv) : Done')
    
def calc_text(text_main_IDs, text_main_subjects, text_all_IDs, text_all_subjects, face_main_IDs, face_main_subjects):
    
    successed_IDs = []
    text_features = []
        
    for i, (text_main_ID, text_main_subject) in enumerate(zip(text_main_IDs, text_main_subjects)):
        
        text_all_ID, text_all_subject, success = _pick_correspond_subject(text_main_ID, 
                                                                text_all_IDs, text_all_subjects,
                                                                main_keyword, all_keyword)
        if not success:
            print('pick correspond error text 1')
            continue

        face_main_ID, face_main_subject, success = _pick_correspond_subject(text_main_ID, 
                                                                face_main_IDs, face_main_subjects,
                                                                main_keyword, main_keyword)
        if not success:
            print('pick correspond error text 2')
            continue
        
        text_feature_names = []
                    
        print('text : {}/{}'.format(i+1, len(text_main_IDs)))
        row = [text_main_ID]
        
        feature = func_text.BERT_sentence(text_main_subject)
        row.extend(feature)
        text_feature_names.append('BERT_self_self_sent')
        if DEBUG:
            print(feature)
        
        feature = func_text.BERT_cont_word(text_main_subject)
        row.extend(feature)
        text_feature_names.append('BERT_self_self_cont')
        if DEBUG:
            print(feature)

        feature = func_text.BERT_sentence(text_all_subject)
        row.extend(feature)
        text_feature_names.append('BERT_self_inter_sent')
        if DEBUG:
            print(feature)
        
        feature = func_text.BERT_cont_word(text_all_subject)
        row.extend(feature)
        text_feature_names.append('BERT_self_inter_cont')
        if DEBUG:
            print(feature)
            
        feature = func_text.count_content_words(text_main_subject)
        row.extend(feature)
        text_feature_names.append('num_content')
        if DEBUG:
            print(feature)
        
        feature = func_text.check_thanks(text_main_subject)
        row.extend(feature)
        text_feature_names.append('thanks_flag')
        if DEBUG:
            print(feature)
        
        feature = func_text.check_seems_sorry(text_main_subject)
        row.extend(feature)
        text_feature_names.append('sorry_flag')
        if DEBUG:
            print(feature)

        feature = func_text.check_explicit_refuse(text_main_subject)
        row.extend(feature)
        text_feature_names.append('explicit_refuse_flag')
        if DEBUG:
            print(feature)

        feature = func_text.count_backchannels(text_main_subject)
        row.extend(feature)
        text_feature_names.append('num_backchannel')
        if DEBUG:
            print(feature)
            
        feature = func_text.check_initial_que(text_main_subject)
        row.extend(feature)
        text_feature_names.append('init_que_flag')
        if DEBUG:
            print(feature)

        feature = func_text.calc_WPM(text_main_subject, face_main_subject)
        row.extend(feature)
        text_feature_names.append('WPM')
        if DEBUG:
            print(feature)
        
        successed_IDs.append(text_main_ID)
        text_features.append(row)
    text_main_IDs = successed_IDs
    
    if DEBUG:
        print(text_features)
    
    return [text_main_IDs, text_features, text_feature_names]
        

def calc_audio(audio_IDs, audio_int_subjects, audio_f0_subjects):
    
    successed_IDs = []
    audio_features = []
    for i, (audio_ID, audio_int_subject, audio_f0_subject) in enumerate(zip(audio_IDs, audio_int_subjects, audio_f0_subjects)):

        audio_feature_names = []
        
        if SHORT_INSPECT:
            if i > SHORT_THRESHOLD:
                break

        print('audio : {}/{}'.format(i+1, len(audio_int_subjects)))
        row = [audio_ID]
        
        feature = func_audio.calc_ave_voice_int(audio_int_subject)
        audio_feature_names.append('ave_voice_int')
        row.extend(feature)
        if DEBUG:
            print(feature)

        feature = func_audio.calc_cv_voice_f0(audio_f0_subject)
        audio_feature_names.append('cv_voice_f0')
        row.extend(feature)
        if DEBUG:
            print(feature)

        successed_IDs.append(audio_ID)
        audio_features.append(row)
    audio_IDs = successed_IDs
    
    if DEBUG:
        print(audio_features)
    
    return [audio_IDs, audio_features, audio_feature_names]

def calc_face(face_main_IDs, face_main_subjects, face_inter_IDs, face_inter_subjects):
    
    successed_IDs = []
    face_features = []    
    for i, (face_main_ID, face_main_subject) in enumerate(zip(face_main_IDs, face_main_subjects)):
        
        if NO_MUTUAL:
            pass
        else:
            face_inter_ID, face_inter_subject, success = _pick_correspond_subject(face_main_ID, 
                                                                    face_inter_IDs, face_inter_subjects,
                                                                    main_keyword, inter_keyword)

            if not success:
                print('pick correspond error face 1')
                continue
        
        face_feature_names = []
        
        if SHORT_INSPECT:
            if i > SHORT_THRESHOLD:
                break
            
        print('face : {}/{}'.format(i+1, len(face_main_subjects)))
        row = [face_main_ID]
        
        feature = func_face.calc_smile_freq(face_main_subject)
        row.extend(feature)
        face_feature_names.append('smile_freq')
        if DEBUG:
            print(feature)
        
        feature = func_face.calc_headpose(face_main_subject)
        row.extend(feature)
        face_feature_names.append('head_mean')        
        face_feature_names.append('head_cv')        
        if DEBUG:
            print(feature)
        
        #feature = count_nods(face_main_subject)
        feature = func_face.count_nods_kawato(face_main_subject)
        row.extend(feature)
        face_feature_names.append('num_nod')
        if DEBUG:
            print(feature)

        #feature = func_face.calc_mutual_smile(face_main_subject, face_inter_subject)
        feature = [0] #TODO! dammy data
        row.extend(feature)
        face_feature_names.append('mutual_smile')
        if DEBUG:
            print(feature)

        feature, names = func_face.calc_AU_stats(face_main_subject)
        row.extend(feature)
        face_feature_names.extend(names)
        if DEBUG:
            print(feature)
        
        successed_IDs.append(face_main_ID)
        face_features.append(row)
    face_main_IDs = successed_IDs
    
    if DEBUG:
        print(face_features)
        
    return [face_main_IDs, face_features, face_feature_names]

def calc_body(body_IDs, body_subjects):
    
    successed_IDs = []
    body_features = []
    for i, (body_ID, body_subject) in enumerate(zip(body_IDs, body_subjects)):
        
        body_feature_names = []
        
        if SHORT_INSPECT:
            if i > SHORT_THRESHOLD:
                break
        
        print('body : {}/{}'.format(i+1, len(body_subjects)))
        row = [body_ID]
        
        feature = func_body.calc_gesture_cv(body_subject)
        row.extend(feature)
        body_feature_names.append('gesture_all_cv')
        body_feature_names.append('gesture_upper_cv')
        body_feature_names.append('gesture_arm_cv')
        if DEBUG:
            print(feature)
            
        successed_IDs.append(body_ID)
        body_features.append(row)
        body_IDs = successed_IDs

    return [body_IDs, body_features, body_feature_names]

def generate_feedbacks(result_dict, task_name, num_positive = 1, num_negative = 1, feedback_csv_path = 'eval_pipeline/2_feedback_sentence.csv'):

    # Dummy function
    # feedbacks = {'PositiveComment':'ぽじてぃぶふぃーどばっく', 'NegativeComment':'ねがてぃぶふぃーどばっく'}
    
    try:
        feedback_list = _load_csv(feedback_csv_path)
    except:
        pass
    try:
        feedback_list = _load_csv(feedback_csv_path, encoding='shift-jis')
    except:
        pass    
    
    
    feedback_list = feedback_list[1:]
    
    feedback_dict = {}
    for row in feedback_list:
        feedback_dict[row[0]] = {}
    for row in feedback_list:
        feedback_dict[row[0]][row[1]] = [row[2], row[3]]
    
    #pp.pprint(feedback_dict)
    
    if STRATEGY_BEST_WORST:
        pos_sent, neg_sent = strategy_best_worst(result_dict, feedback_dict, task_name)
    if STRATEGY_NM:
        pos_sent, neg_sent = strategy_good_n_bad_m(result_dict, feedback_dict, task_name, num_positive, num_negative)

    feedbacks = {'PositiveComment':pos_sent, 'NegativeComment':neg_sent}
    
    return feedbacks
    
def strategy_best_worst(result_dict, feedback_dict, task_name):
    
    min_pair = ['', 999]
    max_pair = ['', -999]
    
    for component in result_dict.keys():
        
        tmp_score = result_dict[component]
        
        if tmp_score < min_pair[1]:
            min_pair = [component, tmp_score]
        
        if max_pair[1] < tmp_score:
            max_pair = [component, tmp_score]
    
    pos_sent = _generate_sentence([max_pair], feedback_dict, task_name, positive=True)
    neg_sent = _generate_sentence([min_pair], feedback_dict, task_name, positive=False)
    
    return pos_sent, neg_sent

def strategy_good_n_bad_m(result_dict, feedback_dict, task_name, n, m):
    
    component_list= []
    score_list = []
    for component in result_dict.keys():
        component_list.append([component, result_dict[component]])
        score_list.append(result_dict[component])
    component_sorted_list = sorted(component_list, key = lambda x:x[1], reverse = True)
    
    print(component_sorted_list)
    
    average = np.average(score_list)
    minimum = min(score_list)
    maximum = max(score_list)
    
    pos_pair_list = []
    neg_pair_list = []
    for i in range(n):
        pos_pair_list.append(component_sorted_list[0])
        component_sorted_list.pop(0)
    for i in range(m):
        neg_pair_list.append(component_sorted_list[-1])
        component_sorted_list.pop(-1)
    
    if minimum > FB_score_thread:
        pos_sent = '全ての項目で完璧に近いロールプレイでした！'
        neg_sent = '修正すべき点は見つかりませんでした。これからも良いコミュニケーションを続けてくださいね！'
    else:
        pos_sent = _generate_sentence(pos_pair_list, feedback_dict, task_name, positive=True)
        neg_sent = _generate_sentence(neg_pair_list, feedback_dict, task_name, positive=False)
    
    return pos_sent, neg_sent

def print_all_feedbacks(feedback_dict):
    
    task_component_pair_list = []
    for task_name in ['TELL', 'LISTEN', 'REFUSE', 'FAVOR']:
        for component in feedback_dict[task_name].keys():
            task_component_pair_list.append([task_name, component])
    
    cnt = 1
    for pair in task_component_pair_list:
        print('#############################################')
        sent = _generate_sentence([[pair[1], 0.05]], feedback_dict, pair[0], positive=True)
        print('-No.{:03d} - {}:{}#####################################'.format(cnt, pair[0], pair[1]))
        print(sent)
        cnt += 1
        sent = _generate_sentence([[pair[1], 0.05]], feedback_dict, pair[0], positive=False)
        print('-No.{:03d} - {}:{}#####################################'.format(cnt, pair[0], pair[1]))
        print(sent)
        cnt +=1 
        print('#############################################')

def rescale(predicted_score):
    
    #print(predicted_score)
    
    original_range = original_scale_range[1] - original_scale_range[0]
    target_range = target_scale_range[1] - target_scale_range[0]
    
    #print(original_range)
    #print(target_range)
    
    print("Rescale score range from", original_scale_range, "to", target_scale_range)
    
    for key in predicted_score.keys():
        original_score = predicted_score[key]
        
        score = original_score - original_scale_range[0]
        score = score * (target_range / original_range)
        score = score + target_scale_range[0]
        
        if score < target_scale_range[0]:
            score = target_scale_range[0]
        
        predicted_score[key] = score
        
    return predicted_score

def _generate_sentence(pairs, feedback_dict, task_name, positive=True):
    
    num_component = len(pairs)
    sent = ''
    
    if positive:

        if num_component == 0:
            sent = ''
        elif num_component == 1:
            #print(pairs[0])
            #print(feedback_dict[task_name][pairs[0][0]][0])
            #sent = '{0:}がよかったと思います。{0:}をうまく扱うことができると、{1:}ので良いコミュニケーションができそうです！'.format(pairs[0][0], feedback_dict[task_name][pairs[0][0]][0])
            sent = '{0:}がよかったと思います。{1:}ので良いコミュニケーションができそうです！'.format(pairs[0][0], feedback_dict[task_name][pairs[0][0]][0])
        elif num_component == 2:
            sent = '{0:}と{1:}がよかったと思います。'.format(pairs[0][0], pairs[1][0])
            sent += '{0:}をうまく扱うことができると、{1:}と思いますし、'.format(pairs[0][0], feedback_dict[task_name][pairs[0][0]][0])
            sent += '{0:}をうまく使えると、{1:}ので、良いコミュニケーションができそうです！'.format(pairs[1][0], feedback_dict[task_name][pairs[1][0]][0])
        else:
            sent = 'すみません、エンジニアさん。フィードバック項目は0個か1個か2個にしてもらえますか？それ以外は私にはちょっと・・・ごめんなさい'
    
    else:
        
        if num_component == 0:
            sent = ''
        elif num_component == 1:
            sent = '{0:}についてはもう少し意識できると、よりよいコミュニケーションがとれそうです。'.format(pairs[0][0])
            # sent += '{0:}をうまく扱えると、{1:}ので、お互いに気持ちのいいコミュニケーションの実現が期待できます。'.format(pairs[0][0], feedback_dict[task_name][pairs[0][0]][0])
            # sent += 'もし{0:}の改善が難しいようであれば、{1:}と、いいかもしれません！'.format(pairs[0][0], feedback_dict[task_name][pairs[0][0]][1])
            sent += '{0:}と、いいかもしれません。頑張ってみてください！'.format(feedback_dict[task_name][pairs[0][0]][1])
        elif num_component == 2:
            sent = '{0:}と{1:}についてはもう少し意識できると、よりよいコミュニケーションがとれそうですね。'.format(pairs[0][0], pairs[1][0])
            sent += '{0:}をうまく扱えると、{1:}ので、お互いに気持ちのいいコミュニケーションの実現が期待できます。'.format(pairs[0][0], feedback_dict[task_name][pairs[0][0]][0])
            sent += 'また、{0:}をうまく扱えると、{1:}ので、お互いに気持ちのいいコミュニケーションの実現が期待できます。'.format(pairs[1][0], feedback_dict[task_name][pairs[1][0]][0])
            sent += 'もし{0:}の改善が難しいようであれば、{1:}と、いいかもしれません！'.format(pairs[0][0], feedback_dict[task_name][pairs[0][0]][1])
            sent += '{0:}については{1:}と改善できそうです！'.format(pairs[1][0], feedback_dict[task_name][pairs[1][0]][1])
        else:
            sent = 'すみません、エンジニアさん。フィードバック項目は0個か1個か2個にしてもらえますか？それ以外は私にはちょっと・・・ごめんなさい'


    return sent


def _feature_loader(csv_file_name):
    features = pd.read_csv(csv_file_name)
    feature_label = features.columns[1:]
    features = features.drop("ID", axis=1).values.tolist()
    return features, feature_label

def _add_delimiter(src):
    tgt = []
    for line in src:
        line = line + '\n'
        tgt.append(line)
    return tgt

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

def _write_csv(file_path, data):
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)

def _load_csv(input_file, delimiter = ',', encoding = 'utf-8', cast = None):
    
    with open(input_file, encoding = encoding) as f:
        reader = csv.reader(f, delimiter = delimiter)
        if cast != None:
            data = []
            for row in reader:
                temp = []
                for x in row:
                    temp.append(cast(x))
                data.append(temp)
        else:
            data = [x for x in reader]
    
    return data

def _load_csv_multiple(input_dir, delimiter = ',', keyword = None, encoding = 'utf-8', cast = None):
    
    input_files = os.listdir(input_dir)
    IDs = []
    data = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):
                        
        if keyword != None:
            if not (keyword in input_file):
                continue

        #if DEBUG or LOCAL_DEBUG:
        #    print(input_file)
        
        ID, ext = input_file.split('.')
        IDs.append(ID)

        path = os.path.join(input_dir, input_file)
        if cast != None:
            data.append(_load_csv(path, delimiter = delimiter, encoding = encoding, cast = cast))
        else:
            data.append(_load_csv(path, delimiter = delimiter, encoding = encoding))
                
        if SHORT_INSPECT:
            if len(IDs) > SHORT_THRESHOLD:
                break        
        
    return [IDs, data]

def _load_text_data(input_dir, keyword = None):
    
    print('Text : Load data : Processing')
    
    output = []
    
    input_files = os.listdir(input_dir)
    IDs = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):
        
        if keyword != None:
            if not (keyword in input_file):
                continue
        
        with open(os.path.join(input_dir, input_file)) as f:
            output_individual = []
            for line in f.readlines():
                line = line.replace('\n', '')
                output_individual.append(line)
        
        ID, ext = input_file.split('.')
        IDs.append(ID)

        output.append(output_individual)        

        if SHORT_INSPECT:
            if len(IDs) > SHORT_THRESHOLD:
                break        

    print('Text : Load data : Done')
    
    return [IDs, output]
            

def _load_audio_int_data(input_dir, keyword = None):

    print('Audio : Load data : Processing')
    
    #with open(os.path.join(feature_dir, ' '+base_name+'_intensity.csv'), 'r') as f:
    #    reader = csv.reader(f, delimiter=' ')
    
    input_files = os.listdir(input_dir)
    IDs = []
    output_list = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):
        
        if not('intensity' in input_file):
            continue
        #print(input_file)
        
        if keyword != None:
            if not (keyword in input_file):
                continue

        base, ext = input_file.split('.')
        parts = base.split('_')
        ID = '_'.join(parts[:-1])
        IDs.append(ID)
        
        data = _load_csv(os.path.join(input_dir, input_file))
    
        temp0 = []
        for y in data:
            temp1 = []
            for z in y:
                if not z=='':
                    temp1.append(z.split())
                    temp0.append(temp1)
        #pp.pprint(temp0[0:50])
        
        individual_output = []
        for i in range(len(temp0)):
            if 'z' in temp0[i][0]:
                if len(temp0[i][0])==5:
                    #print(temp0[i][0])
                    frame_num = temp0[i][0][2]
                    frame_num = frame_num[1:-2]
                    
                    #To avoid exception
                    if frame_num == '':
                        continue
                    
                    individual_output.append([int(frame_num),float(temp0[i][0][4])])
        
        output_list.append(individual_output)
        
        
        if SHORT_INSPECT:
            if len(IDs) > SHORT_THRESHOLD:
                break
    
    
    #data = _load_csv_multiple(input_dir, keyword = 'intensity')

    print('Audio : Load data : Done')

    return [IDs, output_list]

def _load_audio_f0_data(input_dir, keyword = None):

    print('Audio : Load data : Processing')
    
    #with open(os.path.join(feature_dir, ' '+base_name+'_intensity.csv'), 'r') as f:
    #    reader = csv.reader(f, delimiter=' ')
    
    input_files = os.listdir(input_dir)
    IDs = []
    output_list = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):

        #print(input_file)
        
        if not('pitch' in input_file):
            continue
        
        if keyword != None:
            if not (keyword in input_file):
                continue

        base, ext = input_file.split('.')
        parts = base.split('_')
        ID = '_'.join(parts[:-1])
        IDs.append(ID)
        
        data = _load_csv(os.path.join(input_dir, input_file))
    
        data = data[11:]
        frames = []
        frame = []
        for i in range(len(data)):
            #print(data[i])
            if 'frames [' in data[i][0]:
                frames.append(frame)
                frame = [data[i][0].replace(' ', '')]
            else:
                frame.append(data[i][0].replace(' ', ''))
        frames.pop(0)
        #pp.pprint(frames[:3])
        
        frame_candidates = []
        for frame_src in frames:
            frame_tgt = []
            for cand_index in range(4, len(frame_src), 3):
                #print(frame[cand_index])
                freq = frame_src[cand_index+1]
                freq = freq.replace('frequency=','')
                strength = frame_src[cand_index+2]
                strength = strength.replace('strength=','')
                frame_tgt.append([float(freq), float(strength)])
            frame_candidates.append(frame_tgt)
            
        #pp.pprint(frame_candidates[:3])

        individual_output = []
        for i, candidates in enumerate(frame_candidates):
            max_index = np.argmax(candidates, axis=0)
            #print(max_index)
            selected = candidates[max_index[1]]
            #print(selected)
            if selected[0] == 0.0:
                continue
            individual_output.append([i+1, selected[0]])
                        
        #pp.pprint(individual_output)
        
        output_list.append(individual_output)
                
        if SHORT_INSPECT:
            if len(IDs) > SHORT_THRESHOLD:
                break
    
    #data = _load_csv_multiple(input_dir, keyword = 'intensity')

    print('Audio : Load data : Done')

    return [IDs, output_list]


def _load_face_data(input_dir, keyword = None):

    print('Face : Load data : Processing')
    IDs, data = _load_csv_multiple(input_dir, keyword = keyword)
    print('Face : Load data : Done')

    return [IDs, data]

def _load_body_data(input_dir, keyword = None):

    print('Body : Load data : Processing')
    IDs, data = _load_csv_multiple(input_dir, keyword = keyword, cast = float)
    print('Body : Load data : Done')

    return [IDs, data]

def _filter_by_ID(ID_list, features):

    output_list = []
    for tgt_ID in ID_list:
        tgt_idx = -1
        for i in range(len(features)):
            if features[i][0] == tgt_ID:
                tgt_idx = i
                if DEBUG:
                    print(tgt_ID)
                    print(features[i][0])
        if tgt_idx != -1:
            output_list.append(features[tgt_idx])
    
    return output_list

def _pick_correspond_subject(main_ID, inter_IDs, inter_subjects, main_keyword, inter_keyword):
    
    if DEBUG:
        print(main_ID)
    
    tgt_ID = None
    tgt_subject = None
    success = False
    
    key_ID = main_ID.replace(main_keyword, inter_keyword)
    
    for inter_ID, inter_subject in zip(inter_IDs, inter_subjects):
        if DEBUG:
            print(main_ID, inter_ID)
        if inter_ID == key_ID:
            tgt_ID = inter_ID
            tgt_subject = inter_subject
            success = True
            break
    
    return [tgt_ID, tgt_subject, success]

def _check_dir(dir_path):
    
    if not(os.path.exists(dir_path)):
        os.mkdir(dir_path)

def _find_model_path(model_names, task_name, label_name=None):
    
    if label_name == None:
        for model_name in model_names:
            if ('normalizer' in model_name) and (task_name in model_name):
                return model_name
    else:
        for model_name in model_names:
            if ('predicter' in model_name) and (task_name in model_name) and (label_name in model_name):
                return model_name
    
    print('Invalid argument: _find_model_path(model_names, task_name, label_name=None)')
    sys.exit()


def _write_json(data, name, encoding='shift-jis'):
    
    with open(name, 'w', encoding = encoding) as f:
        text = json.dumps(data, ensure_ascii = False, indent=2)
        f.write(text)
        
def _make_dirs(dir_list):
    
    for target_dir in dir_list:
        os.mkdir(target_dir)
        
def _copyfile(src, tgt):
    print('Copy {} : {} ...'.format(src, tgt))
    shutil.copyfile(src, tgt)
    print('Copy {} : {} ... Done'.format(src, tgt))
    
def _copytree(src, tgt):
    print('Copy {} : {} ...'.format(src, tgt))
    shutil.copytree(src, tgt)
    print('Copy {} : {} ... Done'.format(src, tgt))

def _record_data(user_ID = None):
    
    global text_record_dir, audio_record_dir, face_record_dir, body_record_dir
    
    #text_record_dir = os.path.join(base_record_dir, text_record_dir)
    #audio_record_dir = os.path.join(base_record_dir, audio_record_dir)
    #face_record_dir = os.path.join(base_record_dir, face_record_dir)
    #body_record_dir = os.path.join(base_record_dir, body_record_dir)

    if user_ID == None:
        user_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    text_src_path = os.path.join(text_src_dir, 'temp.txt')
    audio_src_path = os.path.join(audio_src_dir, 'temp_user.wav')
    video_src_path = 'temp.mpg'
    json_src_path = json_path
    feature_src_path = feature_file_path
    data_src_dir = 'data'
    
    text_tgt_path = os.path.join(base_record_dir, user_ID + '.txt')
    audio_tgt_path = os.path.join(base_record_dir, user_ID + '.wav')
    video_tgt_path = os.path.join(base_record_dir, user_ID + '.mpg')
    json_tgt_path = os.path.join(base_record_dir, user_ID + '.json')
    feature_tgt_path = os.path.join(base_record_dir, user_ID + '.csv')
    data_tgt_dir = os.path.join(base_record_dir, user_ID)
    
    _copyfile(text_src_path, text_tgt_path)
    _copyfile(audio_src_path, audio_tgt_path)
    _copyfile(video_src_path, video_tgt_path)
    _copyfile(json_src_path, json_tgt_path)
    _copyfile(feature_src_path, feature_tgt_path)
    _copytree(data_src_dir, data_tgt_dir)
    
    
    #for viewer
    
    if base_result_dir in os.listdir('./'):
        shutil.rmtree(base_result_dir)
        os.mkdir(base_result_dir)
    else:
        os.mkdir(base_result_dir)

    text_tgt_path = os.path.join(base_result_dir, 'result.txt')
    audio_tgt_path = os.path.join(base_result_dir, 'result.wav')
    video_tgt_path = os.path.join(base_result_dir, 'result.mpg')
    json_tgt_path = os.path.join(base_result_dir, 'result.json')
    feature_tgt_path = os.path.join(base_result_dir, 'result.csv')
    
    _copyfile(text_src_path, text_tgt_path)
    _copyfile(audio_src_path, audio_tgt_path)
    _copyfile(video_src_path, video_tgt_path)
    _copyfile(json_src_path, json_tgt_path)
    _copyfile(feature_src_path, feature_tgt_path)
    
def _add_name_tag(tgt_dir, tag):
    
    for name in os.listdir(tgt_dir):
        src_path = os.path.join(tgt_dir, name)
        base, ext = name.split('.')
        name = '{}_{}.{}'.format(base, tag, ext)
        tgt_path = os.path.join(tgt_dir, name)
        shutil.move(src_path, tgt_path)

def _remove_name_tag(tgt_dir, tag):
    
    for name in os.listdir(tgt_dir):
        src_path = os.path.join(tgt_dir, name)
        base, ext = name.split('.')
        base = base.strip('_{}'.format(tag))
        name = '{}.{}'.format(base, ext)
        tgt_path = os.path.join(tgt_dir, name)
        shutil.move(src_path, tgt_path)
        

def _replace_path_in_code(src_name, tgt_name, key, content):
        
    data = []
    with open(src_name) as f:
        for line in f.readlines():
            data.append(line)
    
    for i in range(len(data)):
        
        if key in data[i]:
            
            
            before = data[i+1]
            
            data[i+1] = content
            
            after = content
            
            if DEBUG:
                print('Before: ', before)
                print('After : ', after)
                    
    with open(tgt_name, 'w') as f:
        for line in data:
            f.write(line)
            
def _prepare_dirs():
    
    necessary_dirs = [
        'data',
        'data\\data_audio',
        'data\\data_body',
        'data\\data_face',
        'data\\data_text',
        'record',
        'src_audio',
        'src_text',
        'src_video',
        'eval_pipeline\\trained_model'
        ]
    
    for necessary_dir in necessary_dirs:
        if not(os.path.exists(necessary_dir)):
            os.mkdir(necessary_dir)
    
    if len(os.listdir('eval_pipeline\\trained_model')) == 0:
        print('Please check whether you have placed pretrained score predictors properly.')
        sys.exit()
    
    reset_util.reset_data_dir()

if __name__ == '__main__':

    if len(sys.argv) == 4:
        task_name = sys.argv[1]
        num_positive = int(sys.argv[2])
        num_negative = int(sys.argv[3])
        print("positive:{}, negative:{}".format(num_positive, num_negative))
        pipeline(task_name, num_positive, num_negative)
    elif len(sys.argv) == 2:
        task_name = sys.argv[1]
        print(task_name)
        pipeline(task_name)
    else:
        pipeline()
    
