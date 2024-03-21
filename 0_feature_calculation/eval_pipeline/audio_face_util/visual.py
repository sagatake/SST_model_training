#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:10:41 2020
Updated on Apr 25 14:00 2021
Updated on 2022年  2月 17日 木曜日 20:42:00 JST

@author: takeshi-s
"""

import multiprocessing

import pandas as pd
import numpy as np
import subprocess
import shutil
import docker
import pprint
import cv2
import csv
import os

import time

DEBUG = False
user_name = 'TAPAS'

#input_dir = 'src_visual'
#input_dir = 'C:/Users/AHC-Lab/Desktop/tmp/20210425_saga/src_visual'
#input_dir = 'C:/Users/AHC-Lab/Desktop/tmp/20210425_saga/test
#input_dir = 'C:/Users/AHC-Lab/Desktop/tmp/20210425_saga/被験者']
#input_dir = 'C:/Users/AHC-Lab/Desktop/tmp/20210425_saga/20210503_ICMI_1'
input_dir = 'C:/Users/AHC-Lab/Desktop/tmp/20210527_saga/video_processed'

#output_dir = 'tmp_cropped'
#output_dir = 'src_visual'
#output_dir = 'src_video'
#output_dir = 'C:/Users/AHC-Lab/Desktop/tmp/20210527_saga/video_processed'
#output_dir = input_dir

# need to use "/" since this is used in openpose processing running on Ubuntu through docker
openface_output_dir = 'eval_pipeline/audio_face_util/processed'

face_data_dir = 'data\\data_face'

def calc_visual(base_dir_path = None, output_dir = 'src_video'):
    
    if base_dir_path==None:
        base_dir_path = os.getcwd()
    

    """
    if output_dir in os.listdir('./'):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    else:
        os.mkdir(output_dir)
    """    
    openface_output_dir = 'processed'
    """
    if openface_output_dir in os.listdir('./'):
        shutil.rmtree(openface_output_dir)
        os.mkdir(openface_output_dir)
    else:
        os.mkdir(openface_output_dir)
    """
    
    #s_sec = 10
    s_sec = None
    #e_sec = 20
    e_sec = None
    
    x0 = 0
    x1 = 500
    y0 = 0
    y1 = 600
    #crop_size = [x0,x1,y0,y1]
    #crop_size = [0, 1080, 0, 1920]
    crop_size = None
    
    skip_size = 1
    
    #video_crop(input_dir, output_dir, s_sec, e_sec, crop_size, skip_size)
    process_openface(base_dir_path, output_dir)
    #calc_AU(openface_output_dir)


def video_crop(input_dir, output_dir, s_sec=None, e_sec=None, crop_size=None, skip_size=1):
    for file_name in sorted(os.listdir(input_dir)):
        
        print('Visual : Cropping : Processing')
        
        file_path = input_dir + '/' + file_name
        cap = cv2.VideoCapture(file_path)
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if crop_size == None:
            W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            crop_size = [0, int(H), 0, int(W)]
            
        if s_sec == None:
            s_sec = 0
        if e_sec == None:
            e_sec = total_frames / fps
        
        s_index = int(s_sec * fps)
        e_index = int(e_sec * fps)
        if DEBUG:
            print('original_fps:{}, s_index:{}, e_index:{}'.format(fps, s_index, e_index))
    
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #fourcc = cv2.VideoWriter_fourcc(*'H264')
        #fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
                
        name, ext = file_name.split('.')
        #file_name = name + '_cropped.' + ext
        output_path = output_dir + '/' +file_name
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_size[3], crop_size[1]))

        print('in : {} \nout : {}'.format(file_path, output_path))

        #pprint.pprint(locals())
        
        i = 0
        #frames = []
    
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if ret == True:
            
                #Frame skipping
                if i % skip_size == 0:
                
                    if ((s_index < i) or (i < e_index)):
                        frame = frame[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3]]
                        out.write(frame)
                        #frames.append(frame)
                    
                    if DEBUG:
                        print('\r{:05d}/{:05d} - {:3d}%'.format(i,int(total_frames), int((i/total_frames)*100)), end='')
                    
                else:
                    pass
                i += 1
            else:
                break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print('Visual : Cropping : Done')

def process_openface(base_dir_path, input_dir):
    cli = docker.DockerClient()
    #cli = docker.Client()
    image_list = cli.images.list()
    if DEBUG:
        pprint.pprint(image_list)
    
    print('Face : Extracting features : Processing')
    
    # Delete old container if it's runninng
    try:
        container = cli.containers.get('openface')
        container.stop()
        container.remove()
    except:
        pass
    
    target_tag = 'algebr/openface:latest'
    for x in image_list:
        #print(x.tags)
        if len(x.tags)==0:
            pass
        elif target_tag == x.tags[0]:
            image = x
            tag = x.tags[0]
            container_id = x.id
    
    if DEBUG:
        print('tag : ', tag)
        print('id  : ', container_id)
    
    cwd = os.getcwd()
    
    # Sorting file names
    print(cwd)
    path = os.path.join(base_dir_path, input_dir)
    print(path)
    file_names = os.listdir(path)
    #file_names = os.listdir(input_dir)
    tmp_list = []
    for x in file_names:
        tmp_list.append(x.split("."))
    try:
        tmp_list = sorted(tmp_list, key=lambda x:int(x[0]))
    except:
        tmp_list = sorted(tmp_list)
    print(tmp_list)
    for i in range(len(tmp_list)):
        tmp_list[i] = tmp_list[i][0] + "." + tmp_list[i][1]
    file_names = tmp_list
    
    total_size = len(file_names)
    
    for i, file_name in enumerate(file_names):
        
        s_time = time.time()

        #file_path = os.path.join("src_visual", file_name)
        file_path = input_dir + "/" + file_name
        
        print('Face : Processing {}'.format(file_path))
        
        out = cli.containers.run(image='algebr/openface', auto_remove=True, detach=True)
        
        if 'util' in base_dir_path:
            cmd = "-c 'cd ../{}/ && ls -lh && bash visual_1_1_openface_runner.sh {} {}'".format(user_name, file_path, openface_output_dir)
        else:
            cmd = "-c 'cd ../{}/ && ls -lh && bash eval_pipeline/audio_face_util/visual_1_1_openface_runner.sh {} {}'".format(user_name, file_path, openface_output_dir)
         
        logs = cli.containers.run(image='algebr/openface',
                           auto_remove=True,
                           detach=False,
                           cpu_count=multiprocessing.cpu_count(), #ONLY WORKS in WINDOWS
                           volumes={base_dir_path: {'bind': '/home/{}'.format(user_name), 'mode': 'rw'},
                                    '/dev/null':{'bind': '/dev/raw1394', 'mode':'rw'}},
                           #command="""-c 
                           #            'cd ../takeshi-s/ && \
                           #             ls -lah && \
                           #             bash visual_1_1_openface_runner.sh {}'""".format(file_path),
                           command=cmd,
                           name="openface")

        #pprint.pprint(str(logs))
            
        e_time = time.time()
        print('Face : processing time for {} : {} : {:.1f}'.format(file_name, e_time - s_time, (i+1)/total_size*100))
    
        if DEBUG:
            print(logs.decode())
    
            container_list = cli.containers.list(all=True)
            pprint.pprint(container_list)
    
    print('Face : Extracting features : Done')
    
    print('Face : Finalize (Copy csv) : Processing')
    tmp_src_dir = openface_output_dir
    tmp_tgt_dir = face_data_dir
    file_names = os.listdir(tmp_src_dir)
    file_names = [x for x in file_names if '.csv' in x]
    for file_name in file_names:
        shutil.copyfile(os.path.join(tmp_src_dir, file_name),
                    os.path.join(tmp_tgt_dir, file_name))
    print('Face : Finalize (Copy csv) : Done')
    
def calc_AU(openface_output_dir):
        
    print('Face : Calculating AU features : Processing')
    
    
    AU_features = []
    
    csv_files = [x for x in sorted(os.listdir(openface_output_dir)) if '.csv' in x]
    
    #def extract_int(x):
    #    tmp = x.split('.')[0]
    #    return int(tmp)
    #csv_files = sorted(csv_files, key=extract_int)
    
    csv_files = sorted(csv_files)
    
    for file_name in csv_files:
        
        print('Face : Processing {}'.format(file_name))
        
        # Load openface's output
        openface_data = pd.read_csv(os.path.join(openface_output_dir, file_name))
        tmp = [list(openface_data.columns)]
        tmp.extend(openface_data.values.tolist())
        openface_data = np.asarray(tmp).T.tolist()
        
        #print(openface_data[0])
        #['frame', '1.0', '2.0', '3.0',
        
        AU_data = []
        
        # Extract AU features from the openface's output
        for frame_data in openface_data:
            if ("AU" in frame_data[0]):
                AU_data.append(frame_data)
        #AU_data = np.asarray(AU_data).T.tolist()
            
        AU_feature = []
    
        #calculate AU average score
        for frame_data in AU_data:
            num = 0
            if ('AU' in frame_data[0]) and ('_c' in frame_data[0]):
                for x in frame_data:
                    if not (x == '0'):
                        
                        #skip label cell
                        if 'AU' in x:
                            continue
                        else:
                            num = num + float(x)
                            
                feature = num / (len(frame_data) - 1 )
                AU_feature.append(feature)
        
        #calculate happiness
        num = 0
        for i in range(len(AU_data[0])):
            if not ((AU_data[21][i] == '0') and (AU_data[25][i] == '0')):
                if 'AU' in AU_data[21][i]:
                    continue
                num = num + float(AU_data[21][i])*float(AU_data[25][i])
        feature = num / (len(AU_data[0])-1)
        AU_feature.append(feature)
                
        #insert id into top column
        AU_feature.insert(0,file_name.split('.')[0])
        
        AU_features.append(AU_feature)
        
    csv_data = pd.read_csv(os.path.join(openface_output_dir, file_name))
    labels =[x for x in list(csv_data.columns) if ('_c' in x)]
    labels.insert(0, 'ID')
    labels.append('AU06+12_happiness')

    AU_features.insert(0, labels)
    
    with open('feature_visual.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(AU_features)
    
    print('Face : Calculating AU features : Done')
        
if __name__ == '__main__':
    calc_visual()
