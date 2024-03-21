#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 22:35:31 2021

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

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import BayesianRidge, ARDRegression

from sklearn.svm import SVR

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn import preprocessing

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

from scipy.stats import pearsonr

import pickle as pkl

import warnings
warnings.simplefilter('ignore')

DEBUG = False

FEATURE_SELECT = False

num_select = None
thresh_abs = None
thresh_accum = None

NESTED = True
#num_select = 10
#thresh_abs = 0.05
#thresh_accum = 0.5

TUNE = False

#MODEL = 'Linear'
#MODEL = 'ARD'
#MODEL = 'SVR_linear'
#MODEL = 'SVR_rbf'
MODEL = 'RandomForest'
#MODEL = 'Stacking'

LOAD = True

def main():
    "Main function"
    
    feature_dir    = 'aligned_feature_par'
    
    #feature_dir    = 'aligned_feature_par_wo_BERT'
    #feature_dir    = 'aligned_feature_par_wo_interactive'
    
    #feature_dir    = 'aligned_feature_par_wo_face'
    #feature_dir    = 'aligned_feature_par_wo_gesture'
    #feature_dir    = 'aligned_feature_par_wo_audio'
    #feature_dir    = 'aligned_feature_par_wo_text'
    
    label_dir      = 'aligned_subj_score_par'
    
    #tgt_index = 2
    
    feature_files = os.listdir(feature_dir)
    label_files = os.listdir(label_dir)
    
    pairs = _make_pair(feature_files, label_files)
    
    output_metrics = [['TASK', 'LABEL', 'R2', 'RMSE', 'CORREL', 'p-value']]
    output_metrics_file = 'summary.csv'
    
    output_selected_features = [['Task_SubjScore', 'Rank', 'Name', 'Importance']]
    
    #TODO! 1/3 model save/load. model_names list
    model_save_dir = 'trained_model'
    _check_dir(model_save_dir)
    if LOAD:
        file_name = 'model_names.sav'
        file_path = os.path.join(model_save_dir, file_name)
        model_names = pkl.load(open(file_path, 'rb'))
    else:
        model_names = []
    
    for feature_file, label_file in pairs:
        
        tra_par = feature_file.split('_')[1]
        task_name = feature_file.split('_')[2][:-4]

        
        output_list = []
        
        feature_path = os.path.join(feature_dir, feature_file)
        label_path = os.path.join(label_dir, label_file)
        
        features = _load_csv(feature_path)
        labels = _load_csv(label_path, encoding = 'shift-jis')

        feature_columns = features[0]
        features = features[1:]
        label_columns = labels[0]
        labels = labels[1:]
        
        IDs, features = _split_ID(features)
        for i in range(len(features)):
            for j in range(len(features[0])):
                features[i][j] = float(features[i][j])
        
        #TODO! 2/3 model save/load. task specific normalizer
        if LOAD:
            model_name = _find_model_path(model_names, task_name)
            model_path = os.path.join(model_save_dir, model_name)
            normalizer = pkl.load(open(model_path, 'rb'))
        else:
            normalizer = preprocessing.StandardScaler()
            normalizer.fit(features)
            model_name = 'model_normalizer_{}.sav'.format(task_name)
            model_path = os.path.join(model_save_dir, model_name)
            pkl.dump(normalizer, open(model_path, 'wb'))
            model_names.append(model_name)
            
        
        features = normalizer.transform(features)
        features = _concat_ID(IDs, features)
        
        #print(features)
        #input('enter')
        
        IDs, labels = _split_ID(labels)
        for i in range(len(labels)):
            for j in range(len(labels[0])):
                labels[i][j] = float(labels[i][j])
        labels = _concat_ID(IDs, labels)
        
        output_importances = [feature_columns.copy()]
        output_importances[0][0] = 'Task_SubjScore'

        output_norm_importances = [feature_columns.copy()]
        output_norm_importances[0][0] = 'Task_SubjScore'
        
        
        # select subj score type
        for label_index in range(len(labels[0][1:])):
            
            label_name = label_columns[label_index+1]
            print('{} : {}'.format(task_name, label_name))
                        
            IDs, labels = _split_ID(labels)
            single_label_list = _pick_one_column(labels, label_index)
            single_label_list = _concat_ID(IDs, single_label_list)
            labels = _concat_ID(IDs, labels)
            
            #pp.pprint(features[:3])
            #pp.pprint(labels[:3])
            #pp.pprint(single_label_list[:3])
            
            IDs, y = _split_ID(single_label_list)
            print(np.shape(IDs))
            print(np.shape(y))
            IDs, X = _split_ID(features)
            print(np.shape(IDs))
            print(np.shape(X))
            
            #TODO! 3/3 model save/load. task-label specific predicter
            if LOAD:
                model_name = _find_model_path(model_names, task_name, label_name)
                model_path = os.path.join(model_save_dir, model_name)
                model = pkl.load(open(model_path, 'rb'))
                print(model.score(X, y))
            else:
                model = RandomForestRegressor()
                model.fit(X, y)
                print(model.score(X, y))
                
                model_name = 'model_predicter_{}_{}.sav'.format(task_name, label_name)
                model_path = os.path.join(model_save_dir, model_name)
                pkl.dump(model, open(model_path, 'wb'))
                model_names.append(model_name)
            
    model_name = 'model_names.sav'
    model_path = os.path.join(model_save_dir, model_name)
    pkl.dump(model_names, open(model_path, 'wb'))

    
def select_features(features, feature_columns, norm_abs_importances, num_select=None):
    
    if num_select == None:
        print('num_select must be passed to select_features()')
        sys.exit()
    
    feature_IDs, features = _split_ID(features)
    features        = np.asarray(features)
    
    feature_columns = np.asarray(feature_columns[1:].copy())
    
    importance_pairs = [[column, importance] for [column, importance] in zip(feature_columns, norm_abs_importances)]
    sorted_importance_pairs = sorted(importance_pairs, key=lambda x:x[1], reverse=True)
    
    selected_feature_names = [pair[0] for pair in sorted_importance_pairs[:num_select]]
    selected_feature_pairs = [pair for pair in sorted_importance_pairs[:num_select]]
    
    
    if DEBUG:
        print(np.shape(feature_columns))
        print(np.shape(norm_abs_importances))
        
        pp.pprint(importance_pairs)
        pp.pprint(sorted_importance_pairs)
        pp.pprint(selected_feature_names)
        
        pp.pprint(feature_columns)

    selected_features = []
    selected_columns = []
    for tgt_name in selected_feature_names:
        index = np.where(feature_columns == tgt_name)
        #print(tgt_name)
        #print(index)
        selected_features.append(features[:, index[0]].tolist())
        selected_columns.append(feature_columns[index[0]][0].tolist())
        #input()
    
    selected_features = np.concatenate(selected_features, axis=1).astype(np.float32).tolist()
    
    #feature_IDs = np.reshape(feature_IDs, (-1, 1))
    #print(np.shape(feature_IDs))
    #print(np.shape(selected_features))
    #print(np.shape([feature_IDs, selected_features]))
    #selected_features.insert(0, feature_IDs)
    #selected_features = np.concatenate([feature_IDs, selected_features], axis=1).tolist()
    selected_features = _concat_ID(feature_IDs, selected_features)

    #print('Selected', selected_columns)
    #pp.pprint(selected_features)
    #input()

    if DEBUG:
        pp.pprint(selected_features)
        pp.pprint(selected_columns)
        print(np.shape(selected_features))
        print(np.shape(selected_columns))
    

    #sys.exit()
    
    return selected_features, selected_feature_pairs

def select_feature_indices(norm_abs_importances, num_select=None, thresh_accum=None, thresh_abs=None):
    
    if (num_select == None) and (thresh_accum == None) and (thresh_abs == None):
        print('num_select or thresh_accum or thresho_abs must be passed to select_features()')
        print('(select one of them)')
        sys.exit()
    
    #feature_IDs, features = _split_ID(features)
    #features        = np.asarray(features)
    
    #feature_columns = np.asarray(feature_columns[1:].copy())
    
    #importance_pairs = [[column, importance] for [column, importance] in zip(feature_columns, norm_abs_importances)]
    #sorted_importance_pairs = sorted(importance_pairs, key=lambda x:x[1], reverse=True)
    
    #selected_feature_names = [pair[0] for pair in sorted_importance_pairs[:num_select]]
    #selected_feature_pairs = [pair for pair in sorted_importance_pairs[:num_select]]
    
    sorted_index_array = np.argsort(-norm_abs_importances)
    sorted_importances = norm_abs_importances[sorted_index_array]
    
    if num_select != None:
        selected_index_array = sorted_index_array[:num_select]
        selected_importances = norm_abs_importances[selected_index_array]
        #print(selected_index_array)
        #print(selected_importances)

                
    elif thresh_accum != None:
        total = 0
        selected_index_list = []
        for index in sorted_index_array:
            #print(norm_abs_importances[index])
            total += norm_abs_importances[index]
            selected_index_list.append(index)
            #print('tmp sum: ', total)
            if thresh_accum <= total:
                break
        selected_index_array = np.asarray(selected_index_list)
        selected_importances = norm_abs_importances[selected_index_array]
        #print(norm_abs_importances)
        #print(selected_index_array)
        #print(selected_importances)
            
    elif thresh_abs != None:
        select_mask = norm_abs_importances >= thresh_abs
        #print(select_mask)
        selected_index_array = np.where(select_mask == True)[0]
        #print(selected_index_array)
        selected_importances = norm_abs_importances[selected_index_array]
        #print(selected_importances)
        
    if len(selected_importances) == None:
        print('No feature was selected')
        print('Abort')
        sys.exit()
        
    if DEBUG:
        pp.pprint(-norm_abs_importances)
        pp.pprint(norm_abs_importances)
        pp.pprint(sorted_index_array)
        pp.pprint(sorted_importances)
        pp.pprint(selected_index_array)
        pp.pprint(selected_importances)
    
    return selected_index_array
    

def CV_loop(feature_file, label_file, features, single_label_list, feature_columns=None, num_select=None, thresh_accum=None, thresh_abs=None):

    true_list = []
    pred_list = []
    #r2_list = []
    #rmse_list = []

    tmp_importances = []
    selected_feature_names = []
    
    if FEATURE_SELECT and NESTED:
        
        if TUNE:
            print('feature_selection: CV tune mode')
            num_select = None
            thresh_accum = None
            thresh_abs = None
            
        elif num_select != None:
            print('feature_selection: num_select mode')
    
            #thresh_accum = 0.8        
            #num_select = None
            
            thresh_accum = None
            thresh_abs = None
     
        elif thresh_accum != None:
            print('feature_selection: thresh_accum mode')
            num_select = None
            thresh_abs = None
    
        elif thresh_abs != None:
            print('feature_selection: thresh_abs mode')
            num_select = None
            thresh_accum = None

    
    # select one subj sample for test
    for test_index in range(len(single_label_list)):
        
        #print(test_index)
        #print(np.shape(features))
        #print(np.shape(single_label_list))
        
        if TUNE:
            X_train, X_test, y_train, y_test, train_IDs, test_IDs = leave_one_subject_out(features.copy(), single_label_list.copy(), test_index, RETURN_ID=True)
            
            tmp_features = _concat_ID(train_IDs, X_train)
            tmp_labels = _concat_ID(train_IDs, y_train)
            
            X_train, X_valid, y_train, y_valid = leave_one_subject_out(tmp_features, tmp_labels, train_ratio=0.7)
            
            X_train = np.asarray(X_train)
            X_valid = np.asarray(X_valid)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_valid = np.asarray(y_valid)
            y_test = np.asarray(y_test)
            
        else:
            X_train, X_test, y_train, y_test, train_IDs, test_IDs = leave_one_subject_out(features.copy(), single_label_list.copy(), test_index, RETURN_ID=True)
            
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
            
                            
            if DEBUG:
                print(feature_file)
                print(label_file)
                print(np.shape(X_train))
                print(np.shape(y_train))
                print(np.shape(X_test))
                print(np.shape(y_test))
                print(X_train[0])
                print(y_train[0])
                input()
        
        if MODEL == 'ARD':
            model = ARDRegression(compute_score=True)
            model.fit(X_train, y_train)
            importances = model.coef_
        if MODEL == 'SVR_linear':
            model = SVR(kernel='linear')
            model.fit(X_train, y_train)
            importances = model.coef_[0]
        if MODEL == 'SVR_rbf':
            model = SVR(kernel='rbf')
            model.fit(X_train, y_train)
            importances = [0]
        if MODEL == 'RandomForest':
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            importances = model.feature_importances_
        if MODEL == 'Linear':
            model = LinearRegression()
            model.fit(X_train, y_train)
            importances = model.coef_[0]
        if MODEL == 'Stacking':
            estimators = [
                ('Lasso', Lasso()),
                ('SVR_rbf', SVR(kernel='rbf')),
                ('RandomForest', RandomForestRegressor())
                ]
            model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
            model.fit(X_train, y_train)
            importances = [0]

        abs_importances = np.abs(importances)
        norm_abs_importances = abs_importances / np.sum(abs_importances)
        
        if FEATURE_SELECT and NESTED:
            
            if TUNE:
                
                max_r2 = -10000000
                num_select = 0
                for tmp_num_select in range(1, len(X_train[0])+1):
                    #print(tmp_num_select)
                    selected_indices= select_feature_indices(norm_abs_importances, tmp_num_select, thresh_accum, thresh_abs)
                    feature_columns_array = np.asarray(feature_columns[1:])            
                    
                    #print(np.shape(selected_indices))
                    #print(np.shape(feature_columns_array))
                    #print(np.shape(X_train))
                    #print(np.shape(norm_abs_importances))
                    #input('Enter')
                    selected_feature_names.extend(feature_columns_array[selected_indices].tolist())
                    
                    tmp_X_train = X_train[:, selected_indices]
                    tmp_X_valid = X_valid[:, selected_indices]

                    model = ARDRegression(compute_score=True)
                    model.fit(tmp_X_train, y_train)
                    
                    preds, preds_std = model.predict(tmp_X_valid, return_std=True)
                    #preds, preds_std = model.predict(tmp_X_valid)
                    r2 = r2_score(y_valid, preds)
                    #print(r2)
                    #print(y_valid)
                    #print(preds)
                    
                    if r2 > max_r2:
                        max_r2 = r2
                        num_select = tmp_num_select
                        
                print('Tuned num_select: ', num_select)
                
                
            
            selected_indices= select_feature_indices(norm_abs_importances, num_select, thresh_accum, thresh_abs)
            
            feature_columns_array = np.asarray(feature_columns[1:])            
            
            #print(np.shape(selected_indices))
            #print(np.shape(feature_columns_array))
            #print(np.shape(X_train))
            #print(np.shape(norm_abs_importances))
            #input('Enter')
            selected_feature_names.extend(feature_columns_array[selected_indices].tolist())
            
            X_train = X_train[:, selected_indices]
            X_test = X_test[:, selected_indices]
            
            #print(np.shape(X_train))
            #input()
            
            model = ARDRegression(compute_score=True)
            model.fit(X_train, y_train)
            
        
        if MODEL == 'ARD':
            preds, preds_std = model.predict(X_test, return_std=True)
        else:
            preds = model.predict(X_test)
            
        tmp_importances.append(importances)
                
        try:
            preds = np.reshape(preds, (-1, 1))
        except:
            pass
        
        #R_2 = model.score(X_train, y_train)
        #R_2 = model.score(X_test, y_test)
        #rmse = mean_squared_error(y_test, preds, squared = False)
        #input()
        
        true_list.extend(y_test)
        pred_list.extend(preds)
        
    ave_importances = np.average(tmp_importances, axis=0).tolist()
    
    abs_ave_importances = np.abs(ave_importances)
    norm_abs_ave_importances = abs_ave_importances / np.sum(abs_ave_importances)
        
    r2 = r2_score(true_list, pred_list)
    rmse = mean_squared_error(true_list, pred_list)
    correl, p_value = pearsonr(np.reshape(true_list, (-1,)), np.reshape(pred_list, (-1,)))
    
    if FEATURE_SELECT and NESTED:
        
        cnt_dict = {}
        for name in selected_feature_names:
            if name in cnt_dict.keys():
                cnt_dict[name] += 1
            else:
                cnt_dict[name] = 1
        cnt_list = []
        for name in cnt_dict.keys():
            cnt_list.append([name, cnt_dict[name]])
        selected_pairs = sorted(cnt_list, key=lambda x:x[1], reverse=True)[:num_select]
            
        return true_list, pred_list, ave_importances, norm_abs_ave_importances, r2, rmse, correl, p_value, selected_pairs
    
    else:
        
        return true_list, pred_list, ave_importances, norm_abs_ave_importances, r2, rmse, correl, p_value

    
def leave_one_subject_out(features, labels, test_index = 0, RETURN_ID = False, train_ratio = None):
    
    #features = features.copy()
    #labels = labels.copy()
    
    if train_ratio != None:
        
        IDs = []
        for i in range(len(features)):
            ID = labels[i][0].split('_')[0]
            if not(ID in IDs):
                IDs.append(ID)

        train_num = int(len(IDs) * train_ratio)
        train_IDs = []
        for ID in IDs:
            if len(train_IDs) <= train_num:
                train_IDs.append(ID)
            else:
                pass
        
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        
        for i in range(len(features)):
            
            ID = labels[i][0].split('_')[0]
            if ID in train_IDs:
                train_features.append(features[i])
                train_labels.append(labels[i])
            else:
                test_features.append(features[i])
                test_labels.append(labels[i])
                
    else:
                
        test_features = [features.pop(test_index)]
        test_labels = [labels.pop(test_index)]
        test_ID = test_labels[0][0].split('_')[0]
        
        train_features = []
        train_labels = []
        
        for i in range(len(features)):
            
            ID = labels[i][0].split('_')[0]
            #print(test_ID)
            #print(ID)
            if test_ID == ID:
                pass
            else:
                train_features.append(features[i])
                train_labels.append(labels[i])
    
    train_IDs, train_features = _split_ID(train_features)
    test_IDs, test_features = _split_ID(test_features)
    IDs, train_labels = _split_ID(train_labels)
    IDs, test_labels = _split_ID(test_labels)

    
    if RETURN_ID:
        return train_features, test_features, train_labels, test_labels, train_IDs, test_IDs
    else:        
        return train_features, test_features, train_labels, test_labels

def _split_ID(src):

    IDs = []
    data = []
    for row in src:
        IDs.append([row[0]])
        data.append(row[1:])
        
    return IDs, data

def _concat_ID(IDs, data):
    
    output_list = []
    for row_ID, row_data in zip(IDs, data):
        row = []
        row.extend(row_ID)
        row.extend(row_data)
        output_list.append(row)
    
    return output_list
        
def _pick_one_column(input_data, index):
    
    output_data = []
    for row in input_data:
        output_data.append([row[index]])
        
    return output_data
    
def _make_pair(feature_files, label_files):
    
    pairs = []
    
    for task in ['FAVOR', 'REFUSE', 'TELL', 'LISTEN']:
        
        feature_file = [x for x in feature_files if (task in x)][0]
        label_file = [x for x in label_files if (task in x)][0]
        pairs.append([feature_file, label_file])
    
    return pairs

def _load_csv(file_name, encoding = 'utf-8'):
    with open(file_name, encoding = encoding) as f:
        reader = csv.reader(f)
        data = []
        for raw in reader:
            #print(raw)
            data.append(raw)

    return data

def _write_csv(file_name, data, encoding = 'utf-8'):
    with open(file_name, 'w', encoding = encoding) as f:
        writer = csv.writer(f)
        writer.writerows(data)

def _write_preds(file_name, y_test, preds):
    output_list = [['truth', 'preds']]
    for i in range(len(y_test)):
        output_list.append([y_test[i][0], preds[i][0]])
    
    _write_csv(file_name, output_list)

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
    
if __name__ == '__main__':
    main()

