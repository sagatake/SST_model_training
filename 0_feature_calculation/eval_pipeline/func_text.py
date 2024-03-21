#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:48:23 2021

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

from sudachipy import tokenizer
from sudachipy import dictionary
from scipy.spatial.distance import cosine


import torch

# As of Nov 16 2021, for installation, please follow https://qiita.com/m__k/items/863013dbe847dc613844 
from transformers import BertModel, BertForNextSentencePrediction
#from transformers.tokenization_bert_japanese import BertJapaneseTokenizer as BertTokenizer
from transformers import BertJapaneseTokenizer as BertTokenizer
print('Loading BERT ...')
# bert_tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_tokenizer = BertTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
bert_model = BertModel.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
print('Done')

import difflib

#To run following function, need to insatall MeCab and the latest neologdn dictionary files
#import MeCab
#import neologdn
#wakati = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
#tagger = MeCab.Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

#As of Nov 16 2021, transformers(cl-tohoku) uses fugashi for tokenization
#To install fugashi, https://pypi.org/project/fugashi/
from fugashi import Tagger, GenericTagger
#tagger = Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
tagger = Tagger()
#tagger = GenericTagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
#tagger = GenericTagger()

from sudachipy import tokenizer, dictionary
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

#dummy value. This will be alwayse over-rided in the parent source code.
DEBUG = None

MAX_TOKEN_LEN = 512

def calc_WPM(text, face):
    
    cnt = 0
    frame_per_second = 30
    
    for IPU in text:
        tokens = tokenize_sudachi(IPU)
        cnt += len(tokens)
        
    # calc WPM
    feature = cnt / (len(face) - 1) * frame_per_second * 60
    
    return [feature]

def count_backchannels(text):
    
    num_backchannel = 0
    
    for IPU in text:
        words =[m for m in tokenizer_obj.tokenize(IPU, mode)]
        if len(words) < 5:
            num_backchannel += 1
    
    return [num_backchannel]

def BERT_sentence(text):
    
    BERT_embed_list = []
    for IPU in text:
        BERT_word_embed_list = get_BERT_embed(IPU)
        BERT_sentence_embed = np.average(BERT_word_embed_list, axis=0)
        BERT_embed_list.append(BERT_sentence_embed)
    score = calc_average_cos(BERT_embed_list)
    
    return [score]

def BERT_cont_word(text):
        
    tgt_pos = ['動詞', '形容詞', '名詞', '副詞', '形状詞']
    tgt_embeddings = []
    for IPU in text:
        
        BERT_tokens, BERT_ids, flag_list = match_BERT_sudachi_tokens(IPU,tgt_pos=tgt_pos)
        
        if len(BERT_tokens) > MAX_TOKEN_LEN:
            BERT_tokens = BERT_tokens[:MAX_TOKEN_LEN]
            BERT_ids = BERT_ids[:MAX_TOKEN_LEN]
        
        raw_embeddings = calc_BERT_embed(BERT_ids)
        for i in range(len(BERT_ids)):
            if flag_list[i]==1:
                tgt_embeddings.append(raw_embeddings[i])

    score = calc_average_cos(tgt_embeddings)
    
    return [score]
        
def check_thanks(text):
        
    text = ''.join(text)
    flag = _vocab_match(text, ['有り難う', 'サンキュー', 'どうも', '感謝'])
    
    return [flag]

def count_content_words(text):
    
    num_total = 0
    num_content = 0
    #print(text)
    for IPU in text:
        words =[m for m in tokenizer_obj.tokenize(IPU, mode)]
        for word in words:
            
            dict_form = word.dictionary_form()
            norm_form = word.normalized_form()
            pos = word.part_of_speech()
            if DEBUG:
                print(dict_form)
                print(norm_form)
                print(pos)
            
            if pos[0] in ['名詞','動詞', '形容詞', '副詞', '形状詞']:
                #print(dict_form)
                num_total += 1
                num_content += 1
            
            else:
                num_total += 1
                
    feature = num_content/num_total
    
    return [feature]
            

def check_initial_que(text):
    
    flag = 0
    
    target_words = ['すみません', 'ねえ', 'ねえねえ']
    init_IPU = text[0]
    words = [m for m in tokenizer_obj.tokenize(init_IPU, mode)]
    for target_word in target_words:
        if target_word == words[0]:
            flag = 1
            break
    
    return [flag]

def match_BERT_sudachi_tokens(raw_text, tgt_pos=['名詞']):
        extracted_tokens, sudachi_flags, raw_tokens = tokenize_sudachi(raw_text,
                                                tgt_pos=tgt_pos,
                                                return_flags=True)
        BERT_tokens, BERT_ids = tokenize_BERT(raw_text)
        
        matched_pairs = []
        
        output_flags = []
        j=0
        for i in range(1, len(BERT_tokens)-1):
            j_memory = j
            while True:
                temp = BERT_tokens[i].replace('#','')
                match_ratio = difflib.SequenceMatcher(None, temp, raw_tokens[j]).ratio()
                #print(str(match_ratio) + ' : ' + BERT_tokens[i] + ' : ' + raw_tokens[j])
                #input()
                if match_ratio > 0.25:
                    if sudachi_flags[j]==1:
                        output_flags.append(1)
                    else:
                        output_flags.append(0)

                    matched_pairs.append([BERT_tokens[i], raw_tokens[j]])
                        
                    #since raw_tokens by sudach can be devided into several BERT_tokens, this process is needed
                    j-=3
                    if j<0:
                        j=0
                    
                    break
                
                j+=1
                
                #if no matched tokens were found
                if j>=len(raw_tokens):
                    output_flags.append(999999999999999999999)
                    j = j_memory
                    break
                

        #put flags for [CLS][SEP]
        output_flags.insert(0, 0)
        output_flags.append(0)
        
        if len(BERT_tokens)!=len(output_flags):
            print('No matched tokens were detected ...')
            sys.exit()
        
        #print('\n\n\n')           
        #for i in range(len(BERT_tokens)):
        #    print(BERT_tokens[i] + ' : ' + str(output_flags[i]))
        #input()
        
        #print('\n\n\n')           
        #for x in matched_pairs:
        #    print(x[0] + '\t: ' + x[1])
        #input()
        
        return BERT_tokens, BERT_ids, output_flags

def get_BERT_embed(text, ALIGN=False, tgt_pos=['代名詞', '副詞', '助動詞', '助詞',
                                               '動詞', '名詞', '形容詞', '感動詞',
                                               '接尾辞', '接続詞', '接頭辞', '空白',
                                               '補助記号', '記号', '連帯詞',
                                               'フィラー']):
    
    if ALIGN:
        #word_list = word_align(text, tagger)
        word_list = normalize_mecab(text, tgt_pos=tgt_pos)
        text = ''.join(word_list)
    
    bert_tokens, token_ids = tokenize_BERT(text)
    
    if len(bert_tokens) > MAX_TOKEN_LEN:
        bert_tokens = bert_tokens[:MAX_TOKEN_LEN]
        token_ids = token_ids[:MAX_TOKEN_LEN]
    
    outputs = calc_BERT_embed(token_ids)
    return outputs

def tokenize_BERT(sentence):
    #bert_tokens = bert_tokenizer.tokenize(" ".join(["[CLS]"] + output + ["[SEP]"]))
    input_ids = bert_tokenizer.encode(sentence, return_tensors='pt')
    bert_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    #print("BERT tokens: ")
    #for i in range(len(bert_tokens)):
    #    print(str(i) + bert_tokens[i])
    token_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
    #print("BERT token IDs: ", token_ids)
    return bert_tokens, token_ids

def calc_BERT_embed(token_ids):
    """
    ベクトル取得
    """
    #print("\n *** to Vector ***")
    #print(token_ids)
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
    #print(np.shape(tokens_tensor))
    #outputs, _ = bert_model(tokens_tensor)
    outputs = bert_model(tokens_tensor).last_hidden_state
    #print(outputs)
    #print(type(outputs))
    #print(np.shape(outputs))
    #print(outputs[0], "\n (size: ", outputs[0].size(), ")")
    outputs = outputs.detach().numpy().copy()
    return outputs[0]

def normalize_mecab(raw_sentence, tgt_pos=['代名詞', '副詞', '助動詞', '助詞',
                                           '動詞', '名詞', '形容詞', '感動詞',
                                           '接尾辞', '接続詞', '接頭辞', '空白',
                                           '補助記号', '記号', '連帯詞', 'フィラー']):
    sentence = tagger.parse(raw_sentence).split('\n')
    
    #eliminate 'EOS' and ''
    sentence.pop(-1)
    sentence.pop(-1)
    
    for i in range(len(sentence)):
        sentence[i] = sentence[i].split('\t')
        #print(sentence[i])
        sentence[i][1] = sentence[i][1].split(',')

    #sentence = [x[0] for x in sentence if (x[1][0] != 'フィラー') and (x[1][0] != '記号') and (x[1][0] != '感動詞') and (x[1][0] != '連体詞')]
    #sentence = [x[0] for x in sentence if (x[1][0] != '記号')]
    #sentence = [x[0] for x in sentence if ((x[1][0] == '名詞') or (x[1][0] == '動詞') or (x[1][0] == '形容詞') or (x[1][0] == '副詞'))]
    #sentence = [x[0] for x in sentence]
    
    output = []
    for x in sentence:
        if ((x[1][0] in tgt_pos) or (x[1][1] in tgt_pos)):
            output.append(x[0])
        
    return output

def calc_average_cos(embed_list):
    temp = []
    for i in range(0, len(embed_list)-1):
        #print(embed_list[i])
        #print(np.shape(embed_list[i+1]))
        temp.append(cosine(embed_list[i], embed_list[i+1]))
        #print(temp[-1])
        #input()
    ave = np.average(temp)
    return ave

def normalize_sudachi(text, 
                      tgt_pos = ['補助記号','空白','名詞','記号',
                                 '接頭辞','感動詞','副詞','接尾辞',
                                 '代名詞','形状詞','動詞','形容詞',
                                 '連体詞','接続詞',
                                 '助詞','助動詞']):
    
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer_obj.SplitMode.C
        
    out = [x.normalized_form() for x in tokenizer_obj.tokenize(text, mode) if x.part_of_speech()[0] in tgt_pos]
        
    #x.surface()
    #x.reading_form()
    
    return out

def tokenize_sudachi(text, 
                     tgt_pos = ['補助記号','空白','名詞','記号',
                                '接頭辞','感動詞','副詞','接尾辞',
                                '代名詞','形状詞','動詞','形容詞',
                                '連体詞','接続詞',
                                '助詞','助動詞'],
                     return_flags=False):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer_obj.SplitMode.C
    #out = tokenizer_obj.tokenize(text,mode)
    
    extracted_tokens = []
    flags = []
    raw_tokens = []
    for x in tokenizer_obj.tokenize(text, mode):
        if x.part_of_speech()[0] in tgt_pos:
            x = str(x)
            extracted_tokens.append(x)
            flags.append(1)
            raw_tokens.append(x)
        else:
            x = str(x)
            flags.append(0)
            raw_tokens.append(x)
    #out = [x for x in tokenizer_obj.tokenize(text, mode) if x.part_of_speech()[0] in tgt_pos]

    #print(out)
    if return_flags:
        return extracted_tokens, flags, raw_tokens
    else:
        return extracted_tokens

def check_seems_sorry(text):
    
    text = ''.join(text)
    flag = _vocab_match(text, ['申し訳', '御免'])
    
    return [flag]

"""
####################
--- Task : refuse
####################
"""
def check_explicit_refuse(text):

    text = ''.join(text)
    flag = _vocab_match(text, ['無理', '厳しい', 'できない'])
    #"できない"　どうやって検出
    
    return [flag]

def _vocab_match(text, tgt_vocab_list):
    
    flag = 0
    
    words =[m for m in tokenizer_obj.tokenize(text, mode)]
    for word in words:
        
        dict_form = word.dictionary_form()
        norm_form = word.normalized_form()
        pos = word.part_of_speech()
        
        if DEBUG:
            print(dict_form)
            print(norm_form)
            print(pos)
        
        if norm_form in tgt_vocab_list:
            flag = 1
            break
        
    return flag

    

