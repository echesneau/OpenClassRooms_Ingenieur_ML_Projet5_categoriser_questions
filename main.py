#!/bin/python3.6

#=========================
# Python's Module Loading 
#=========================
import argparse
import os
import pickle
import re
import nltk
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

import tensorflow as tf
tf.get_logger().setLevel('INFO')
import tensorflow_hub as hub

import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#=========================
# functions
#=========================
def clean_tag_rm(l, rm_list=[]) :
    """
    function to remove tag in the tag list
    parameters : l list of tags
                 rm_list : list of tags to be removed
    returns : list : new list of tags
    """
    return list(set(l)-set(rm_list))

def clean_tag_keep(l, tags=[]) :
    """
    function to keep only tag in the tag list
    parameters : l list of tags
                 list : list of tags to keep
    returns : list : new list of tags
    """
    return [t for t in l if t in tags]

def tokenizer(txt) :
    tag_tokenizer = nltk.RegexpTokenizer(r'</?(?:b|p)>', gaps=True)
    txt_tokenizer = nltk.RegexpTokenizer(r'\w+')

    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = re.sub(r'_+', ' ', txt)
    
    tokens = txt_tokenizer.tokenize(' '.join(tag_tokenizer.tokenize(txt.lower())))
    return ' '.join(tokens)

def use_features(sentences, embedding, batch_size=16):
    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embedding(sentences[idx:idx+batch_size])
        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))
    return features
    

if __name__ == "__main__" :
    #=========================
    # Args reading 
    #=========================
    parser = argparse.ArgumentParser(description='Create clustering model '+\
                                     'for questions of StackOverFlow '+\
                                     '\n Written by E. Chesneau')
    parser.add_argument('--ifile', '-i', \
                        help="Input Database export from Stackoverflow")
    parser.add_argument('--verbose', '-v', \
                        help="Verbose Mode", \
                        action="store_true")
    parser.add_argument('--debug', \
                        help="Debug Mode", \
                        action="store_true") 

    args=parser.parse_args()
    if args.verbose :
        print("Run in verbose mode")
    if args.debug :
        print("Run in debug mode")

    #=========================
    # Inputs checking
    #=========================
    if args.verbose or args.debug :
        print("Data loading...")
    if args.ifile is None or not os.path.isfile(args.ifile) :
        print(f"Error : {args.ifile} is not a file")
        exit()
    else :
        DATA = pd.read_csv("QueryResults.csv")
        for param in ['Body', 'Tags'] :
            if param not in DATA.columns :
                print(f"Error : {param} is not in the DataFrame")
                exit()
        if args.verbose or args.debug :
            print(f"Number of rows : {len(DATA)}")
            print(f"Number of variable : {len(DATA.columns)}")
        if args.debug :
            print(DATA)
    
    #=========================
    # Process tags list
    #=========================
    if args.verbose or args.debug : 
        print("Tags processing...")
    tags_tokenizer = nltk.SExprTokenizer(parens='<>')
    DATA['Tags_list'] = DATA['Tags'].apply(tags_tokenizer.tokenize)
    count = DATA['Tags_list'].explode("Tags_list").value_counts()
    freq = count/len(DATA)*100
    tags = freq[:20].index
    if args.debug :
        print('Tags keeped :')
        print(tags)
    DATA['Tags_list'] = DATA['Tags_list'].apply(clean_tag_keep, tags=tags)
    DATA['nTags'] = DATA['Tags_list'].apply(len)
    if args.debug :
        print(DATA[['Tags_list', 'nTags']])
    DATA = DATA.drop(DATA[DATA['nTags']==0].index)
    DATA['one_tag'] = DATA['Tags_list'].apply(lambda x : x[0])
    if args.verbose or args.debug :
        print(f"Number of rows after cleaning : {len(DATA)}")

    #=========================
    # tags encoding
    #=========================
    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(DATA['Tags_list'])

    enc = LabelEncoder()
    y_one = enc.fit_transform(DATA['one_tag'])

    #=========================
    # USE features
    #=========================
    embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    DATA['Body_tokenized'] = DATA['Body'].apply(tokenizer)
    if args.debug :
        print(DATA['Body_tokenized'])
    features = use_features(DATA['Body_tokenized'].to_list(), embedding, batch_size=5)
    if args.verbose or args.debug :
        print(f"Shape of USE features : {features.shape}")
    if args.debug :
        print(features)
    #Make a pipeline : creation des features, fit, prediction, retour de la liste des tags en format str
