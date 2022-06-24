#!/bin/python3.6

"""
Module to create sklearn pipeline to predict tags of Stackoverflow question
"""
#=========================
# Python's Module Loading
#=========================
import warnings
import logging
import argparse
import os
import re
import sys
import pickle
import joblib
import nltk
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import sklearn.pipeline
import sklearn.svm
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#=========================
# functions
#=========================
def clean_tag_rm(ltag, rm_list=None) :
    """
    function to remove tag in the tag list
    parameters : l list of tags
                 rm_list : list of tags to be removed
    returns : list : new list of tags
    """
    if rm_list is None :
        rm_list = []
    return list(set(ltag)-set(rm_list))

def clean_tag_keep(ltag, keep_tags=None) :
    """
    function to keep only tag in the tag list
    parameters : l list of tags
                 list : list of tags to keep
    returns : list : new list of tags
    """
    if keep_tags is None :
        keep_tags = []
    return [t for t in ltag if t in keep_tags]

def tokenizer(txt) :
    """
    function to tokenize question of stackoverflow without lemmatation
    Parameters : txt : str
    returns : str
    """
    tag_tokenizer = nltk.RegexpTokenizer(r'</?(?:b|p)>', gaps=True)
    txt_tokenizer = nltk.RegexpTokenizer(r'\w+')

    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = re.sub(r'_+', ' ', txt)

    tokens = txt_tokenizer.tokenize(' '.join(tag_tokenizer.tokenize(txt.lower())))
    return ' '.join(tokens)

def use_features(sentences, embedding, batch_size=16):
    """
    function to create USE features 
    Parameters : sentences : str
                 embedding : tensorflow model
                 batch_size : int
    Returns : features : np.array
    """
    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embedding(sentences[idx:idx+batch_size])
        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))
    return features

class UseTransformer() :
    """
    Class to transform questions of stackoverflow to USE features
    """
    def __init__(self, tokenizer=None, embedding=None, batch_size=1) :
        self.batch_size = batch_size
        self.x_tokenized = None
        if tokenizer is None :
            self.tokenizer = self.dummy
        else :
            self.tokenizer = tokenizer
        if embedding is not None :
            self.embedding = embedding
        else :
            print("Error : Please give me embedding")
            sys.exit()

    def transform(self, X, y=None) :
        """
        Method to transform str
        Parameters : X : pd.Series
        returns : np.array
        """
        self.x_tokenized = X.apply(self.tokenizer)
        sentences = self.x_tokenized.to_list()
        for step in range(len(sentences)//self.batch_size) :
            idx = step*self.batch_size
            feat = self.embedding(sentences[idx:idx+self.batch_size])
            if step ==0 :
                features = feat
            else :
                features = np.concatenate((features,feat))
        return features

    def fit(self, X, y=None) :
        """
        Method needed in sklearn pipeline 
        Do nothing
        return self
        """
        return self

    def dummy(self, txt):
        """
        function to return the txt without processing
        """
        return txt

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
        sys.exit()
    else :
        DATA = pd.read_csv("QueryResults.csv")
        for param in ['Body', 'Tags'] :
            if param not in DATA.columns :
                print(f"Error : {param} is not in the DataFrame")
                sys.exit()
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
    if args.debug :
        features = UseTransformer(tokenizer=tokenizer, \
                                  embedding=hub.load("https://tfhub.dev/google/universal-sentence-encoder/4"), \
                                  batch_size=1).transform(DATA['Body'])
        print(f"Shape of USE features : {features.shape}")
        print(features)

    #=========================
    # Pipeline
    #=========================
    if args.verbose or args.debug :
        print("Model fitting...")
    use_pipeline = sklearn.pipeline.Pipeline(steps=[\
                                                    ('use_features_trans', \
                                                     UseTransformer(tokenizer=tokenizer, \
                                                                    embedding=hub.\
                                                                    load("https://tfhub.dev/google/universal-sentence-encoder/4"), \
                                                                    batch_size=1)
                                                 ),\
                                                    ('SVM_classifier', sklearn.svm.SVC() )\
                                                ])

    #=========================
    # Training
    #=========================
    X_train, X_test, y_train, y_test = train_test_split(DATA['Body'].iloc[:200], \
                                                        y_one[:200], \
                                                        test_size=0.3,
                                                        random_state=42)
    use_pipeline.fit(X_train, y_train)

    y_pred_train = use_pipeline.predict(X_train)
    y_pred_test = use_pipeline.predict(X_test)
    if args.debug :
        print(f"y_pred_train = {y_pred_train}")
        print(f"y_pred_test = {y_pred_test}")
    print(f"Accuracy train set: {accuracy_score(y_train, y_pred_train)}")
    print(f"Accuracy test set: {accuracy_score(y_test, y_pred_test)}")

    #=========================
    # save pipeline
    #=========================
    #with open('pipeline_use-svc.pkl', 'wb') as ofile :
    #    pickle.dump(use_pipeline, ofile, pickle.HIGHEST_PROTOCOL)
    #joblib.dump(use_pipeline, 'pipeline_use-svc.joblib')
