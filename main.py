#!/bin/python3.6

#=========================
# Python's Module Loading 
#=========================
import argparse
import os
import pickle
import nltk
import pandas as pd
import numpy as np

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
    if args.verbose or args.debug :
        print(f"Number of rows after cleaning : {len(DATA)}")
