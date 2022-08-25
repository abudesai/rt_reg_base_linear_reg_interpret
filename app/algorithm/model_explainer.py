#!/usr/bin/env python

import os, warnings, sys
import numpy as np, pandas as pd
import pprint
warnings.filterwarnings('ignore') 

from interpret import show
import matplotlib.pyplot as plt



def save_global_explaination(model): 
    global_explanations = model.explain_global("x")
    # print(dir(global_explanations))
    explain_data = global_explanations.data()
    # pprint.pprint(global_explanations.data())
    
    explain_df = convert_explanations_to_df(explain_data)
    sys.exit()
    show(global_explanations)
    sys.exit()
    

def convert_explanations_to_df(explain_data): 
    # features and scores
    df = pd.DataFrame()
    
    names_list = explain_data['names']
    names_list.append(explain_data['extra']['names'][0])    
    scores_list = explain_data['scores']
    scores_list.append(explain_data['extra']['scores'][0])    
    
    df['feature'] = names_list
    df['score'] = scores_list
    
    df['abs_score'] = np.abs(explain_data['scores'])
    
    df.sort_values(by=['abs_score'], inplace=True, ascending=False)
    print(df)
    return df