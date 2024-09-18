import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# GradientBoostingRegressor
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,cross_validate
from sklearn.metrics import make_scorer

def data_split(trainset,element_names,per):
############trainset and testset split using holdouttest
    i = 0
    testset = pd.DataFrame()#columns=trainset.columns
    count = int(trainset.shape[0]*per/2/len(element_names))+1
    while i < count:
      for element in element_names:
        max_element = trainset[element].max()
        max_index = trainset[element].idxmax()

        min_element = trainset[element].min()
        min_index = trainset[element].idxmin()

        min_row=trainset.loc[min_index:min_index,:]
        max_row=trainset.loc[max_index:max_index,:]
        if max_index !=min_index:
          testset = pd.concat([testset, min_row], ignore_index=True)
          testset = pd.concat([testset, max_row], ignore_index=True)
          trainset = trainset.drop(min_index)
          trainset = trainset.drop(max_index)
        if max_index ==min_index:
          testset = pd.concat([testset, min_row], ignore_index=True)
          trainset = trainset.drop(min_index)
      i =i+1
    return trainset,testset

def data_sampling(trainset,comp_major_low,comp_major_high,comp_major_inter,
                 comp_minor_low,comp_minor_high,comp_minor_inter,
                 T_low,T_high,T_inter,size,element_names):
    ##################################################################################
    comp_major_bins = np.arange(comp_major_low, comp_major_high, comp_major_inter)
    comp_minor_bins = np.arange(comp_minor_low, comp_minor_high, comp_minor_inter)
    temp_bins = np.arange(T_low, T_high, T_inter)
    temp_bins  = 1000/(temp_bins+273)
    temp_bins.sort()
    ##################################################################################
    major_feature_names = ['Ni','Co','Cr','Al','Fe'] ###major element name
    minor_feature_names = set(element_names) - set(major_feature_names)
    minor_feature_names = list(minor_feature_names) ###minor element name
    ##################################################################################
    for compy in major_feature_names:
        trainset[compy+'_Bin'] = pd.cut(trainset[compy],bins=comp_major_bins, labels=range(len(comp_major_bins)-1))
    for compy in minor_feature_names:
        trainset[compy+'_Bin'] = pd.cut(trainset[compy],bins=comp_minor_bins, labels=range(len(comp_minor_bins)-1))
    trainset['Test_Temperature_Bin'] = pd.cut(trainset['invT'],bins=temp_bins, labels=range(len(temp_bins)-1))
    all_bin_cols = [col for col in trainset.columns if '_Bin' in col]
    size = size        
    replace = True  # with replacement
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
    Sampled_trainset = trainset.groupby(all_bin_cols, as_index=False, group_keys=False).apply(fn)
    ##################################################################################################
    keep_cols = [col for col in Sampled_trainset.columns if '_Bin' not in col]
    Sampled_trainset = Sampled_trainset[keep_cols]
    return Sampled_trainset

# Function to calculate Euclidean distance
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2)**2))

# Apply the function to each group
def calculate_distance(group):
    if len(group) == 1:
        group['Distance'] = None
        # return 0
    elif len(group) == 2:
        row1 = group.iloc[0, 1:][['Ni', 'Cr', 'Co', 'Al', 'Fe']].values
        row2 = group.iloc[1, 1:][['Ni', 'Cr', 'Co', 'Al', 'Fe']].values
        group['Distance'] = euclidean_distance(row1, row2 )
    else:
        # return np.nan
        group['Distance'] = None
    return group

# Apply the function to each group
def finalscore(group):
    alpha = 0.5
    beta = 1-alpha 
    if len(group) == 1:
        print('wrong single phase alloy')
    elif len(group) == 2:
        s1 = group.iloc[0, 1:]['Phase_Score']#.values
        s2 = group.iloc[1, 1:]['Phase_Score']#.values
        s3 = group.iloc[0, 1:]['Distance_Rank']#.values
        
        group['Final Score'] = (s1+s2)*beta+s3*alpha
    else:
         print('wrong multiple phase alloy')
    return group