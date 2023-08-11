import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def prospective_conditions(sampleNum):
    # ground truths
    CTRL = ['ND09','ND16','ND17','ND20','ND21','ND24','ND27','ND28','785','786']
    TAM = ['ND14','ND15','ND22','ND23','ND30']
    DSS = ['621','622','ND10','ND18','ND19','ND25','ND26','ND29','208','332']
    COMBO = ['595','610','611']
    
    if sampleNum in CTRL:
        return 'Ctrl'
    elif sampleNum in TAM:
        return 'TAM_colitis'
    elif sampleNum in DSS:
        return 'DSS_colitis'
    elif sampleNum in COMBO:
        return 'Combined_Induction'

def prospective_conditions_numEncoded(sampleNum):
    # ground truths
    CTRL = ['ND09','ND16','ND17','ND20','ND21','ND24','ND27','ND28','785','786']
    TAM = ['ND14','ND15','ND22','ND23','ND30']
    DSS = ['621','622','ND10','ND18','ND19','ND25','ND26','ND29','208','332']
    COMBO = ['595','610','611']
    
    if sampleNum in CTRL:
        return 3
    elif sampleNum in TAM:
        return 0
    elif sampleNum in DSS:
        return 1
    elif sampleNum in COMBO:
        return 2

def decode_conditions(cond):
    if cond == 0:
        return 'TAM_colitis'
    elif cond == 1:
        return 'DSS_colitis'
    elif cond == 2:
        return 'Combined_Induction'
    elif cond ==3:
        return 'Ctrl'

def LDA_infer_ROI(config):
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    save_dir = os.path.join(BASE_DIR,'LDA_Predictions')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Load up DFs
    train_df = pd.read_csv('./6_mouseModelInference/archivedMouseCohort_Proportions/archivedMouseCohort.csv')
    
    test_df = pd.read_csv(os.path.join(config['directories']['GEN_PATCHES_ROI_DIR'],'invUninvolvedPatchCounts_test_wProps.csv'))

    # Extract column Names with patch proportions
    test_cols = [a for a in test_df.columns if str(a).endswith('percent')]
    train_cols = [a for a in train_df.columns if str(a).endswith('percent')]
    
    # need to get rid of Uninvolved cluster percents in the test df but still need UninvCount_percent
    test_cols = [a for a in test_cols if '_Inv_' in a]
    #test_cols.append('UNinvCount_percent')
    test_cols.insert(0,'UNinvCount_percent')
    
    # Extract Archived Mouse Cohort Proportions and GT Labels
    train_y = np.array(train_df['condition'])
    train_x = np.array(train_df[train_cols])
    
    # Extract Prospective or Inferene Cohort Proportions
    test_x = np.array(test_df[test_cols])
    test_sample = np.array(test_df['sample'])
    test_ROI = np.array(test_df['ROI'])

    # Fit LDA Classifier on Archived Mouse Cohort and Predict
    clf = LinearDiscriminantAnalysis(n_components=1)
    clf.fit(train_x, train_y)

    y_pred = clf.predict(test_x)
        
    save_fn = "InferenceCohort_LDA_summary.txt"
    summaryFilePath = os.path.join(save_dir,save_fn)
    writeSummary = open(summaryFilePath, 'w')
    for i in range(len(y_pred)):
        writeSummary.write('Sample # ' + str(test_sample[i]) + '__' + str(test_ROI[i]) + ': ' + decode_conditions(y_pred[i]) + '\n')
    writeSummary.close()
    
    return config
