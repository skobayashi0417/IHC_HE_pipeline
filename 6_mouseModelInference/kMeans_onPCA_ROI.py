import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import os
import copy
import shutil
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kMeansPCA_ROI(config, patchType):
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    PCA_DIR = config['directories']['PCA_OUTPUTS_ROI_DIR']
    
    saveDir = os.path.join(BASE_DIR,'kMeans_Outputs_ROI')
    
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    
    config['directories']['KMEANS_OUTPUT_ROI_DIR'] = saveDir
    
    if patchType == 'Involved':
        K_MEANS_MODEL = './6_mouseModelInference/kmeans_model/kmeans_model_k4.pkl'
        k = 4
    
    elif patchType == 'Uninvolved':
        K_MEANS_MODEL = './6_mouseModelInference/kmeans_model_UNinvolved/kMeanssave_onPCA_k3_UninvolvedwOverlaps_255Comps_Uninvolved.pkl'
        k = 3
        
    df = pd.read_csv(os.path.join(PCA_DIR,[a for a in os.listdir(PCA_DIR) if a.endswith('testData_' + str(patchType) + '.csv')][0]))
        
    values = df.drop(['fns'],axis=1)
    arrayvalues = np.array(values)

    labels = df[['fns']]
    
    kmeans = pickle.load(open(K_MEANS_MODEL,'rb'))
    pred_y = kmeans.predict(arrayvalues)
    
    output_df = copy.deepcopy(labels)

    output_df.columns = ['fn']
    output_df['Predictions'] = pred_y
    
    output_df.to_csv(os.path.join(saveDir,'kmeansClusters_onPCA_k' + str(k) + '_' + str(patchType) + '_ROI_wOverlaps_test.csv'),header=None,index=False)

    return config
