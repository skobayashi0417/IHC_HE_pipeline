import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

### https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/ ###

def determine_n_comp(scaled_data, variance, ID):
    pca = PCA(n_components = float(variance))
    pca.fit(scaled_data)
    reduced = pca.transform(scaled_data)
    
    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()
    xi = np.arange(1, reduced.shape[1]+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, reduced.shape[1], step=5),rotation='vertical') #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()
    fig = ax.get_figure()
    
    saveName = 'n_Comp_Check_' + str(ID) + '.png'
    fig.savefig(saveName)
    
def performPCA_ROI(config,patchType):
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    save_dir = os.path.join(BASE_DIR,'PCA_outputs_ROI')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    config['directories']['PCA_OUTPUTS_ROI_DIR'] = save_dir
    
    if patchType == 'Involved':
        RN_FEATURES_DIR = config['directories']['INVOLVED_ROI_RN_EXTRACTED_FEATURES_DIR']
        
        # Path to archived mouse cohort to fit PCA on
        #PCA_TRAIN_CSV = './7_mouseModelInference/archivedMouseCohort_InvolvedPatches_RN_extracted_features/involved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv'
        PCA_TRAIN_CSV = './6_mouseModelInference/archivedMouseCohort_InvolvedPatches_RN_extracted_features/involved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv'
        PCA_OPT_NUM_COMPONENTS = 250 ## determined on archived mouse cohort
    
    elif patchType == 'Uninvolved':
        RN_FEATURES_DIR = config['directories']['UNINVOLVED_ROI_RN_EXTRACTED_FEATURES_DIR']
        
        # Path to archived mouse cohort to fit PCA on
        #PCA_TRAIN_CSV = './7_mouseModelInference/archivedMouseCohort_InvolvedPatches_RN_extracted_features/involved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv'
        PCA_TRAIN_CSV = './6_mouseModelInference/archivedMouseCohort_InvolvedPatches_RN_extracted_features/UNinvolved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv'
        PCA_OPT_NUM_COMPONENTS = 255 ## determined on archived mouse cohort
    
    scaler = MinMaxScaler()
    scaler_test = MinMaxScaler()
    
    # Inference cohort RN extracted features from earlier in pipeline
    test_csv = pd.read_csv(os.path.join(RN_FEATURES_DIR,[z for z in os.listdir(RN_FEATURES_DIR) if z.endswith('_ROI_wOverlap_RN_extractedFeatures.csv')][0]),header=None)
    test_fns = list(test_csv[0])

    test_df = test_csv.drop([0],axis=1)

    rescaled_test_data = scaler_test.fit_transform(test_df)
    
    train_df = pd.read_csv(PCA_TRAIN_CSV,header=None)
    fns = list(train_df[0])
    train_df = train_df.drop([0,1,2],axis=1)
        
    rescaled_data = scaler.fit_transform(train_df)
        
    #initialize PCA separately on test and training data (for checking)
    pca = PCA(n_components=PCA_OPT_NUM_COMPONENTS)
    pca_test = PCA(n_components=PCA_OPT_NUM_COMPONENTS)
        
    # fit pca_test on training data
    pca_test.fit(rescaled_data)
        
    # transform test data
    PCs_test = pca_test.transform(rescaled_test_data)
        
    # save test data PCs
    writeVariance_test = open('PCAVariance_' + 'prospectiveCohort_testData_' + str(patchType) + '.txt', 'w')
    writeVariance_test.write(str(pca_test.explained_variance_ratio_))
    writeVariance_test.close()
        
    cols = ['PC'+str(a) for a in list(range(0,PCA_OPT_NUM_COMPONENTS))]
        
    principalDF_test = pd.DataFrame(data = PCs_test, columns = cols)
    principalDF_test['fns'] = test_fns
        
    saveName_test = 'PCA_'+ 'propectiveCohort_variance_testData_' + str(patchType) + '.csv'
    principalDF_test.to_csv(os.path.join(save_dir,saveName_test),index=False)
        
    return config
