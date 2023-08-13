import os
import json
import sys
sys.path.insert(0,'./1_PatchExtraction')
sys.path.insert(0,'./2_Inference')
sys.path.insert(0,'./5_ProbMaps')
sys.path.insert(0,'./6_mouseModelInference')
sys.path.insert(0,'./7_combineHE_IHC')
from generateJSON import *
from extractOverlapPatches_ROI import *
from largerPatchPredictions_ROI import *
from generateProbMaps_ROI import *
from mergeMice_Involved_Uninvolved_ROI import *
from RN_FeatureExtractor_ROI import *
from conductPCA_ROI import *
from kMeans_onPCA_ROI import *
from merge_Uninvolved_InvolvedkMeansCounts_ROI import *
from getProps_ROI import *
from LDA_inference_ROI import *
from sortkMeansClusters import *
from combine_HE_IHC_Outputs_ROI import *
from generate_HE_IHC_CSVs_andGraphs import *

def prepare_directories(config):
    INV_UNV_wIHC_BASE_DIR = os.path.join(config['directories']['BASE_DIR'],'INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI')
    if not os.path.exists(INV_UNV_wIHC_BASE_DIR):
        os.mkdir(INV_UNV_wIHC_BASE_DIR)
    
    config['directories']['INV_UNV_wIHC_BASE_DIR_ROI'] = INV_UNV_wIHC_BASE_DIR
    
    return config

def runPipeline(config):
    ## extract HE overlap patches at sf8
    print('Extracting overlapping patches...')
    #config = extract_overlapPatches_ROI(config)
    
    ## perform HE sf8 Involved versus Uninvolved predictions
    print('Performing Involved versus Uninvolved predictions...')
    #config = perform_predictions_ROI(config)
    
    ## Generate HE SF8 ProbMaps
    print('Generating Prob Maps...')
    #config = generateProbMaps_ROI(config)

    ### Gather Involved and Uninvolved Patches...
    print('Gathering Involved and Uninvolved Patches...')
    #config = mergeMice_ROI(config)

    ### Perform RN Feature Extraction
    print('Performing RN Feature Extraction on Involved Patches...')
    #config = RN_FeatureExtraction_ROI(config, 'Involved')
     
    print('Performing RN Feature Extraction on UNinvolved Patches...')
    #config = RN_FeatureExtraction_ROI(config, 'Uninvolved')
    
    config['directories']['extractedPatches_HE_sf8_wOverlaps_ROI'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/extractedPatches_HE_sf8_wOverlaps_ROI/bySample'
    
    config['directories']['PREDICTIONS_ROI_DIR'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/predictions_wOverlaps_ROI'
     
    config['directories']['bySample_ROI_probmaps'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/probMapsandMasks_ROI/bySample'
     
    config['directories']['probMaps_masks_ROI_base_dir'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/probMapsandMasks_ROI'
     
    config['directories']['INVOLVED_PATCHES_ROI_DIR'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/Involved_UninvolvedPatches_ROI/involvedPatches_ROI_wOverlaps'
    config['directories']['UNINVOLVED_PATCHES_ROI_DIR'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/Involved_UninvolvedPatches_ROI/UNinvolvedPatches_ROI_wOverlaps'
    config['directories']['GEN_PATCHES_ROI_DIR'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/Involved_UninvolvedPatches_ROI/'
    
    config['directories']['INVOLVED_ROI_RN_EXTRACTED_FEATURES_DIR'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/involved_ROI_patch_RN_FeatureExtraction'
    config['directories']['UNINVOLVED_ROI_RN_EXTRACTED_FEATURES_DIR'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/UNinvolved_ROI_patch_RN_FeatureExtraction'
    config['directories']['KMEANS_OUTPUT_ROI_DIR'] = '/data01/shared/skobayashi/github_test/INV_UNV_wIHC_Outputs_wUNINVkmeans_ROI/kMeans_Outputs_ROI'
    ### Perform PCA
    print('Conducting PCA on Involved Patches...')
    #config = performPCA_ROI(config, 'Involved')
    
    print('Conducting PCA on UNinvolved Patches...')
    #config = performPCA_ROI(config, 'Uninvolved')

    ### Perform kMeans
    print('performing kMeans on Involved Patches...')
    #config = kMeansPCA_ROI(config, 'Involved')
    
    print('performing kMeans on UNinvolved Patches...')
    #config = kMeansPCA_ROI(config, 'Uninvolved')

    ### Merge InvolvedUninvolved Counts
    print('Merging Involved and Uninvolved Counts...')
    #config = mergeCounts_ROI(config)

    ### Generate Proportions
    print('Generating Uninvolved and Involved k-mean Class Proportions...')
    #config = generateProps_ROI(config)
        
    ### Perform Mouse Model Inference
    print('Performing LDA mouse model inference...')
    #config = LDA_infer_ROI(config)
        
    ### Sort kMeans Clusters to Visualize
    print('Sorting kMeans Clusters...')
    #sortClusters(config)
    
     ### Generate IHC HE agg DFs and graphs
    print('Generating IHC HE agg dataframes and graph Outputs...')
    IHC_HE_aggregatedDF_graphs_generation(config)
    
    ### Generate IHC HE Outputs
    print('Generating visual IHC HE overlay Outputs...')
    combine_HE_IHC_ROI(config)

    
if __name__ == '__main__':
    generate_JSON()
    
    config = json.load(open('config_part3.json'))

    config = prepare_directories(config)
    
    runPipeline(config)
