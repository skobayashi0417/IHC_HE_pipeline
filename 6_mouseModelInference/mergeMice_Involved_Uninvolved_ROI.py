import numpy as np
import pandas as pd
import os
import copy
import pickle
import shutil
import csv

def aggregate_to_directory(OVERLAP_PATCHES_DIR,DEST_PATCHES_DIR,df):
    for index,row in df.iterrows():
        # initial and overlap patches have slightly different naming schemes...
        #sample = str(('_').join(str(row[4][1:]).split('_')[0:3]))
        sample = str(row[4][1:]).split('_')[0]
        ROI = str(row[4][1:]).split('_')[1]
        fn = str(row[4][1:])

        src = os.path.join(os.path.join(os.path.join(OVERLAP_PATCHES_DIR,sample),ROI),fn)
            
        dest = os.path.join(DEST_PATCHES_DIR,fn)
        shutil.copy(src,dest)
        

def merge_csvs(DEST_DIR,PRED_DIR, target):
    trigger = 0
    
    samples = [s for s in os.listdir(PRED_DIR) if not s.startswith('.')]
    
    for sample in samples:
        sampleDir = os.path.join(PRED_DIR,sample)
        
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        for ROI in ROIS:
            csvPath = os.path.join(os.path.join(sampleDir,ROI),'adjustlabel_predicts.csv')
        
            if target == 'Involved':
                ID = 2
            
            elif target == 'Uninvolved':
                ID = 1
        
            if trigger == 0:
                mergedDF = pd.read_csv(csvPath,header=None)
                mergedDF = mergedDF[mergedDF[3]==ID]
                #print(mergedDF.head)
                trigger += 1
            else:
                df = pd.read_csv(csvPath,header=None)
                df = df[df[3]==ID]
                #print(df.head)
                mergedDF = pd.concat([mergedDF,df])
    
    output_file = os.path.join(DEST_DIR,'merged_ROI_' + str(target) + '.csv')
    
    mergedDF.to_csv(output_file,header=None,index=False)
    
    return mergedDF

def mergeMice_ROI(config):
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    
    PRED_DIR = config['directories']['PREDICTIONS_ROI_DIR']
    
    OVERLAP_PATCHES_DIR = config['directories']['extractedPatches_HE_sf8_wOverlaps_ROI']
    
    DEST_DIR = os.path.join(BASE_DIR,'Involved_UninvolvedPatches_ROI')
    if not os.path.exists(DEST_DIR):
        os.mkdir(DEST_DIR)
    
    INVOLVED_PATCHES_DIR = os.path.join(DEST_DIR,'involvedPatches_ROI_wOverlaps')
    UNINVOLVED_PATCHES_DIR = os.path.join(DEST_DIR,'UNinvolvedPatches_ROI_wOverlaps')
    
    if not os.path.exists(INVOLVED_PATCHES_DIR):
        os.mkdir(INVOLVED_PATCHES_DIR)

    if not os.path.exists(UNINVOLVED_PATCHES_DIR):
        os.mkdir(UNINVOLVED_PATCHES_DIR)

    involved_df = merge_csvs(DEST_DIR,PRED_DIR,'Involved')
    uninvolved_df = merge_csvs(DEST_DIR,PRED_DIR,'Uninvolved')
    
    aggregate_to_directory(OVERLAP_PATCHES_DIR,INVOLVED_PATCHES_DIR,involved_df)
    aggregate_to_directory(OVERLAP_PATCHES_DIR,UNINVOLVED_PATCHES_DIR,uninvolved_df)

    config['directories']['INVOLVED_PATCHES_ROI_DIR'] = INVOLVED_PATCHES_DIR
    config['directories']['UNINVOLVED_PATCHES_ROI_DIR'] = UNINVOLVED_PATCHES_DIR
    config['directories']['GEN_PATCHES_ROI_DIR'] = DEST_DIR
    
    return config
