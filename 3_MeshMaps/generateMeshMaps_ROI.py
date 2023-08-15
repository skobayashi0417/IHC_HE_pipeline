import os
import PIL
from PIL import Image, ImageDraw, ImageOps
import shutil
import pandas as pd
import numpy as np

def rgb(value,classification,minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    halfway = (minimum+maximum)/2
    if classification == 'bg':
        if value >= halfway:
            g = 1 + int(180 - (value-halfway)*(180/halfway))
            r = 1 + int(180 - (value-halfway)*(180/halfway))
            b = 1 + int(180 - (value-halfway)*(180/halfway))
        elif value < halfway:
            r = 0
            g = 0
            b = 0
        #r = 10
        #g = 10
        #b = 10
    elif classification == 'muscle':
        if value >= halfway:
            r = 255
            g = int(180 - (value-halfway)*(180/halfway))
            b = int(180 - (value-halfway)*(180/halfway))
        elif value < halfway:
            r = 0
            g = 0
            b = 0
    elif classification == 'tissue':
        if value >= halfway:
            b = 255
            g = int(180 - (value-halfway)*(180/halfway))
            r = int(180 - (value-halfway)*(180/halfway))
        elif value < halfway:
            r = 0
            g = 0
            b = 0
    elif classification == 'submucosa':
        if value >= halfway:
            g = 255
            b = int(180 - (value-halfway)*(180/halfway))
            r = int(180 - (value-halfway)*(180/halfway))
        elif value < halfway:
            r = 0
            g = 0
            b = 0
    return r, g, b

def return_Xcoord(fn):
    Xcoord = str(fn).split('_')[-4][:-1]
    
    return Xcoord

def return_Ycoord(fn):
    Ycoord = str(fn).split('_')[-3][:-1]
    
    return Ycoord

def createMeshMaps_ROI(config):
    BASE_DIR = config['directories']['DEST_DIR']
    ROI_DIR = config['directories']['SCALED_ROI_DIR_SF8']
    MESH_PREDICTIONS_DIR = config['directories']['meshPREDICTIONS_DIR_ROI']
    PATCH_SIZE = config['PatchInfo']['meshPATCH_SIZE']
    
    DEST_DIR = os.path.join(BASE_DIR, 'meshMaps_ROI')
    if not os.path.exists(DEST_DIR):
        os.mkdir(DEST_DIR)
        
    DEST_DIR_WLA = os.path.join(BASE_DIR, 'meshMaps_ROI_wLA')
    if not os.path.exists(DEST_DIR_WLA):
        os.mkdir(DEST_DIR_WLA)
    
    config['directories']['MESH_MAPS_ROI_DIR'] = DEST_DIR
    config['directories']['LA_meshMaps_dir'] = DEST_DIR_WLA
    
    samples = [a for a in os.listdir(MESH_PREDICTIONS_DIR)]
    sampleCounter = 1
    for sample in samples:
        
        if not sample.startswith('.'):
            sample_dest_dir = os.path.join(DEST_DIR,sample)
            if not os.path.exists(sample_dest_dir):
                os.mkdir(sample_dest_dir)
            
            sample_dest_dir_WLA = os.path.join(DEST_DIR_WLA,sample)
            if not os.path.exists(sample_dest_dir_WLA):
                os.mkdir(sample_dest_dir_WLA)
        
        sampleDir = os.path.join(MESH_PREDICTIONS_DIR,sample)
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            print('Generating MeshMaps for FIXED_HE_1 image for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total samples.' % (sample,ROI,ROICOUNTER,len(ROIS),sampleCounter,len(samples)))
            cur_ROI_DIR = os.path.join(sampleDir,ROI)
            ROI_DEST_DIR = os.path.join(sample_dest_dir,ROI)
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
            
            ROI_DEST_DIR_WLA = os.path.join(sample_dest_dir_WLA,ROI)
            if not os.path.exists(ROI_DEST_DIR_WLA):
                os.mkdir(ROI_DEST_DIR_WLA)
                    
            pred_df = pd.read_csv(os.path.join(cur_ROI_DIR,'adjustlabel_predicts.csv'), header=None)
                    
            #print(pred_df)
                    
            pred_df = pred_df.drop(pred_df.columns[0:4],axis=1)
                    
            pred_df.columns=['conf','pred','fn']
                    
            pred_df['XCoord'] = pred_df['fn'].map(return_Xcoord)
            pred_df['YCoord'] = pred_df['fn'].map(return_Ycoord)
                    
            sample_ROI_tif_DIR = os.path.join(ROI_DIR,sample)
            fileName = [a for a in os.listdir(os.path.join(sample_ROI_tif_DIR,ROI)) if a.endswith('.tif')]
            fileName = [f for f in fileName if str(f).split('_')[3]=='1'][0]

            base = Image.open(os.path.join(os.path.join(sample_ROI_tif_DIR,ROI),fileName))
                    
            im_width, im_height = base.size
            base.close()

            meshMap_check = np.zeros((im_height,im_width), dtype=int)
            
            #print(meshMap_check.shape)
                    
            for index, row in pred_df.iterrows():
                topLeftX = int(row['XCoord'])
                topLeftY = int(row['YCoord'])
                        
                conf = float(row['conf'])
                prediction = str(row['pred'])
                        
                if conf < 0.5:
                    next
                else:
                    #print(topLeftX,topLeftY)
                    if prediction=='1':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_check[u,i] = 0
                    elif prediction=='2':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_check[u,i] = 0
                    elif prediction=='3':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_check[u,i] = 1
                    elif prediction=='4':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_check[u,i] = 1
                        
            saveName = 'meshMap_' + str(sample) + '_' + ROI + '.tif'
            meshMap_check = np.multiply(meshMap_check,255)
            Image.fromarray(meshMap_check.astype(np.uint8)).save(os.path.join(ROI_DEST_DIR,saveName))

            meshMap_Muscle = np.zeros((im_height,im_width), dtype=int)
                    
            for index, row in pred_df.iterrows():
                topLeftX = int(row['XCoord'])
                topLeftY = int(row['YCoord'])
                
                conf = float(row['conf'])
                prediction = str(row['pred'])
                        
                if conf < 0.5:
                    next
                        
                else:
                    if prediction=='1':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_Muscle[u,i] = 1
                    elif prediction=='2':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_Muscle[u,i] = 0
                    elif prediction=='3':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_Muscle[u,i] = 1
                    elif prediction=='4':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_Muscle[u,i] = 1
                        
            saveName = 'MusclemeshMap_' + str(sample) + '_' + ROI + '.tif'
            meshMap_Muscle = np.multiply(meshMap_Muscle,255)
            Image.fromarray(meshMap_Muscle.astype(np.uint8)).save(os.path.join(ROI_DEST_DIR,saveName))
            
            meshMap_lymphAgg_muscle_bg = np.zeros((im_height,im_width), dtype=int)
                    
            for index, row in pred_df.iterrows():
                topLeftX = int(row['XCoord'])
                topLeftY = int(row['YCoord'])
                
                conf = float(row['conf'])
                prediction = str(row['pred'])
                        
                if conf < 0.5:
                    next
                        
                else:
                    if prediction=='1':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg_muscle_bg[u,i] = 0
                    elif prediction=='2':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg_muscle_bg[u,i] = 0
                    elif prediction=='3':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg_muscle_bg[u,i] = 1
                    elif prediction=='4':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg_muscle_bg[u,i] = 1
                    elif prediction=='5':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg_muscle_bg[u,i] = 0
                        
            saveName = 'LA_muscle_bg_meshMap_' + str(sample) + '_' + ROI + '.tif'
            meshMap_lymphAgg_muscle_bg = np.multiply(meshMap_lymphAgg_muscle_bg,255)
            Image.fromarray(meshMap_lymphAgg_muscle_bg.astype(np.uint8)).save(os.path.join(ROI_DEST_DIR_WLA,saveName))
            
            meshMap_lymphAgg = np.zeros((im_height,im_width), dtype=int)
                    
            for index, row in pred_df.iterrows():
                topLeftX = int(row['XCoord'])
                topLeftY = int(row['YCoord'])
                
                conf = float(row['conf'])
                prediction = str(row['pred'])
                        
                if conf < 0.5:
                    next
                        
                else:
                    if prediction=='1':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg[u,i] = 1
                    elif prediction=='2':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg[u,i] = 1
                    elif prediction=='3':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg[u,i] = 1
                    elif prediction=='4':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg[u,i] = 1
                    elif prediction=='5':
                        for i in range(topLeftX,topLeftX+PATCH_SIZE):
                            for u in range(topLeftY,topLeftY+PATCH_SIZE):
                                if i<im_width and u<im_height:
                                    meshMap_lymphAgg[u,i] = 0
                        
            saveName = 'LAmeshMap_' + str(sample) + '_' + ROI + '.tif'
            meshMap_lymphAgg = np.multiply(meshMap_lymphAgg,255)
            Image.fromarray(meshMap_lymphAgg.astype(np.uint8)).save(os.path.join(ROI_DEST_DIR_WLA,saveName))
            
            ROICOUNTER += 1
        sampleCounter += 1
        
    return config
