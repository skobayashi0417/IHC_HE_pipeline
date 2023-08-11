import os
import PIL
from PIL import Image, ImageDraw, ImageOps
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy
import shutil
import openslide

def extract_newX(fn):
    newX = str(fn).split('_')[-4][:-1]
    return newX

def extract_newY(fn):
    newY = str(fn).split('_')[-3][:-1]
    return newY

def prepare_dirs(sample,ROI):
    global BYSAMPLE_DIR
    
    sample_Dir = os.path.join(BYSAMPLE_DIR,sample)
    if not os.path.exists(sample_Dir):
        os.mkdir(sample_Dir)
    
    ROI_dest_dir = os.path.join(sample_Dir,ROI)
    if not os.path.exists(ROI_dest_dir):
        os.mkdir(ROI_dest_dir)
        
    PROB_MAP_DIR = os.path.join(ROI_dest_dir,'prob_maps')
    if not os.path.exists(PROB_MAP_DIR):
        os.mkdir(PROB_MAP_DIR)

    MASKS_DIR = os.path.join(ROI_dest_dir,'masks')
    if not os.path.exists(MASKS_DIR):
        os.mkdir(MASKS_DIR)

    return MASKS_DIR, PROB_MAP_DIR
    

def generateProbMaps_ROI(config):
    global BASE_DIR
    global BYSAMPLE_DIR
    
    PREDICTIONS_DIR = config['directories']['PREDICTIONS_ROI_DIR']
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']
    SCALED_ROII_DIR = config['directories']['SCALED_ROI_DIR_SF8']
    MESHMAPS_DIR = config['directories']['MESH_MAPS_ROI_DIR']
    
    ### Generate Aggregated Prob-Map and Masks Dirs ###
    DEST_DIR = os.path.join(config['directories']['INV_UNV_wIHC_BASE_DIR_ROI'], 'probMapsandMasks_ROI')
    if not os.path.exists(DEST_DIR):
        os.mkdir(DEST_DIR)
    
    AGG_MASK_DIR = os.path.join(DEST_DIR, 'all_masks')
    AGG_PROBMAPS_DIR = os.path.join(DEST_DIR, 'all_probmaps')
    BYSAMPLE_DIR = os.path.join(DEST_DIR, 'bySample')
    
    if not os.path.exists(AGG_MASK_DIR):
        os.mkdir(AGG_MASK_DIR)
    
    if not os.path.exists(AGG_PROBMAPS_DIR):
        os.mkdir(AGG_PROBMAPS_DIR)

    if not os.path.exists(BYSAMPLE_DIR):
        os.mkdir(BYSAMPLE_DIR)
    
    config['directories']['bySample_ROI_probmaps'] = BYSAMPLE_DIR
    config['directories']['probMaps_masks_ROI_base_dir'] = DEST_DIR
        
    ####################################################
                   
    samples = [d for d in os.listdir(config['directories']['PREDICTIONS_ROI_DIR'])]
    tot_num = len(samples)
    samplecounter = 1
    
    for sample in samples:
        samplePredPath = os.path.join(PREDICTIONS_DIR,sample)
        sample_ROI_sf8_DIR = os.path.join(SCALED_ROII_DIR,sample)
        sampleMeshMaps_dir = os.path.join(MESHMAPS_DIR,sample)
    
        sampleDir = os.path.join(config['directories']['PREDICTIONS_ROI_DIR'],sample)
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        
        for ROI in ROIS:
            print('Generating Prob Maps for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total samples.' % (sample,ROI,ROICOUNTER,len(ROIS),samplecounter,tot_num))
            
            # Prepare Dirs
            MASKS_DIR, PROB_MAP_DIR = prepare_dirs(sample,ROI)
            
            # Open predictions
            predictions_dfPath = os.path.join(os.path.join(samplePredPath,ROI),'adjustlabel_predicts.csv')
            predictions_df = pd.read_csv(predictions_dfPath,header=None)
            predictions_df.columns = ['healthyConf','InfConf','MaxConf','Predicted','FN']

            predictions_df = predictions_df.drop(columns=['healthyConf','InfConf'])

            predictions_df['newX'] = predictions_df['FN'].map(extract_newX)
            predictions_df['newY'] = predictions_df['FN'].map(extract_newY)
            
        
            # Open WSI and Get Dimensions
            cur_ROI_sf8_dir = os.path.join(sample_ROI_sf8_DIR,ROI)
            WSIs = [a for a in os.listdir(cur_ROI_sf8_dir) if str(a).endswith(WSI_EXTENSION)]
            FIXED = [os.path.join(cur_ROI_sf8_dir,z) for z in WSIs if str(z).split('_')[3]=='1'][0]
        
            im = openslide.open_slide(FIXED)
            im_width, im_height = im.dimensions
            
            WSI_file = im.read_region((0,0),0,(im_width,im_height)).convert("RGB")

            # Initialize Empty arrays
            prob_map = np.zeros((im_height,im_width), dtype=float)
            prob_map_healthy = np.zeros((im_height,im_width), dtype=float)
            div_mask = np.zeros((im_height,im_width), dtype=int)
            div_mask_healthy = np.zeros((im_height,im_width), dtype=int)


            # Fill in any probabilities > 50%

            tot_num = len(predictions_df)

            counter = 1
                
            edgeCounter = 0
                

            for index, row in predictions_df.iterrows():
                print('On row %d out of %d total' % (counter, tot_num))

                newX = int(row['newX'])
                newY = int(row['newY'])
                maxProb = float(row['MaxConf'])
                Prediction = int(row['Predicted'])
                if maxProb < 0.5:
                    continue
                else:
                    if Prediction == 2:
                        # Pathology
                        if newX == im_width or newY == im_height:
                            edgeCounter += 1
                            continue
                        elif newX+224 > im_width or newY+224 > im_height:
                            if newX+224 < im_width:
                                for i in range(newX, newX+224):
                                    for u in range(newY, im_height):
                                        prob_map[u,i] += maxProb
                                        div_mask[u,i] += 1
                            elif newY+224 < im_height:
                                for i in range(newX, im_width):
                                    for u in range(newY, newY+224):
                                        prob_map[u,i] += maxProb
                                        div_mask[u,i] += 1
                        else:
                            for i in range(newX, newX+224):
                                for u in range(newY, newY+224):
                                    prob_map[u,i] += maxProb
                                    div_mask[u,i] += 1

                    elif Prediction == 1:
                        # Healthy
                        if newX == im_width or newY == im_height:
                            edgeCounter += 1
                        elif newX + 224 > im_width or newY + 224 > im_height:
                            if newX + 224 < im_width:
                                for i in range(newX, newX+224):
                                    for u in range(newY, im_height):
                                        prob_map_healthy[u,i] += maxProb
                                        div_mask_healthy[u,i] += 1
                            elif newY + 224 < im_height:
                                for i in range(newX, im_width):
                                    for u in range(newY, newY+224):
                                        prob_map_healthy[u,i] += maxProb
                                        div_mask_healthy[u,i] += 1
                        else:
                            for i in range(newX, newX+224):
                                for u in range(newY, newY+224):
                                    prob_map_healthy[u,i] += maxProb
                                    div_mask_healthy[u,i] += 1
                    
                counter += 1
            print('Final Edge Count: %d' %(edgeCounter))

            # div_masks have 0 where no probs added so cannot divide. Just change these to 1s.
            div_mask[div_mask==0] = 1
            div_mask_healthy[div_mask_healthy==0] = 1
                
            # divide by div masks
            prob_map = np.divide(prob_map,div_mask)
            prob_map_healthy = np.divide(prob_map_healthy,div_mask_healthy)
            
            # read in relevant meshMap and change all nonZero values to 1s
            meshMap = Image.open(os.path.join(os.path.join(sampleMeshMaps_dir, ROI),'meshMap_' + sample + '_' + str(ROI) + '.tif'))
            meshMap = np.array(meshMap)
            meshMap = np.divide(meshMap,255)
            
            # now do elementwise multiplication with prob maps... areas with no Tissue (i.e. = 0 in meshMap) will now become 0 in probmaps
            prob_map = prob_map * meshMap
            prob_map_healthy = prob_map_healthy * meshMap

            # Generate Output Visualizations of Masks
            Image.fromarray(prob_map).save(os.path.join(MASKS_DIR,'prob_map_test_' + sample + '_' + str(ROI) + '.tif'))
            Image.fromarray(prob_map_healthy).save(os.path.join(MASKS_DIR,'prob_map_healthy_test_' + sample + '_' + str(ROI) + '.tif'))

            prob_map_mult = np.multiply(prob_map,255)
            prob_map_mult_healthy = np.multiply(prob_map_healthy,255)

            Image.fromarray(prob_map_mult).save(os.path.join(MASKS_DIR,'prob_map_mult_test_' + sample + '_' + str(ROI) + '.tif'))
            Image.fromarray(prob_map_mult_healthy).save(os.path.join(MASKS_DIR,'prob_map_mult_healthy_test_' + sample + '_' + str(ROI) + '.tif'))
            
            # Save Mult Masks to AllMasks Dir Too
            Image.fromarray(prob_map_mult).save(os.path.join(AGG_PROBMAPS_DIR,'prob_map_mult_test_' + sample + '_' + str(ROI) + '.tif'))
            Image.fromarray(prob_map_mult_healthy).save(os.path.join(AGG_PROBMAPS_DIR,'prob_map_mult_healthy_test_' + sample + '_' + str(ROI) + '.tif'))

            ### Generate Pathology Mask Overlays
            base = WSI_file
            base2 = copy.deepcopy(base)
            base3 = copy.deepcopy(base)
            im = Image.open(os.path.join(MASKS_DIR,'prob_map_mult_test_' + sample + '_' + str(ROI) + '.tif'))
            z = np.array(im)
                
            ### Renormalize 50-100% probabilities on 0-255 scale
            z=np.subtract(z,127.5)
            z[z<0] = 0
            z = np.divide(z,127.5)
            z = np.multiply(z,255)

            # deep copy for combo overlay later
            prob_map_cp = copy.deepcopy(z)
                
            ### Generate Red Maskh
            v = z.astype(np.uint8)
            G = np.zeros((im.height, im.width))
            B = np.zeros((im.height, im.width))
            c=np.stack([v,G,B],axis=-1)
            c = c.astype(np.uint8)
            n = Image.fromarray(c).convert("RGB")
            n.putalpha(100)
            base.paste(im=n, box=(0,0), mask=n)

            saveName = 'Overlayed_' + sample + '_' + str(ROI) + '_Inflamed_PixelMap.tif'
            base.save(os.path.join(PROB_MAP_DIR,saveName))
            shutil.copy(os.path.join(PROB_MAP_DIR,saveName),os.path.join(AGG_PROBMAPS_DIR,saveName))

            # Repeat process for healthy calls
            ### Generate Healthy Mask Overlays
            im = Image.open(os.path.join(MASKS_DIR,'prob_map_mult_healthy_test_' + sample + '_' + str(ROI) + '.tif'))
            z1 = np.array(im)
        
            ### Renormalize 50-100% probabilities on 0-255 scale
            z1=np.subtract(z1,127.5)
            z1[z1<0] = 0
            z1 = np.divide(z1,127.5)
            z1 = np.multiply(z1,255)

            # deep copy for combo overlay later
            prob_map_healthy_cp = copy.deepcopy(z1)
            
            ### Generate Green Mask
            v = z1.astype(np.uint8)
            R = np.zeros((im.height, im.width))
            B = np.zeros((im.height, im.width))
            c=np.stack([R,v,B],axis=-1)
            c = c.astype(np.uint8)
            n = Image.fromarray(c).convert("RGB")
            n.putalpha(100)
            base2.paste(im=n, box=(0,0), mask=n)

            saveName = 'Overlayed_' + sample + '_' + str(ROI) + '_Healthy_PixelMap.tif'
            base2.save(os.path.join(PROB_MAP_DIR,saveName))
            shutil.copy(os.path.join(PROB_MAP_DIR,saveName),os.path.join(AGG_PROBMAPS_DIR,saveName))

            ## generate combo

            for i in range(0,im_height):
                for u in range(0,im_width):
                    if prob_map_cp[i,u] > prob_map_healthy_cp[i,u]:
                        prob_map_healthy_cp[i,u] = 0
                    elif prob_map_cp[i,u] < prob_map_healthy_cp[i,u]:
                        prob_map_cp[i,u] = 0

            ### Generate Red Mask
            v = prob_map_cp.astype(np.uint8)
            G = np.zeros((im_height, im_width))
            B = np.zeros((im_height, im_width))
            c=np.stack([v,G,B],axis=-1)
            c = c.astype(np.uint8)
            n = Image.fromarray(c).convert("RGB")
            n.putalpha(75)
            base3.paste(im=n, box=(0,0), mask=n)
                
            ### Generate Green Mask
            v1 = prob_map_healthy_cp.astype(np.uint8)
            R1 = np.zeros((im_height, im_width))
            B1 = np.zeros((im_height, im_width))
            c1=np.stack([R1,v1,B1],axis=-1)
            c1 = c1.astype(np.uint8)
            n1 = Image.fromarray(c1).convert("RGB")
            n1.putalpha(75)
            base3.paste(im=n1, box=(0,0), mask=n1)
            
            saveName = 'Overlayed_' + sample + '_' + str(ROI) + '_HealthyInf_PixelMap.tif'
            base3.save(os.path.join(PROB_MAP_DIR,saveName))
            shutil.copy(os.path.join(PROB_MAP_DIR,saveName),os.path.join(AGG_PROBMAPS_DIR,saveName))
        
        samplecounter+=1

    return config
    
if __name__=='__main__':
    main()
