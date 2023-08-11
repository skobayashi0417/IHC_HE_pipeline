import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
import pandas as pd
import json

def return_mask(Classification, numPositive):
    if Classification == 'Involved':
        r,g,b = 255, 0, 0
    elif Classification == 'Uninvolved':
        r,g,b = 0, 255, 0
        
    mask = Image.new('RGB', (224, 224), color = (0, 0, 0))
    inner = Image.new('RGB',(214,214),color=(r, g, b))
    
    mask.paste(im=inner, box=(5,5))
    
    font = ImageFont.truetype('/home/skobayashi/arial/arial.ttf', 100)
    maskDrawer = ImageDraw.Draw(mask)
    
    #maskDrawer.text((50, 100), numPositive, font = font, fill=(0, 0, 0))
    #maskDrawer.text((75, 90), numPositive, font = font, fill="#FFFF00")
    maskDrawer.text((75, 90), numPositive, font = font, fill="#000000")
    
    mask.putalpha(100)

    return mask

def return_sf2_fn_from_sf8(fn_sf8):
    Xcoord = str(int(str(fn_sf8).split('_')[-4][:-1])*4)
    Ycoord = str(int(str(fn_sf8).split('_')[-3][:-1])*4)
    
    split_list = str(fn_sf8).split('_')
    split_list[-4] = Xcoord + 'X'
    split_list[-3] = Ycoord + 'Y'
    split_list[-2] = 'w896'
    split_list[-1] = 'h896.png'
    
    return ('_').join(split_list[5:])
    
def generate_overlay(df, sample, ROI, target):
    global WSI_dir
    global DEST_DIR_OVERLAYS
    WSI_EXTENSION = '.tif'
    
    print('Now generating IHC count overlay...')
        
    sampleWSIDir = os.path.join(WSI_dir,sample)
    # load WSI
    WSI = Image.open([os.path.join(os.path.join(sampleWSIDir,ROI),w) for w in os.listdir(os.path.join(sampleWSIDir,ROI)) if w.endswith(WSI_EXTENSION) and str(w).split('_')[-2]=='1'][0])
    
    # just do this initially to make sure that the column names are preserved

    for index,row in df.iterrows():
        numPositive = str(row['targetCount'])
        Classification = str(row['Classification'])
        Xcoord = int(str(row['Xcoord']))
        Ycoord = int(str(row['Ycoord']))
        
        mask = return_mask(Classification, numPositive)
        WSI.paste(im=mask,box=(Xcoord,Ycoord),mask=mask)
        
    savename = 'IHCCounts_HE_Overlay_Sample_' + str(sample) + ROI + '_target_' + str(target) + '.tif'
    WSI.save(os.path.join(DEST_DIR_OVERLAYS,savename))
    
    ## REPEAT FOR PERC
    # load WSI
    WSI = Image.open([os.path.join(os.path.join(sampleWSIDir,ROI),w) for w in os.listdir(os.path.join(sampleWSIDir,ROI)) if w.endswith(WSI_EXTENSION) and str(w).split('_')[-2]=='1'][0])
    
    
    for index,row in df.iterrows():
        posPerc = str(row['targetPerc'])
        Classification = str(row['Classification'])
        Xcoord = int(str(row['Xcoord']))
        Ycoord = int(str(row['Ycoord']))
        
        mask = return_mask(Classification, posPerc)
        WSI.paste(im=mask,box=(Xcoord,Ycoord),mask=mask)
        
    savename = 'IHCCounts_HE_Overlay_Sample_' + str(sample) + '_' + ROI + '_target_' + str(target) + '_perPOSAREA.tif'
    sampleSaveDir = os.path.join(DEST_DIR_OVERLAYS,sample)
    if not os.path.exists(sampleSaveDir):
        os.mkdir(sampleSaveDir)
    saveDir = os.path.join(sampleSaveDir,ROI)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
        
    WSI.save(os.path.join(saveDir,savename))
        
        

def save_info_csv(saveList,saveName):
    global DEST_DIR_CSVs
    
    appendDict = {}
    savePath = os.path.join(DEST_DIR_CSVs,saveName)

    with open(savePath,'w') as csvfile:
        fieldnames = ['fn','target', 'Xcoord', 'Ycoord', 'targetCount', 'targetPerc', 'InvolvedProp', 'Classification']
        testwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        testwriter.writeheader()
        for i in range(len(saveList)):
            append = saveList[i]
            appendDict['fn']=append[0]
            appendDict['target']=append[1]
            appendDict['Xcoord']=append[2]
            appendDict['Ycoord']=append[3]
            appendDict['targetCount']=append[4]
            appendDict['targetPerc']=append[5]
            appendDict['InvolvedProp']=append[6]
            appendDict['Classification']=append[7]
            testwriter.writerow(appendDict)
    
    return os.path.join(DEST_DIR_CSVs,saveName)

def combine_HE_IHC_ROI(config):
    global DEST_DIR_CSVs
    global WSI_dir
    global DEST_DIR_OVERLAYS
    
    Image.MAX_IMAGE_PIXELS = None
    
    # play with this setting later
    InvolvedThreshold = .5
    PATCH_SIZE = 224
    
    #

    WSI_dir = config['directories']['SCALED_ROI_DIR_SF8']
    
    probMaps_dir = config['directories']['bySample_ROI_probmaps']
    
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']
    
    HE_DIR = config['directories']['extractedPatches_HE_sf8_wOverlaps_ROI']
    
    HE_DIR_INITIALPATCHES = config['directories']['bySamplePatches_ROI_FIXED_sf8']
    
    IHC_CSVS_COUNT_DIR = config['directories']['IHC_CSV_Counts_ROI_newDetect']
    
    DEST_DIR = os.path.join(config['directories']['INV_UNV_wIHC_BASE_DIR_ROI'],'IHC_HE_Combo_outputs')
    if not os.path.exists(DEST_DIR):
        os.mkdir(DEST_DIR)
        
    DEST_DIR_CSVs = os.path.join(DEST_DIR,'CSVs')
    DEST_DIR_OVERLAYS = os.path.join(DEST_DIR,'Combined_Overlays')
    if not os.path.exists(DEST_DIR_CSVs):
        os.mkdir(DEST_DIR_CSVs)
    if not os.path.exists(DEST_DIR_OVERLAYS):
        os.mkdir(DEST_DIR_OVERLAYS)
    #
    
    samples = [s for s in os.listdir(probMaps_dir)]
    counter = 1
    
    for sample in samples:
        print('On %s... %d out of %d'%(sample,counter,len(samples)))
        
        sample_HE_DIR_INITIALPATCHES = os.path.join(HE_DIR_INITIALPATCHES,sample)
        IHC_count_csvs = [a for a in os.listdir(IHC_CSVS_COUNT_DIR) if sample in a]
        sampleDir = os.path.join(probMaps_dir,sample)
        
        ROIs = [r for r in os.listdir(sampleDir) if 'ROI' in r]
    
        for ROI in ROIs:
            ROI_HE_initialPatches_dir = os.path.join(sample_HE_DIR_INITIALPATCHES,ROI)
        
            # grab relevant probMaps
            masksDir = os.path.join(os.path.join(sampleDir,ROI),'masks')
    
            healthyprobMap = Image.open([os.path.join(masksDir, p) for p in os.listdir(masksDir) if 'prob_map_mult_healthy_test' in p][0])
    
            pathprobMap = Image.open([os.path.join(masksDir, p) for p in os.listdir(masksDir) if 'prob_map_mult_test' in p][0])
    
            # convert probMaps to numpy
            healthyprobMap_np = np.array(healthyprobMap)
            pathprobMap_np = np.array(pathprobMap)
        
            # combine info so that everything pixel with higher confidence as healthy gets set to 0 in pathprobMap
            for i in range(0,pathprobMap_np.shape[0]):
                for u in range(0,pathprobMap_np.shape[1]):
                    if pathprobMap_np[i,u] > healthyprobMap_np[i,u]:
                        healthyprobMap_np[i,u] = 0
                    elif pathprobMap_np[i,u] < healthyprobMap_np[i,u]:
                        pathprobMap_np[i,u] = 0
            ## now the path prob maps should have numbers or 0s
            ## for now, set all positive values to 1
            ### LATER ON PLAY WITH INCORPORATING ACTUAL CONFIDENCES, for now, there should only be values with atleast 50% confidence ###
            pathprobMap_np[pathprobMap_np>0] = 1
            healthyprobMap_np[healthyprobMap_np>0] = 1
        
            # now we can refer to this to identify patches as involved or not
        
            ### Find the number of initla H&E extracted patches at SF8 for this sample (should equal number of IHC patches per target ###
            initial_HE = [p for p in os.listdir(ROI_HE_initialPatches_dir) if p.endswith('.png')]
            num_initial_HE = len(initial_HE)
        
            # load up IHC Counts CSV for this sample
            IHC_counts_df = pd.read_csv(os.path.join(IHC_CSVS_COUNT_DIR,[a for a in IHC_count_csvs if ROI in a][0]))
            
            # check length of IHC_Counts_df and whether it matches # HE patches
            assert len(IHC_counts_df) == num_initial_HE
        
            # get targets from column names
            nonTarget_col_names = ['Inv_kMeans_cluster','UNinv_kMeans_cluster','InvProp', 'Classification']
            
            targets = [z for z in list(IHC_counts_df.columns) if 'patch_fn' not in z]
            targets = [z for z in targets if z not in nonTarget_col_names]
            targets = [z for z in targets if '_percPosArea' not in z]
            
            # initialize separate dictionary keys will be filenames and targets...
            # filename values will be the SF8 HE filenamesin a list corresponding to IHC
            # target values will be a list of counts... in same order as HE filenames
        
            num_targets = len(targets)
            
            sampleDic = {}
            sampleDic['fn'] = []
            sampleDic['Xcoord'] = []
            sampleDic['Ycoord'] = []
            sampleDic['origfns'] = []
        
            for target in targets:
                sampleDic[target] = []
        
            for index, row in IHC_counts_df.iterrows():
                IHC_fn = row['patch_fn']
                newX = str(int(int(row['patch_fn'].split('_')[0][:-1])/4))
                newY = str(int(int(row['patch_fn'].split('_')[1][:-1])/4))
                sampleDic['fn'].append([a for a in initial_HE if newX + 'X_' + newY +'Y' in a][0])
                sampleDic['Xcoord'].append(newX)
                sampleDic['Ycoord'].append(newY)
                sampleDic['origfns'].append(IHC_fn)
                # iterate and add number of detected IHC per target
                for target in targets:
                    sampleDic[target].append((row[target],row[target+'_percPosArea']))

            # save sample dict TMP for debug
            with open(os.path.join(DEST_DIR,sample+ '_' + ROI +'_patchDict.txt'), 'w') as convert_file:
                convert_file.write(json.dumps(sampleDic))
        
            ## iterate sample Dictionary and create overlays
        
            appendList = []
        
            counter = 1
            
            # separate out list of patch fns, Xs, Ys
            fns = sampleDic['fn']
            Xs = sampleDic['Xcoord']
            Ys = sampleDic['Ycoord']
            origfns = sampleDic['origfns']

            CSVUpdateTrigger = 0
            # iterate the targets again and generate individual overlays for each
            for target in targets:
                appendList = []
            
                countsList = sampleDic[target]
                
                for i in range(len(fns)):
                    fn = fns[i]
                    Xcoord = Xs[i]
                    Ycoord = Ys[i]
                    targetCount = countsList[i][0]
                    targetPerc = countsList[i][1]

                    # load corresponding location on np patch
                    correspondingProbMapPatch = pathprobMap_np[int(Ycoord):int(Ycoord)+PATCH_SIZE, int(Xcoord):int(Xcoord)+PATCH_SIZE]
                    correspondingProbMapPatch_healthy = healthyprobMap_np[int(Ycoord):int(Ycoord)+PATCH_SIZE, int(Xcoord):int(Xcoord)+PATCH_SIZE]
                
                    # calculate percent of total tissue area that is Involved
                    InvolvedProp = (np.sum(correspondingProbMapPatch)/(PATCH_SIZE*PATCH_SIZE))/(np.sum(correspondingProbMapPatch)/(PATCH_SIZE*PATCH_SIZE) + np.sum(correspondingProbMapPatch_healthy)/(PATCH_SIZE*PATCH_SIZE))

                    # use threshold to make patch determination
                    if InvolvedProp >= InvolvedThreshold:
                        Classification = 'Involved'
                    elif InvolvedProp < InvolvedThreshold:
                        Classification = 'Uninvolved'
                    
                    # save per patch info for CSV logging later
                    appendList.append([return_sf2_fn_from_sf8(fn),str(target), Xcoord, Ycoord, targetCount, targetPerc, InvolvedProp, Classification])
                
                if CSVUpdateTrigger == 0:
                    # udpate the IHC count csvs... only need to do for one target per sample as the updated info (Involved Prop and Classification) are same for all targets since based on HE
                    toUpdate = [(a[0],a[6],a[7]) for a in appendList]
                    
                    # extract order of fns in IHC counts df
                    IHC_counts_fns = list(IHC_counts_df['patch_fn'])
                    
                    # sort toUpdate accordingly
                    print(toUpdate[0:5])
                    print(IHC_counts_fns[0:5])
                    sorted_toUpdate = sorted(toUpdate, key = lambda x: IHC_counts_fns.index(x[0]))
                    # update CSV
                    IHC_counts_df['InvProp'] = [a[1] for a in sorted_toUpdate]
                    IHC_counts_df['Classification'] = [a[2] for a in sorted_toUpdate]
                    
                    # save CSV
                    IHC_counts_df.to_csv(os.path.join(IHC_CSVS_COUNT_DIR,[a for a in IHC_count_csvs if ROI in a][0]),index=False)
                    
                
                csvPath = save_info_csv(appendList, str(sample)+'_' + ROI + '_' +str(target)+'_summaryData.csv')
                
                #csvPath = save_info_csv(appendList, '053_HE_IHC_combinedOutputs.csv')
        
                df = pd.read_csv(csvPath)
        
                generate_overlay(df, sample, ROI, target)
    counter += 1


if __name__ == '__main__':
    combine_HE_IHC(config)
