import os
import numpy
import PIL
import csv
import openslide
import shutil
from PIL import Image
from autofilter import *

def extractPatches(os_WSI, WSI, coords, PATCH_SIZE, sample_dest_dir, WSI_EXTENSION):

    
    for coord in coords:
        newTile = os_WSI.read_region((coord[0],coord[1]), 0, (PATCH_SIZE,PATCH_SIZE)).convert("RGB")
        
        savePath = os.path.join(sample_dest_dir,str(WSI) + '_%dX_%dY_w%d_h%d.png' % (int(coord[0]), int(coord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
        
        newTile.save(savePath)
        
        newTile.close()
            
def patchExtraction_firstLevel_ROI(config):
    Image.MAX_IMAGE_PIXELS = None
    
    BASE_DIR = config['directories']['BASE_DIR']
    PATCH_SIZE = config['PatchInfo']['PATCH_SIZE_SF2']
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']
    SECOND_LEVEL_PATCHES_DIR = config['directories']['bySamplePatches_ROI_FIXED_sf8']
    #ROI_DIR = config['directories']['ROI_postReg']
    
    ROI_DIR_POSTREG = config['directories']['ROI_postReg']
    ROI_DIR_PREREG = config['directories']['preRegROI_registrationOutputs']
    
    PATCH_DEST_DIR = os.path.join(BASE_DIR,'extractedPatches_ROI_FIXED_sf2')
    if not os.path.exists(PATCH_DEST_DIR):
        os.mkdir(PATCH_DEST_DIR)
    
    bySampleDir = os.path.join(PATCH_DEST_DIR,'bySample')
    if not os.path.exists(bySampleDir):
        os.mkdir(bySampleDir)
    
    filtered_bySampleDir = os.path.join(PATCH_DEST_DIR,'filteredOut_bySample')
    if not os.path.exists(filtered_bySampleDir):
        os.mkdir(filtered_bySampleDir)
    
    config['directories']['bySamplePatches_ROI_FIXED_sf2'] = bySampleDir
    
    ### FIRST DO POST_REG SAMPLES
    samples = [a for a in os.listdir(ROI_DIR_POSTREG) if not a.endswith('.csv')]
    #samples = [s[:-len('_scaledFactor8')] for s in samples]
    tot_num = len(samples)
    
    counter = 1
    for sample in samples:
        sample_ROI_dir = os.path.join(ROI_DIR_POSTREG,sample)
        samplePatchesDir = os.path.join(SECOND_LEVEL_PATCHES_DIR,sample)
        
        sample_dest_dir = os.path.join(bySampleDir,sample)
        if not os.path.exists(sample_dest_dir):
            os.mkdir(sample_dest_dir)
            
        ROIS = [r for r in os.listdir(sample_ROI_dir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            ROI_DIR = os.path.join(sample_ROI_dir,ROI)
            ROI_DEST_DIR = os.path.join(sample_dest_dir,ROI)
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
        
            ims = [w for w in os.listdir(ROI_DIR) if w.endswith('.tif')]
            FIXED_WSI = [z for z in ims if str(z).split('.')[0].split('_')[-1]=='1'][0]

            #print('Extracting patches from sample %s ROI %s.. ------ %d out of %d total postreg samples.' % (FIXED_WSI,ROI,counter,tot_num))
            print('Extracting sf2 patches from FIXED image (%s) for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total postreg samples.' % (FIXED_WSI,sample,ROI,ROICOUNTER,len(ROIS),counter,tot_num))
            os_WSI = openslide.open_slide(os.path.join(ROI_DIR,FIXED_WSI))
        
            patches = [a for a in os.listdir(os.path.join(samplePatchesDir,ROI)) if a.endswith('.png')]
            coords = [(int(str(patch).split('_')[-4][:-1])*4,int(str(patch).split('_')[-3][:-1])*4) for patch in patches]
        
            extractPatches(os_WSI, FIXED_WSI[:-len('.tif')], coords, PATCH_SIZE, ROI_DEST_DIR, WSI_EXTENSION)
            
            ROICOUNTER += 1
        
        counter += 1
    
    ### NEXT DO PRE_REG SAMPLES
    samples = [a for a in os.listdir(ROI_DIR_PREREG)]
    #samples = [s[:-len('_scaledFactor8')] for s in samples]
    tot_num = len(samples)
    
    counter = 1
    for sample in samples:
        sample_ROI_dir = os.path.join(ROI_DIR_PREREG,sample)
        samplePatchesDir = os.path.join(SECOND_LEVEL_PATCHES_DIR,sample)
        
        sample_dest_dir = os.path.join(bySampleDir,sample)
        if not os.path.exists(sample_dest_dir):
            os.mkdir(sample_dest_dir)
            
        ROIS = [r for r in os.listdir(sample_ROI_dir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            ROI_DIR = os.path.join(sample_ROI_dir,ROI)
            
            ROI_DEST_DIR = os.path.join(sample_dest_dir,ROI)
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
        
            ims = [w for w in os.listdir(ROI_DIR) if w.endswith('.tif')]
            FIXED_WSI = [z for z in ims if str(z).split('.')[0].split('_')[-1]=='1'][0]

            #print('Extracting patches from sample  %s ROI %s.. ------ %d out of %d total samples.' % (FIXED_WSI,ROI,counter,tot_num))
            print('Extracting sf2 patches from FIXED_HE_1 image for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total prereg samples.' % (sample,ROI,ROICOUNTER,len(ROIS),counter,tot_num))
    
            os_WSI = openslide.open_slide(os.path.join(ROI_DIR,FIXED_WSI))
        
            patches = [a for a in os.listdir(os.path.join(samplePatchesDir,ROI)) if a.endswith('.png')]
            coords = [(int(str(patch).split('_')[-4][:-1])*4,int(str(patch).split('_')[-3][:-1])*4) for patch in patches]
        
            extractPatches(os_WSI, FIXED_WSI[:-len('.tif')], coords, PATCH_SIZE, ROI_DEST_DIR, WSI_EXTENSION)
            ROICOUNTER += 1
        
        counter += 1
            
    return config
    
if __name__=='__main__':
    main()
