import os
import numpy
import PIL
import csv
import openslide
import shutil
from PIL import Image
from autofilter import *

def expandPts(tupledOrigCoords):
    expandedCoordsList = []

    x = tupledOrigCoords[0]
    y = tupledOrigCoords[1]
    for x_c in range(x-220,x+221,20):
        if x_c < 0:
            next
        else:
            for y_c in range(y-220,y+221,20):
                if y_c < 0:
                    next
                else:
                    coords = (x_c,y_c)
                    #print(coords)
                    expandedCoordsList.append(coords)

    final = [a for a in expandedCoordsList if a != tupledOrigCoords]
    
    return final

def extractPatches(im, WSI,coords, PATCH_SIZE, sampleDir, WSI_EXTENSION,IHC_target):

    target_dest = os.path.join(sampleDir,IHC_target)
    if not os.path.exists(target_dest):
        os.mkdir(target_dest)
    
    for coord in coords:
        newTile = im.read_region((coord[0],coord[1]), 0, (PATCH_SIZE,PATCH_SIZE)).convert("RGB")
        savePath = os.path.join(target_dest,str(WSI)+'_' + IHC_target + '_%dX_%dY_w%d_h%d.png' % (int(coord[0]), int(coord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
        
        newTile.save(savePath)
        
        newTile.close()
            
def patchExtraction_Moving_ROI(config):
    Image.MAX_IMAGE_PIXELS = None
    
    BASE_DIR = config['directories']['DEST_DIR']
    PATCH_SIZE_SF8 = config['PatchInfo']['PATCH_SIZE_SF8']
    PATCH_SIZE_SF2 = config['PatchInfo']['PATCH_SIZE_SF2']
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']
    
    HE_PATCHES_DIR_SF8 = config['directories']['bySamplePatches_ROI_FIXED_sf8']
    HE_PATCHES_DIR_SF2 = config['directories']['bySamplePatches_ROI_FIXED_sf2']
    
    ROI_DIR_SF8 = config['directories']['SCALED_ROI_DIR_SF8']
    #ROI_DIR = config['directories']['ROI_postReg']
    ROI_DIR_POSTREG = config['directories']['ROI_postReg']
    ROI_DIR_PREREG = config['directories']['preRegROI_registrationOutputs']
    
    PATCH_DEST_DIR_MOVING_SF8 = os.path.join(BASE_DIR,'extractedPatches_ROI_SF8_MOVING')
    PATCH_DEST_DIR_MOVING_SF2 = os.path.join(BASE_DIR,'extractedPatches_ROI_SF2_MOVING')
    
    if not os.path.exists(PATCH_DEST_DIR_MOVING_SF8):
        os.mkdir(PATCH_DEST_DIR_MOVING_SF8)

    if not os.path.exists(PATCH_DEST_DIR_MOVING_SF2):
        os.mkdir(PATCH_DEST_DIR_MOVING_SF2)

    bySampleDir_SF8 = os.path.join(PATCH_DEST_DIR_MOVING_SF8,'bySample')
    if not os.path.exists(bySampleDir_SF8):
        os.mkdir(bySampleDir_SF8)
    
    bySampleDir_SF2 = os.path.join(PATCH_DEST_DIR_MOVING_SF2,'bySample')
    if not os.path.exists(bySampleDir_SF2):
        os.mkdir(bySampleDir_SF2)
    
    preRegsamples = [s for s in os.listdir(ROI_DIR_PREREG)]
    
    samples = [a for a in os.listdir(HE_PATCHES_DIR_SF2)]
    tot_num = len(samples)
    counter = 1
    for sample in samples:
        curSampleDir = os.path.join(HE_PATCHES_DIR_SF2,sample)
        sample_dest = os.path.join(bySampleDir_SF2,sample)
        if not os.path.exists(sample_dest):
            os.mkdir(sample_dest)
        
        if sample in preRegsamples:
            srcDir = ROI_DIR_PREREG
        else:
            srcDir = ROI_DIR_POSTREG
        sampleSrcDir = os.path.join(srcDir,sample)
        ROIS = [r for r in os.listdir(curSampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            srcSampleDir = os.path.join(sampleSrcDir,ROI)
            ROI_DEST_DIR = os.path.join(sample_dest,ROI)
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
            patches = [p for p in os.listdir(os.path.join(curSampleDir,ROI)) if p.endswith('.png')]
        
            SF2_Samples_coords = [(int(str(a).split('_')[-4][:-1]),int(str(a).split('_')[-3][:-1])) for a in patches]
        
            #sampleNumber = str(IHC_SF2_Sample).split('_')[1].split('-')[0]
            WSIs = [a for a in os.listdir(srcSampleDir) if str(a).endswith('.tif')]
            MOVING_SLIDES = [z for z in WSIs if str(z).split('.')[0].split('_')[-1]!='1']
        
            MOVINGCOUNTER = 1
            for MOVING in MOVING_SLIDES:
                if '_HE_' in MOVING:
                    strSplit = str(MOVING).split('.')[0].split('_')
                    IHC_target = ('').join(strSplit[2:4])
                else:
                    IHC_target = str(MOVING).split('_')[2]
                print('Extracting moving sf2 patches from sample %s TARGET %s: %s.. (%d/%d ROIs)(%d/%d moving Slides) ------ %d out of %d total samples.' % (sample,IHC_target,ROI,ROICOUNTER,len(ROIS),MOVINGCOUNTER,len(MOVING_SLIDES),counter,tot_num))
                os_WSI = openslide.open_slide(os.path.join(srcSampleDir,MOVING))
                extractPatches(os_WSI, sample, SF2_Samples_coords, PATCH_SIZE_SF2, ROI_DEST_DIR, WSI_EXTENSION,IHC_target)
                
                MOVINGCOUNTER += 1
            
            ROICOUNTER += 1
        
        counter += 1
        
    SF8_Samples = [a for a in os.listdir(HE_PATCHES_DIR_SF8)]
    tot_num = len(SF8_Samples)
    counter = 1
    for SF8_Sample in SF8_Samples:
        curSampleDir = os.path.join(HE_PATCHES_DIR_SF8,SF8_Sample)
        srcSampleDir = os.path.join(ROI_DIR_SF8,SF8_Sample)
        
        sample_dest = os.path.join(bySampleDir_SF8,SF8_Sample)
        if not os.path.exists(sample_dest):
            os.mkdir(sample_dest)
        
        ROIS = [r for r in os.listdir(curSampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            ROI_srcSampleDir = os.path.join(srcSampleDir,ROI)
            ROI_DEST_DIR = os.path.join(sample_dest,ROI)
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
            
            patches = [p for p in os.listdir(os.path.join(curSampleDir,ROI)) if p.endswith('.png')]
            SF8_Samples_coords = [(int(str(a).split('_')[-4][:-1]),int(str(a).split('_')[-3][:-1])) for a in patches]
            
            WSIs = [a for a in os.listdir(srcSampleDir) if str(a).endswith('.tif')]
            MOVING_SLIDES = [z for z in WSIs if str(z).split('_')[3]!='1']
            MOVINGCOUNTER = 1
            for MOVING in MOVING_SLIDES:
                if '_HE_' in MOVING:
                    strSplit = str(MOVING).split('.')[0].split('_')
                    IHC_target = ('').join(strSplit[2:4])
                else:
                    IHC_target = str(MOVING).split('_')[2]
                
                print('Extracting moving sf8 patches from sample %s TARGET %s: %s.. (%d/%d ROIs)(%d/%d moving Slides) ------ %d out of %d total samples.' % (sample,IHC_target,ROI,ROICOUNTER,len(ROIS),MOVINGCOUNTER,len(MOVING_SLIDES),counter,tot_num))
                os_WSI = openslide.open_slide(os.path.join(srcSampleDir,MOVING))
                extractPatches(os_WSI, SF8_Sample, SF8_Samples_coords, PATCH_SIZE_SF8, ROI_DEST_DIR, WSI_EXTENSION,IHC_target)
                
                MOVINGCOUNTER += 1
            ROICOUNTER += 1
        
        counter += 1
            
    return config
    
if __name__=='__main__':
    main()
