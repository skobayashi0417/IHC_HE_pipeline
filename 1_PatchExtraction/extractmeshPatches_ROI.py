import os
import numpy
import PIL
import csv
import openslide
import shutil
from PIL import Image

def extractPatches(im, possibleCoords, PATCH_SIZE, sampleDir, WSI, WSI_EXTENSION):
    
    for possibleCoord in possibleCoords:
        newTile = im.read_region((possibleCoord[0],possibleCoord[1]), 0, (PATCH_SIZE,PATCH_SIZE)).convert("RGB")

        savePath = os.path.join(sampleDir,str(WSI)[:-(len(WSI_EXTENSION))] + '_%dX_%dY_w%d_h%d.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
        
        newTile.save(savePath)
        
        newTile.close()
            
def meshpatchExtraction_ROI(config):
    Image.MAX_IMAGE_PIXELS = None
    
    BASE_DIR = config['directories']['DEST_DIR']
    PATCH_SIZE = config['PatchInfo']['meshPATCH_SIZE']
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']
    SCALED_ROI_DIR = config['directories']['SCALED_ROI_DIR_SF8']

    PATCH_DEST_DIR = os.path.join(BASE_DIR,'extracted_meshPatches_HE_ROI_SF8')
    if not os.path.exists(PATCH_DEST_DIR):
        os.mkdir(PATCH_DEST_DIR)
        
    bySampleMeshDir = os.path.join(PATCH_DEST_DIR,'bySample_Mesh')
    if not os.path.exists(bySampleMeshDir):
        os.mkdir(bySampleMeshDir)
    
    config['directories']['bySampleMeshPatches_ROI'] = bySampleMeshDir
    
    samples = [s for s in os.listdir(SCALED_ROI_DIR)]
    num_samples = len(samples)
    sampleCounter = 1
    for sample in samples:
        sample_dest_dir = os.path.join(bySampleMeshDir,sample)
        if not os.path.exists(sample_dest_dir):
            os.mkdir(sample_dest_dir)
        
        sampleDir = os.path.join(SCALED_ROI_DIR,sample)
        
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            ROI_DEST_DIR = os.path.join(sample_dest_dir,ROI)
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
        
            ROI_DIR = os.path.join(sampleDir,ROI)
            ims = [a for a in os.listdir(ROI_DIR) if str(a).endswith(WSI_EXTENSION)]
            
            #FIXED = [f for f in ims if str(f).split('.')[0].split('_')[-1]=='1'][0]
            FIXED = [f for f in ims if str(f).split('_')[3]=='1'][0]
            
            print('Extracting mesh patches from FIXED image (%s) for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total samples.' % (FIXED,sample,ROI,ROICOUNTER,len(ROIS),sampleCounter,num_samples))
            
            im = openslide.open_slide(os.path.join(ROI_DIR,FIXED))
            orig_width, orig_height = im.dimensions
                
            possibleXs = list(range(0,orig_width,PATCH_SIZE))
            possibleYs = list(range(0,orig_height,PATCH_SIZE))
                
            possibleCoords = []
            for i in possibleXs:
                possibleCoords += list(tuple(zip([i]*len(possibleYs),possibleYs)))
                
            extractPatches(im=im, possibleCoords=possibleCoords, PATCH_SIZE = PATCH_SIZE, sampleDir=ROI_DEST_DIR, WSI = FIXED, WSI_EXTENSION = WSI_EXTENSION)
            
            ROICOUNTER += 1 
            
        sampleCounter += 1
    
    
    return config
    
if __name__=='__main__':
    main()
