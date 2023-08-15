import os
import sys
import numpy
import PIL
import csv
import openslide
#from autofilter import *
import shutil

def extractPatches_byCoord(WSI, coords, PATCH_SIZE, target_dest_dir):
    # no need for any filtering. just extract patches at same locations taken from H&E
    for coord in coords:
        newTile = WSI.read_region((int(coord[0]),int(coord[1])), 0, (PATCH_SIZE,PATCH_SIZE)).convert("RGB")
        savePath = os.path.join(target_dest_dir + '/_%dX_%dY_w%d_h%d.png' % (int(coord[0]), int(coord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
        newTile.save(savePath)
        
        newTile.close()

            
def extract_OverlayVisPatches_ROI_newDetect_moreMarkers(config):
    extractedPatches_DIR = config['directories']['bySamplePatches_ROI_FIXED_sf2']
    GEN_DEST_DIR = config['directories']['DEST_DIR']

    DEST_DIR = os.path.join(GEN_DEST_DIR,'extractedPatches_byMarker_fromregistrationIHCVisOverlays_ROI')
    if not os.path.exists(DEST_DIR):
        os.mkdir(DEST_DIR)
    
    OVERLAYS_DIR = config['directories']['VIS_OVERLAYS_DIR_ROI_newDetect']
    
    PATCH_SIZE = config['PatchInfo']['PATCH_SIZE_SF2']
    
    ###
    samples = [s for s in os.listdir(OVERLAYS_DIR) if not s.endswith('.txt')]
    
    for sample in samples:
        print('On %s' %(sample))
        sample_dest = os.path.join(DEST_DIR, sample)
        if not os.path.exists(sample_dest):
            os.mkdir(sample_dest)
            
        sampleDir = os.path.join(OVERLAYS_DIR,sample)
        samplepatches_dir = os.path.join(extractedPatches_DIR,sample)
            
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        
        for ROI in ROIS:
            ROIpatches_dir = os.path.join(samplepatches_dir,ROI)
            patches = [p for p in os.listdir(ROIpatches_dir) if p.endswith('.png')]
            
            ROI_DEST = os.path.join(sample_dest,ROI)
            if not os.path.exists(ROI_DEST):
                os.mkdir(ROI_DEST)

            # extract patch coordinates
            coords = [(str(z).split('_')[-4][:-1],str(z).split('_')[-3][:-1]) for z in patches]
            
            # load the overlay for this sample
            overlayWSI = openslide.open_slide(os.path.join(os.path.join(os.path.join(sampleDir,ROI),[w for w in os.listdir(os.path.join(sampleDir,ROI)) if w.endswith('enhanced.tif')][0])))
            
            # extract same patch locations but from overlayed image
            extractPatches_byCoord(WSI=overlayWSI, coords=coords, PATCH_SIZE = PATCH_SIZE, target_dest_dir=ROI_DEST)
