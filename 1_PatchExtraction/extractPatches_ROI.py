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

def extractPatches(im, possibleCoords, PATCH_SIZE, sampleDir, filtered_sampleDir, WSI, WSI_EXTENSION, MESHMAP, MusclemeshMap):
    sampleID = str(sampleDir).split('/')[-1]
    
    for possibleCoord in possibleCoords:
        newTile = im.read_region((possibleCoord[0],possibleCoord[1]), 0, (PATCH_SIZE,PATCH_SIZE)).convert("RGB")
        
        decision = filter_and_sort(newTile,PATCH_SIZE)
        
        if decision == 'NoKeep':
            # tmp save to check
            #savePath = os.path.join(filtered_mouseDir,str(WSI)[:-(len(WSI_EXTENSION))] + '_%dX_%dY_w%d_h%d_BGFILTER.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
            #newTile.save(savePath)
            next
            
        else:
            ### Calculate Mesh Slice amount of Non-Tissue (bg + muscle) but with a higher thresh ###
            meshSlice = MESHMAP[possibleCoord[1]:possibleCoord[1]+PATCH_SIZE,possibleCoord[0]:possibleCoord[0]+PATCH_SIZE]
            sliceSize = PATCH_SIZE * PATCH_SIZE
            
            ## Get Amount of Non Tisses in Array Slice -- should be 0s
            nonTissueValues = sliceSize - np.count_nonzero(meshSlice)
            
            ### Calculate Muscle Mesh Slice amount of Non-Tissue (Muscle) with a lower thresh ###
            MusclemeshSlice = MusclemeshMap[possibleCoord[1]:possibleCoord[1]+PATCH_SIZE,possibleCoord[0]:possibleCoord[0]+PATCH_SIZE]
            
            ## Get Amount of Muscle in Array Slice -- should be 0s
            MuscleValues = sliceSize - np.count_nonzero(MusclemeshSlice)

            if nonTissueValues/sliceSize >= 0.65:
                ### Too little tissue
                
                # tmp save to check
                #savePath = os.path.join(filtered_mouseDir,str(WSI)[:-(len(WSI_EXTENSION))] + '_%dX_%dY_w%d_h%d_MESHFILTER_NONTISSUE_%s_MUSCLE_%s.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE),str(float(nonTissueValues/sliceSize)),str(float(MuscleValues/sliceSize))))
                #newTile.save(savePath)
                
                next
                
            elif MuscleValues/sliceSize >= 0.35:
                ### Too much Muscle
                
                # tmp save to check
                #savePath = os.path.join(filtered_mouseDir,str(WSI)[:-(len(WSI_EXTENSION))] + '_%dX_%dY_w%d_h%d_MESHFILTER_NONTISSUE_%s_MUSCLE_%s.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE),str(float(nonTissueValues/sliceSize)),str(float(MuscleValues/sliceSize))))
                #newTile.save(savePath)
                
                next
            
            else:
                savePath = os.path.join(sampleDir,str(WSI)[:-(len(WSI_EXTENSION))] + '_%dX_%dY_w%d_h%d.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
        
                newTile.save(savePath)
        
        newTile.close()
            
def patchExtraction_ROI(config):
    Image.MAX_IMAGE_PIXELS = None
    
    BASE_DIR = config['directories']['DEST_DIR']
    PATCH_SIZE = config['PatchInfo']['PATCH_SIZE_SF8']
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']
    SCALED_ROI_DIR = config['directories']['SCALED_ROI_DIR_SF8']
    MESHMAPS_DIR = config['directories']['MESH_MAPS_ROI_DIR']

    PATCH_DEST_DIR = os.path.join(BASE_DIR,'extractedPatches_ROI_FIXED_sf8')
    if not os.path.exists(PATCH_DEST_DIR):
        os.mkdir(PATCH_DEST_DIR)
        
    bySampleDir = os.path.join(PATCH_DEST_DIR,'bySample')
    if not os.path.exists(bySampleDir):
        os.mkdir(bySampleDir)
        
    filtered_bySampleDir = os.path.join(PATCH_DEST_DIR,'filteredOut_bySample')
    if not os.path.exists(filtered_bySampleDir):
        os.mkdir(filtered_bySampleDir)
        
    config['directories']['bySamplePatches_ROI_FIXED_sf8'] = bySampleDir
    
    samples = [s for s in os.listdir(SCALED_ROI_DIR)]
    num_samples = len(samples)
    sampleCounter = 1
    
    for sample in samples:
        sampleDir = os.path.join(SCALED_ROI_DIR,sample)
        
        sample_dest_dir = os.path.join(bySampleDir,sample)
        if not os.path.exists(sample_dest_dir):
            os.mkdir(sample_dest_dir)
        
        filtered_sampleDir = os.path.join(filtered_bySampleDir,sample)
        if not os.path.exists(filtered_sampleDir):
            os.mkdir(filtered_sampleDir)
        
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            WSIs = [a for a in os.listdir(os.path.join(sampleDir,ROI)) if str(a).endswith(WSI_EXTENSION)]
            FIXED = [z for z in WSIs if str(z).split('_')[3]=='1'][0]
            
            print('Extracting sf8 patches from FIXED image (%s) for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total samples.' % (FIXED,sample,ROI,ROICOUNTER,len(ROIS),sampleCounter,num_samples))
            #print('Extracting patches from sample  %s ROI %s... ------ %d out of %d total samples.' % (sample,ROI,sampleCounter,num_samples))
            
            ROI_DEST_DIR = os.path.join(sample_dest_dir,ROI)
            ROI_FILTERED_DIR = os.path.join(filtered_sampleDir,ROI)
            
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
            if not os.path.exists(ROI_FILTERED_DIR):
                os.mkdir(ROI_FILTERED_DIR)
                
            im = openslide.open_slide(os.path.join(os.path.join(sampleDir,ROI),FIXED))
            orig_width, orig_height = im.dimensions
                
            possibleXs = list(range(0,orig_width,PATCH_SIZE))
            possibleYs = list(range(0,orig_height,PATCH_SIZE))
                
            possibleCoords = []
            for i in possibleXs:
                possibleCoords += list(tuple(zip([i]*len(possibleYs),possibleYs)))
                    
            ### GET MESH MAPS ###
            sampleMeshMapsDir = os.path.join(MESHMAPS_DIR,sample)
            
            meshMap = Image.open(os.path.join(os.path.join(sampleMeshMapsDir, ROI),'meshMap_' + str(sample) + '_' + ROI + '.tif'))
            meshMap = np.array(meshMap)
            
            MusclemeshMap = Image.open(os.path.join(os.path.join(sampleMeshMapsDir, ROI),'MusclemeshMap_' + str(sample) + '_' + ROI + '.tif'))
            MusclemeshMap = np.array(MusclemeshMap)
                
            extractPatches(im=im, possibleCoords=possibleCoords, PATCH_SIZE = PATCH_SIZE, sampleDir=ROI_DEST_DIR, filtered_sampleDir = ROI_FILTERED_DIR, WSI = FIXED, WSI_EXTENSION = WSI_EXTENSION, MESHMAP = meshMap, MusclemeshMap = MusclemeshMap)
            
            ROICOUNTER += 1
            
        sampleCounter += 1
            
    return config
    
if __name__=='__main__':
    main()
