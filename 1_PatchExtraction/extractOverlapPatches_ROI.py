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

def extractPatches(im, possibleCoords, PATCH_SIZE, dest_dir, filtered_dir, sample, ROI, WSI_EXTENSION, MESHMAP, MusclemeshMap):
    sampleID = str(dest_dir).split('/')[-1]
    
    for possibleCoord in possibleCoords:
        newTile = im.read_region((possibleCoord[0],possibleCoord[1]), 0, (PATCH_SIZE,PATCH_SIZE)).convert("RGB")
        
        decision = filter_and_sort(newTile,PATCH_SIZE)
        
        if decision == 'NoKeep':
            # tmp save to check
            savePath = os.path.join(filtered_dir,sample + '_' + ROI + '_%dX_%dY_w%d_h%d_BGFILTER.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
            newTile.save(savePath)
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
                #savePath = os.path.join(filtered_dir,sample + '_%dX_%dY_w%d_h%d_MESHFILTER_NONTISSUE_%s_MUSCLE_%s.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE),str(float(nonTissueValues/sliceSize)),str(float(MuscleValues/sliceSize))))
                #newTile.save(savePath)
                
                next
                
            elif MuscleValues/sliceSize >= 0.35:
                ### Too much Muscle
                
                # tmp save to check
                #savePath = os.path.join(filtered_dir,sample + '_%dX_%dY_w%d_h%d_MESHFILTER_NONTISSUE_%s_MUSCLE_%s.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE),str(float(nonTissueValues/sliceSize)),str(float(MuscleValues/sliceSize))))
                #newTile.save(savePath)
                
                next
            
            else:
                #savePath = os.path.join(dest_dir,sample + '_NONTISSUE_%s_MUSCLE_%s_%dX_%dY_w%d_h%d.png' % (str(float(nonTissueValues/sliceSize)),str(float(MuscleValues/sliceSize)),int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
                
                savePath = os.path.join(dest_dir,sample + '_' + ROI + '_%dX_%dY_w%d_h%d.png' % (int(possibleCoord[0]), int(possibleCoord[1]),int(PATCH_SIZE),int(PATCH_SIZE)))
        
                newTile.save(savePath)
        
        newTile.close()
            
def extract_overlapPatches_ROI(config):
    Image.MAX_IMAGE_PIXELS = None
    
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    PATCH_SIZE = config['PatchInfo']['PATCH_SIZE_SF8']
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']

    gen_dest = os.path.join(BASE_DIR,'extractedPatches_HE_sf8_wOverlaps_ROI')
    if not os.path.exists(gen_dest):
        os.mkdir(gen_dest)
        
    filtered_overlapsDir = os.path.join(gen_dest,'filteredOut_overlaps_ROI')
    if not os.path.exists(filtered_overlapsDir):
        os.mkdir(filtered_overlapsDir)
        
    overlapsDir = os.path.join(gen_dest,'bySample')
    if not os.path.exists(overlapsDir):
        os.mkdir(overlapsDir)
        
    config['directories']['extractedPatches_HE_sf8_wOverlaps_ROI'] = overlapsDir
    
    samples = [s for s in os.listdir(config['directories']['SCALED_ROI_DIR_SF8'])]
    num_samples = len(samples)
    counter = 1
    for sample in samples:
        print('Extracting Overlap patches from sample  %s.. ------ %d out of %d total samples.' % (sample,counter,len(samples)))
        
        # define dest_dir for overlaps + initial patches
        dest_dir = os.path.join(overlapsDir,sample)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        
        filter_dir = os.path.join(filtered_overlapsDir,sample)
        if not os.path.exists(filter_dir):
            os.mkdir(filter_dir)
        
        ### Get directory of the already extracted sf8 patches for this sample, ROI directories are in this one
        samplePatchDir = os.path.join(config['directories']['bySamplePatches_ROI_FIXED_sf8'],sample)
        
        ### also defined meshmaps sample dir for later
        sampleMeshMapsDir = os.path.join(config['directories']['MESH_MAPS_ROI_DIR'],sample)
        
        sampleDir = os.path.join(config['directories']['SCALED_ROI_DIR_SF8'],sample)
        
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        
        for ROI in ROIS:
            ROI_DEST_DIR = os.path.join(dest_dir,ROI)
            if not os.path.exists(ROI_DEST_DIR):
                os.mkdir(ROI_DEST_DIR)
            ROI_FILTER_DIR = os.path.join(filter_dir,ROI)
            if not os.path.exists(ROI_FILTER_DIR):
                os.mkdir(ROI_FILTER_DIR)
            
            ROI_DIR = os.path.join(sampleDir,ROI)
            
            # open SF8 HE
            WSIs = [w for w in os.listdir(ROI_DIR) if w.endswith(config['SlideInfo']['WSI_EXTENSION'])]
            FIXED = [f for f in WSIs if str(f).split('_')[3]=='1'][0]
            print('Extracting overlap sf8 patches from FIXED image (%s) for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total samples.' % (FIXED,sample,ROI,ROICOUNTER,len(ROIS),counter,num_samples))
        
            #print(os.path.join(ROI_DIR,FIXED))
            im = openslide.open_slide(os.path.join(ROI_DIR,FIXED))

            orig_width, orig_height = im.dimensions
            
            ### Find the already extracted patches for this sample
            ROIpatchDir = os.path.join(samplePatchDir,ROI)
            #print(ROIpatchDir)
            initialPatches = [p for p in os.listdir(ROIpatchDir) if p.endswith('.png')]
            #print(initialPatches)
            
            num_orig = len(initialPatches)
            initialPatchCounter = 1
            
            ## trackers to give percent updates on iteration... only want to say once per quarter percent finished ##
            firstQTrigger = True
            secondQTrigger = True
            thirdQTrigger = True
            fourthQTrigger = True
                
            ## figure out quarter mark of patch #s
            quarterIndicator = num_orig * 0.25
        
                
            for initialPatch in initialPatches:
                if initialPatchCounter >= quarterIndicator and firstQTrigger == True:
                    print('Samples %s (%d out of %d samples) - 25 percent complete.' %(sample,counter,len(samples)))
                    firstQTrigger = False
                        
                elif initialPatchCounter >= (quarterIndicator*2) and secondQTrigger == True:
                    print('Samples %s (%d out of %d samples) - 50 percent complete.' %(sample,counter,len(samples)))
                    secondQTrigger = False

                elif initialPatchCounter >= (quarterIndicator*3) and thirdQTrigger == True:
                    print('Samples %s (%d out of %d samples) - 75 percent complete.' %(sample,counter,len(samples)))
                    thirdQTrigger = False

                elif initialPatchCounter == num_orig and fourthQTrigger == True:
                    print('Samples %s (%d out of %d samples) - 100 percent complete.' %(sample,counter,len(samples)))
                    fourthQTrigger = False
                
                # copy original patch to new overlaps destination
                # rename to get rid of 'reg_' in front
                src = os.path.join(ROIpatchDir,initialPatch)
                dest = os.path.join(ROI_DEST_DIR,initialPatch)
                shutil.copy(src,dest)
                    
                # get this patch's coordinates
                origX = str(initialPatch).split('_')[-4][:-1]
                origY = str(initialPatch).split('_')[-3][:-1]
                    
                # get possible overlap Coords
                shiftedCoords = expandPts((int(origX),int(origY)))
                    
                ### GET MESH MAPS ###
                meshMap = Image.open(os.path.join(os.path.join(sampleMeshMapsDir, ROI),'meshMap_' + str(sample) + '_' + ROI + '.tif'))
                meshMap = np.array(meshMap)
                
                MusclemeshMap = Image.open(os.path.join(os.path.join(sampleMeshMapsDir, ROI),'MusclemeshMap_' + str(sample) + '_' + ROI + '.tif'))
                MusclemeshMap = np.array(MusclemeshMap)
                    
                # extract and filter these overlap patches
                extractPatches(im=im, possibleCoords=shiftedCoords, PATCH_SIZE = PATCH_SIZE, dest_dir=ROI_DEST_DIR, filtered_dir = ROI_FILTER_DIR, sample = sample, ROI = ROI, WSI_EXTENSION = WSI_EXTENSION, MESHMAP = meshMap, MusclemeshMap = MusclemeshMap)
                    
                    
                initialPatchCounter += 1
            ROICOUNTER += 1
            
        counter += 1
        
    return config

    
if __name__=='__main__':
    main()
