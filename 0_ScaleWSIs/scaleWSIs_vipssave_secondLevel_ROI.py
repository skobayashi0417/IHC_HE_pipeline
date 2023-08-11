import os
import PIL
from PIL import Image
import csv
import math
import openslide
import numpy as np

def gen_HE_stack(HE_ims):
    wOrder = [(a,int(a.split('-')[0].split('_')[2])) for a in HE_ims]
    new = sorted(wOrder, key=lambda x: x[1], reverse=True)
    
    return [c[0] for c in new]
    
def gen_IHC_stack(IHC_WSIs, IHC_ORDER):
    IHC_ORDER = [a.lower() for a in IHC_ORDER]
    
    wTarget = [(b,b.split('-')[0].split('_')[1].lower()) for b in IHC_WSIs]
    new = sorted(wTarget, key = lambda x: IHC_ORDER.index(x[1]),reverse=True)
    
    return [d[0] for d in new]
    
    

def prepare_iteration_list(HE_WSIs,IHC_WSIs,IHC_ORDER):
    newList = []
    
    HE_STACK = gen_HE_stack(HE_WSIs)
    IHC_STACK = gen_IHC_stack(IHC_WSIs, IHC_ORDER)
    
    newList.append(HE_STACK.pop())

    while len(HE_STACK)>0 and len(IHC_STACK)>0:
        if len(newList)%4==0: # there are 4 slide here, so HE, 3 IHCS,... time for next HE
            newList.append(HE_STACK.pop())
        else:
            newList.append(IHC_STACK.pop())
    
    newList.reverse()
    
    return newList
    
def return_scaled_PIL(WSI,SCALE_FACTOR):
    os_WSI = openslide.open_slide(WSI)
    PIL_WSI = os_WSI.read_region((0,0),0,size=(os_WSI.dimensions[0],os_WSI.dimensions[1])).convert("RGB")
    
    scaled_w = int(math.floor(PIL_WSI.size[0]/SCALE_FACTOR))
    scaled_h = int(math.floor(PIL_WSI.size[1]/SCALE_FACTOR))

    scaled_WSI = PIL_WSI.resize((scaled_w, scaled_h), PIL.Image.BILINEAR)
    
    return scaled_WSI
    

def return_cropped_PIL(orig_WSI, INPUT_WSI_DIR, SCALE_FACTOR, PATCH_SIZE):
    oWSI = openslide.open_slide(os.path.join(INPUT_WSI_DIR,orig_WSI))
    
    orig_w, orig_h = oWSI.dimensions
    expectedScale = SCALE_FACTOR * PATCH_SIZE
    
    new_w = int((math.floor(orig_w / expectedScale)) * expectedScale)
    new_h = int((math.floor(orig_h / expectedScale)) * expectedScale)
    
    wDiff = int(orig_w - new_w)
    hDiff = int(orig_h - new_h)
    
    xStart = int(wDiff/2)
    yStart = int(hDiff/2)
    
    #cropped_whole_slide_image = oWSI.read_region(location=(xStart, yStart), level=0, size = (new_w,new_h))
    #level = oWSI.get_best_level_for_downsample(SCALE_FACTOR)
    
    # for some reason, these images scanned with a weird black border... crop an extra patch worth (8*224)
    
    cropped_whole_slide_image = oWSI.read_region((xStart, yStart), 0, size = (new_w,new_h))
    cropped_img = cropped_whole_slide_image.convert("RGB")
    
    return cropped_img, new_w, new_h, orig_w, orig_h,
    
def scale_WSI_secondLevel_ROI(ROI_DIR_POSTREG,ROI_DIR_PREREG_REGISTERED,SCALED_ROI_DIR, IHC_ORDER, PATCH_SIZE = 224, WSI_EXTENSION='.tif',SCALE_FACTOR=4):
    global appendList
    
    appendList = []
    appendDict = {}
    
    #samples = [d for d in os.listdir(ROI_DIR) if not d.endswith('.csv')]
    
    samples = [os.path.join(ROI_DIR_PREREG_REGISTERED,s) for s in os.listdir(ROI_DIR_PREREG_REGISTERED) if 'reformat' not in s]
    
    samples_postReg = [os.path.join(ROI_DIR_POSTREG,r) for r in os.listdir(ROI_DIR_POSTREG) if not r.endswith('.csv')]
    
    samples += samples_postReg
    
    tot_num = len(samples)
    counter = 1
    
    for sampleDir in samples:
        sample = str(sampleDir).split('/')[-1]
        sample_dest_dir = os.path.join(SCALED_ROI_DIR,sample)
        if not os.path.exists(sample_dest_dir):
            os.mkdir(sample_dest_dir)

        ROI_DIRS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        
        for ROI_DIR in ROI_DIRS:
            ROI_DEST = os.path.join(sample_dest_dir,ROI_DIR)
            if not os.path.exists(ROI_DEST):
                os.mkdir(ROI_DEST)
                
            ROI_DIR_PATH = os.path.join(sampleDir,ROI_DIR)
            
            ROIS = [o for o in os.listdir(ROI_DIR_PATH) if o.endswith('.tif')]
            
            ROIcounter = 1
            for ROI in ROIS:
                print('Performing Scaling on %s.. (%d/%d) ------ %d out of %d total samples.' % (ROI,ROIcounter,len(ROIS),counter,tot_num))
                scaled_img= return_scaled_PIL(WSI = os.path.join(ROI_DIR_PATH,ROI), SCALE_FACTOR=SCALE_FACTOR)

                scaledSaveName = str(ROI)[:-len('.tif')] + '_scaledFactor' + str(8) + WSI_EXTENSION
                
                scaled_img.save(os.path.join(ROI_DEST,scaledSaveName))
                ROIcounter += 1
        
        counter += 1
