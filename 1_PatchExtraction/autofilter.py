import PIL
from PIL import Image
import numpy as np
import os
import shutil
import csv

def filter_and_sort(PILPatch,PATCH_SIZE):
    num_pixels = PATCH_SIZE * PATCH_SIZE

    patchArray = np.array(PILPatch)

    R = patchArray[:,:,0]
    G = patchArray[:,:,1]
    B = patchArray[:,:,2]

    redWhiteCount = (R>200).sum()
    greenWhiteCount = (G>200).sum()
    blueWhiteCount = (B>200).sum()

    redWhiteFreq = redWhiteCount/num_pixels
    greenWhiteFreq = greenWhiteCount/num_pixels
    blueWhiteFreq = blueWhiteCount/num_pixels

    toCheck = [redWhiteFreq, greenWhiteFreq, blueWhiteFreq]
    
    ### add this for any black boxes from scanning too ###
    
    redBlackCount = (R<20).sum()
    greenBlackCount = (G<20).sum()
    blueBlackCount = (B<20).sum()

    redBlackFreq = redBlackCount/num_pixels
    greenBlackFreq = greenBlackCount/num_pixels
    blueBlackFreq = blueBlackCount/num_pixels

    if (redWhiteFreq>.65) and (greenWhiteFreq>.65) and (blueWhiteFreq>.65):
        ## this is white background --> assign to Other
        return 'NoKeep'

    elif len([a for a in toCheck if a>0.9]) >=2:
        return 'NoKeep'
    
    elif (redBlackFreq>.3) and (greenBlackFreq>.3) and (blueBlackFreq>.3):
        # get rid of any black tiles
        return 'NoKeep'
    
    else:
        return 'Keep'

def autofilterPatch(PILpatch, PATCH_SIZE):
    decision = filter_and_sort(PILpatch, PATCH_SIZE)
    return decision
    
if __name__ == '__main__':
    main()
