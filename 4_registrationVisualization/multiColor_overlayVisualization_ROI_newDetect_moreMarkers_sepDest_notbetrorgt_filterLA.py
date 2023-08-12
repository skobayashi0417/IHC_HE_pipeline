import os
import scipy
import skimage
from skimage import data, measure, morphology
from skimage.color import rgb2hed, hed2rgb
import PIL
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import openslide
import io
import pyvips
import math
import copy
import staintools
import shutil
import csv
import json
import time
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


PARAM_DICT = {'cd3': {'blackArtifactThreshold':155,
                     'minArtifactSize':5,
                     'ENHANCE_SCALE':5,
                     'tissueThreshold':30,
                     'minTissueSize':70,
                     'maxTissueSize':5000},
              'cd4': {'blackArtifactThreshold':180,
                     'minArtifactSize':10,
                     'ENHANCE_SCALE':1.5,
                     'tissueThreshold':170,
                     'minTissueSize':70,
                     'maxTissueSize':802817}, # not needed, larger than patch size
              'cd8b': {'blackArtifactThreshold':170,
                     'minArtifactSize':5,
                     'ENHANCE_SCALE':1.5,
                     'tissueThreshold':160,
                     'minTissueSize':70,
                     'maxTissueSize':802817}, # not needed, larger than patch size
              'rorgt':{'blackArtifactThreshold':170, ### RORGT NOT OPTIMIZED, THIS IS A PLACEHOLDER
                     'minArtifactSize':5,
                     'ENHANCE_SCALE':1.5,
                     'tissueThreshold':160,
                     'minTissueSize':70,
                     'maxTissueSize':802817}, # not needed, larger than patch size
              'tbet':{'blackArtifactThreshold':170, ### TBET NOT OPTIMIZED, THIS IS A PLACEHOLDER
                     'minArtifactSize':5,
                     'ENHANCE_SCALE':1.5,
                     'tissueThreshold':160,
                     'minTissueSize':70,
                     'maxTissueSize':802817}, # not needed, larger than patch size
}

POSSIBLE_COLORS = [(255,0,0),
                   (0,255,0),
                   (0,255,255),
                   (255,255,0),
                   (0,0,255),
                   (255,0,255),
                   (0,0,128),
                   (0,128,0),
                   (128,0,0),
                   (0,128,128),
                   (128,0,128),
                   (128,128,0)]

def save_downsized(img, SCALEFACTOR, saveDir, saveName, wsi_ext='.tif'):
    print('generating smaller dim overlay...')
    scaled_w = int(math.floor(img.size[0]/SCALEFACTOR))
    scaled_h = int(math.floor(img.size[1]/SCALEFACTOR))

    scaled_WSI = img.resize((scaled_w, scaled_h), PIL.Image.BILINEAR)

    scaled_WSI.save(os.path.join(saveDir, saveName[:-len(wsi_ext)] + '_sf10' + wsi_ext))

def generate_noLA_overlay(sample,ROI,Moving_dir,keptCoords,relevantIHC_ROI_path):
    global noLA_OVERLAYS_DIR

    base = openslide.open_slide(relevantIHC_ROI_path)
    base = base.read_region((0,0),0,(base.dimensions[0],base.dimensions[1])).convert("RGB")
    
    for keptCoord in keptCoords:
        image = Image.new('RGB',(224,224),(0,0,255))
        image.putalpha(100)
        base.paste(im=image,box=(keptCoord[0],keptCoord[1]),mask=image)
    
    saveName = 'Overlayed_' + str(sample) + '_' + str(ROI) + '_' + str(Moving_dir) + '.tif'
    #base.save(os.path.join(sample_dest_dir,saveName))
    save_downsized(base, 10, noLA_OVERLAYS_DIR, saveName, '.tif')
    
    base=None
                   
def return_params(ihc_target):
    return PARAM_DICT[ihc_target.lower()]['blackArtifactThreshold'], PARAM_DICT[ihc_target.lower()]['minArtifactSize'], PARAM_DICT[ihc_target.lower()]['ENHANCE_SCALE'], PARAM_DICT[ihc_target.lower()]['tissueThreshold'], PARAM_DICT[ihc_target.lower()]['minTissueSize'], PARAM_DICT[ihc_target.lower()]['maxTissueSize']
    

def save_img(img, saveName, saveDir):

    dtype_to_format = {
      'uint8': 'uchar',
      'int8': 'char',
      'uint16': 'ushort',
      'int16': 'short',
      'uint32': 'uint',
      'int32': 'int',
      'float32': 'float',
      'float64': 'double',
      'complex64': 'complex',
      'complex128': 'dpcomplex',
    }
    
    img_path = os.path.join(saveDir,saveName)
    npimg = np.asarray(img)
    height, width, bands = npimg.shape
    linear = npimg.reshape(width * height * bands)
    vimg = pyvips.Image.new_from_memory(linear.data, width, height, bands, dtype_to_format[str(npimg.dtype)])
    vimg.tiffsave(img_path,compression='lzw',tile=True,tile_width=256,tile_height=256,pyramid=True,bigtiff=True)
    
    return img_path

def save_downsized(img, SCALEFACTOR, saveDir, saveName, wsi_ext='.tif'):
    print('generating smaller dim overlay...')
    scaled_w = int(math.floor(img.size[0]/SCALEFACTOR))
    scaled_h = int(math.floor(img.size[1]/SCALEFACTOR))

    scaled_WSI = img.resize((scaled_w, scaled_h), PIL.Image.BILINEAR)

    scaled_WSI.save(os.path.join(saveDir, saveName[:-len(wsi_ext)] + '_sf10' + wsi_ext))

def return_coloredMask(npMask,RGB):
    r = npMask*RGB[0]
    g = npMask*RGB[1]
    b = npMask*RGB[2]
    
    return Image.fromarray((np.stack([r,g,b],axis=-1)).astype(np.uint8))

def generate_overlay(HE_WSI,sample_mask_dest,sample_dest,sample,color_palette):
    print('generating orig dim overlay...')
    
    # Opening base HE
    HE_base = HE_WSI.read_region((0,0),0,(HE_WSI.dimensions[0],HE_WSI.dimensions[1])).convert("RGB")
    
    # TMP
    save_downsized(HE_base, 10, sample_dest, 'HE_base_check.tif', '.tif')
    
    # iterate the masks
    maskWSIs = [w for w in os.listdir(sample_mask_dest) if w.endswith('.tif')]
    
    counter = 1
    for maskWSI in maskWSIs:
        # identify the IHC target and get right RGB palette
        target = str(maskWSI).split('_')[1]
        maskRGB = color_palette[target.lower()]
        
        # convert the mask into an array so we can perform operations on it
        mask_np = np.array(Image.open(os.path.join(sample_mask_dest,maskWSI)))
        
        # return a colored 3 channel mask image based on chosen color palette
        colored_mask = return_coloredMask(mask_np,maskRGB)
        
        # generate mask for the Image.composite function... makes it so overlay should only occur where positive stain detected
        mask_adjusted = Image.fromarray((mask_np*128).astype(np.uint8))
        mask_np = None
        
        # add on the colored mask
        HE_base = Image.composite(colored_mask,HE_base,mask_adjusted)
        mask_adjusted = None
        
        # TMP DEBUG
        save_downsized(HE_base, 10, sample_dest, str(counter)+'_test.tif', '.tif')
        save_downsized(colored_mask, 10, sample_dest, str(target)+'_test.tif', '.tif')
        counter += 1
    
    print('Generating increased brightness and contrast outputs...')
    # increase brightness
    enhancer = ImageEnhance.Brightness(HE_base)
    brightened = enhancer.enhance(1.1)
    
    HE_base = None # save memory

    # now increase contrast
    contrastenhancer = ImageEnhance.Contrast(brightened)
    final = contrastenhancer.enhance(1.5)
    
    brightened = None
     
    enhanced_saveName ='Overlayed_' + sample + '_colors_enhanced.tif'
    save_img(img= final, saveName = enhanced_saveName, saveDir = sample_dest)
    
    save_downsized(final, 10, sample_dest, enhanced_saveName, '.tif')

def return_tissueDetection_pixels(img):
    # convert to grayscale and then to numpy array
    grayscale_img = ImageOps.grayscale(img)
    grayscale_array = np.array(grayscale_img)
    
    # generally detect tissue regions
    tissueMap = grayscale_array < 200
      
    # convert into array of 1s and 0s
    tissue_num_map = tissueMap * 1
    tissueDetectionPixels = np.count_nonzero(tissue_num_map)
      
    # return GT True/False tissue map too for later overlay
    return tissueDetectionPixels

def detectStain(im, IHCpatch, posControlPath, trackerDict, sample_renamed_IHC_dir, fn_coordinfo, target):
    ### collect relevant values for this IHC target
    artThresh, minArtSize, enhance_scale, tissueThresh, minTissueSize, maxTissueSize = return_params(target.lower())

    IHC_patch_mask_dir = os.path.join(sample_renamed_IHC_dir,'masks')
    if not os.path.exists(IHC_patch_mask_dir):
        os.mkdir(IHC_patch_mask_dir)
    
    ##################################
    # prepare sample specific directories
    img_channels_dest = os.path.join(sample_renamed_IHC_dir,'extractedImageChannels')
    renamed_dest = os.path.join(sample_renamed_IHC_dir,'renamedwCounts')
    inv_dest = os.path.join(sample_renamed_IHC_dir,'inverted_wDetect')
    dab_dest = os.path.join(sample_renamed_IHC_dir,'dab_renamed')
    
    if not os.path.exists(img_channels_dest):
        os.mkdir(img_channels_dest)
    if not os.path.exists(renamed_dest):
        os.mkdir(renamed_dest)
    if not os.path.exists(inv_dest):
        os.mkdir(inv_dest)
    if not os.path.exists(dab_dest):
        os.mkdir(dab_dest)
        
    # perform detection
    #im = Image.open(IHCpatch)
    ihc_rgb = np.array(im)

    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(ihc_rgb)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    # Display
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(ihc_rgb)
    ax[0].set_title("Original image")

    ax[1].imshow(ihc_h)
    ax[1].set_title("Hematoxylin")

    ax[2].imshow(ihc_e)
    ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

    ax[3].imshow(ihc_d)
    ax[3].set_title("DAB")

    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()

    #toSave = ax.get_figure()
    fig.savefig(os.path.join(img_channels_dest,str(IHCpatch).split('/')[-1][:-4] + '_channels.png'))
    
    plt.close('all')

    #ihc_rgb_copy = copy.deepcopy(ihc_rgb)

    invert = skimage.util.invert(ihc_rgb)

    #test2 = copy.deepcopy(invert)

    whiteThresh = artThresh

    white_pixels = np.where(
            (invert[:, :, 0] > whiteThresh) &
            (invert[:, :, 1] > whiteThresh) &
            (invert[:, :, 2] > whiteThresh)
        )

    b_np = np.full((896,896),False)
    b_np[white_pixels] = True

    l = morphology.remove_small_objects(b_np,minArtSize,8)
    white_pixels_nosmall = np.where(l==True)
    #invert[white_pixels_nosmall] = [255, 0, 0]

    black_artifacts = measure.label(l)
    num_black_artifacts = np.max(black_artifacts)

    # copied dab
    #ihc_d_copy = copy.deepcopy(ihc_d)
        

    d_np = np.array(ihc_d)
    grayscale_img = ImageOps.grayscale(Image.fromarray((ihc_d*255).astype(np.uint8)))
    enhancer = ImageEnhance.Contrast(grayscale_img)
    im_output = enhancer.enhance(enhance_scale)

    grayscale_array = np.array(im_output)

    tissueMap = grayscale_array < tissueThresh
    #l2 = morphology.remove_small_objects(tissueMap,minTissueSize,8)
       
    selem = ndi.generate_binary_structure(tissueMap.ndim, connectivity=8)
    ccs = np.zeros_like(tissueMap, dtype=np.int32)
    ndi.label(tissueMap, selem, output=ccs)

    component_sizes = np.bincount(ccs.ravel())

    # just eliminate anything above thresh
    max_size = maxTissueSize

    #new_comps_mask = component_sizes < max_size
    #new_comps = component_sizes * new_comps_mask

    too_large = component_sizes > max_size
    too_large_mask = too_large[ccs]
    tissueMap[too_large_mask] = 0

    l2 = morphology.remove_small_objects(tissueMap,70,8)
    
    ###
    DAB_positive = measure.label(l2)
    num_DAB_positive = np.max(DAB_positive)
        
    #total_num_positive = num_DAB_positive - num_black_artifacts
        
    ##
    dab_pixels = np.where(l2[:, :] == True)
    dab_indices = [(r[0], r[1]) for r in zip(dab_pixels[0],dab_pixels[1])]
    black_art_indices = [(r[0], r[1]) for r in zip(white_pixels_nosmall[0],white_pixels_nosmall[1])]
    dab_indices = [a for a in dab_indices if a not in black_art_indices]

    only_dab_pixels = (np.array([a[0] for a in dab_indices],dtype=int),np.array([a[1] for a in dab_indices],dtype=int))
    #print(only_dab_pixels)
    #invert[only_dab_pixels] = [0, 255, 0]
    
    ## perform perc stain area detection
    bad_labels = set([DAB_positive[a[0],a[1]] for a in black_art_indices])

    for u in range(0,896):
        for i in range(0,896):
            if DAB_positive[u,i] in bad_labels:
                DAB_positive[u,i]=0
    
    DAB_positive = DAB_positive > 0
    
    final_dab_pixels = np.where(DAB_positive[:,:]==True)
    final_dab_positive = measure.label(DAB_positive)
    
    total_num_positive = np.max(final_dab_positive)
    
    stain_positivity_area = np.count_nonzero(DAB_positive)
    tissueDetectionPixels = return_tissueDetection_pixels(im)
            
    # calculate percent of stain_positivity area out of total tissue detection pixels
    if tissueDetectionPixels == 0:
        # this has no tissue, just set percStainPos to 0
        percStainPos = 0
    else:
        percStainPos = (stain_positivity_area/tissueDetectionPixels)*100

    invert[final_dab_pixels] = [0, 255, 0]

    for a,b in black_art_indices:
        invert[a,b,0] = 255
        invert[a,b,1] = 0
        invert[a,b,2] = 0
                
    #save all outputs
    Image.fromarray((l*255).astype(np.uint8)).save(os.path.join(IHC_patch_mask_dir,str(IHCpatch).split('/')[-1][:-4] +'_blackArtifactMask_' + 'stainPosPerce_' + str(percStainPos) + '_Count_' + str(total_num_positive) + '.png'))
    Image.fromarray((l2*255).astype(np.uint8)).save(os.path.join(IHC_patch_mask_dir,str(IHCpatch).split('/')[-1][:-4] +'_DABDetection_' + 'stainPosPerce_' + str(percStainPos) + '_Count_' + str(total_num_positive) + '.png'))
    
    Image.fromarray(invert.astype(np.uint8)).save(os.path.join(inv_dest,str(IHCpatch).split('/')[-1][:-4] +'_invDetection_' + 'stainPosPerce_' + str(percStainPos) + '_Count_'+ str(total_num_positive) + '.png'))
    Image.fromarray((ihc_d*255).astype(np.uint8)).save(os.path.join(dab_dest,str(IHCpatch).split('/')[-1][:-4] +'_DABchannel_' + 'stainPosPerce_' + str(percStainPos) + '_Count_' + str(total_num_positive) + '.png'))
    
    im.save(os.path.join(renamed_dest,str(IHCpatch).split('/')[-1][:-4] +'_counted_' + str(total_num_positive) + '.png'))
    ##################################
            
    # find relevant IHC patch
    IHC_saveName = str(IHCpatch).split('/')[-1][:-4] + 'stainPosPerce_' + str(percStainPos) + '_Count_' + str(total_num_positive) + '.png'
    shutil.copy(IHCpatch,os.path.join(sample_renamed_IHC_dir,IHC_saveName))
            
    # save detection mask for this IHC patch
    #saveName = str(IHCpatch).split('/')[-1][:-4] + '_Mask' + '_stainPosPerce_' + str(percStainPos) + '_Count_' + str(num_objects) + '.png'
    #Image.fromarray(all_labels_filtered_boolMap).save(os.path.join(IHC_patch_mask_dir,saveName))

            
    # save info
    trackerDict[fn_coordinfo][target] = (str(total_num_positive),str(percStainPos))

    return DAB_positive, trackerDict

def black_space_check(PILPatch,PATCH_SIZE):
    num_pixels = PATCH_SIZE * PATCH_SIZE

    patchArray = np.array(PILPatch)

    R = patchArray[:,:,0]
    G = patchArray[:,:,1]
    B = patchArray[:,:,2]
    
    redBlackCount = (R<20).sum()
    greenBlackCount = (G<20).sum()
    blueBlackCount = (B<20).sum()

    redBlackFreq = redBlackCount/num_pixels
    greenBlackFreq = greenBlackCount/num_pixels
    blueBlackFreq = blueBlackCount/num_pixels

    if (redBlackFreq>.25) and (greenBlackFreq>.25) and (blueBlackFreq>.25):
        # get rid of any black tiles
        return 'NoKeep'
    
    else:
        return 'Keep'

    
def generate_det_mask(sample,ROI,counter, tot_num,IHCpatches, target, HE_WSI_dimensions, trackerDict, sample_renamed_IHC_dir, SF2_PATCH_SIZE):
    global LA_MESH_MAPS_DIR
    global ROI_filtered_LA_dir
    global noLA_OVERLAYS_DIR
    global ROI_filtered_MASK_LA_dir
    
    # to save kept LA masks
    LA_masks_dest = os.path.join(sample_renamed_IHC_dir,'LA_masks')
    if not os.path.exists(LA_masks_dest):
        os.mkdir(LA_masks_dest)
    
    # open LA mesh map to help filter out areas with LA
    LAmeshMap = Image.open(os.path.join(os.path.join(os.path.join(LA_MESH_MAPS_DIR,sample),ROI),'LAmeshMap_' + str(sample) + '_' + str(ROI) + '.tif'))
    LAMESHMAP = np.array(LAmeshMap)
    
    # track coords that are KEPT after LA filter
    keptCoorderTracker = []
    
    # initialize empty template that is the size of the WSI at SF2
    #prob_map = np.zeros((HE_WSI_dimensions[1],HE_WSI_dimensions[0]))
    prob_map = np.full((HE_WSI_dimensions[1],HE_WSI_dimensions[0]),False)
    pos_control_path = os.path.join('./5_registrationVisualization/POS_CONTROLS',target.lower()+'.png')
    
    num_orig = len(IHCpatches)
    initialPatchCounter = 1
            
    ## trackers to give percent updates on iteration... only want to say once per quarter percent finished ##
    firstQTrigger = True
    secondQTrigger = True
    thirdQTrigger = True
    fourthQTrigger = True
            
    ## figure out quarter mark of patch #s
    quarterIndicator = num_orig * 0.25
    
    for IHC_patch in IHCpatches:
        ### counter
        if initialPatchCounter >= quarterIndicator and firstQTrigger == True:
            print('Samples %s ROI %s (%d out of %d samples) - 25 percent complete.' %(sample, ROI, counter,tot_num))
            firstQTrigger = False
                    
        elif initialPatchCounter >= (quarterIndicator*2) and secondQTrigger == True:
            print('Samples %s ROI %s (%d out of %d samples) - 50 percent complete.' %(sample, ROI, counter,tot_num))
            secondQTrigger = False

        elif initialPatchCounter >= (quarterIndicator*3) and thirdQTrigger == True:
            print('Samples %s ROI %s (%d out of %d samples) - 75 percent complete.' %(sample, ROI, counter,tot_num))
            thirdQTrigger = False

        elif initialPatchCounter == num_orig and fourthQTrigger == True:
            print('Samples %s ROI %s (%d out of %d samples) - 100 percent complete.' %(sample, ROI, counter,tot_num))
            fourthQTrigger = False
            
        initialPatchCounter += 1
    
        ## actual function
        IHCpatch = str(IHC_patch).split('/')[-1]
        #print(IHCpatch)
        fn_coordinfo = ('_').join(str(IHCpatch).split('_')[2:])
        if fn_coordinfo not in trackerDict:
            trackerDict[fn_coordinfo] = {}
        
        # extract IHC patch coords (sf2)
        X_coord = int(fn_coordinfo.split('_')[0][:-1])
        Y_coord = int(fn_coordinfo.split('_')[1][:-1])
        
        # mesh patches are at sf8
        # internal control that these sf2 coords should divide into sf8 with no decimal points
        assert X_coord%4==0
        assert Y_coord%4==0
        
        X_coord_sf8 = int(X_coord/4)
        Y_coord_sf8 = int(Y_coord/4)
        
        SF8_PATCH_SIZE = int(SF2_PATCH_SIZE/4)
        
        meshSLICESize = SF8_PATCH_SIZE * SF8_PATCH_SIZE
        
        meshSlice = LAMESHMAP[Y_coord_sf8:Y_coord_sf8+SF8_PATCH_SIZE,X_coord_sf8:X_coord_sf8+SF8_PATCH_SIZE]
        
        LA_Values = meshSLICESize - np.count_nonzero(meshSlice)
        
        if LA_Values/meshSLICESize >= 0.2:
            # too much LA
            Image.open(IHC_patch).save(os.path.join(ROI_filtered_LA_dir,IHCpatch))
            
            maskSaveName = str(IHCpatch)[:-len('.png')] + '_LAperc_' + str(float(LA_Values/meshSLICESize)) + '.png'
            
            #print(IHCpatch)
            #print(meshSlice.shape)
            #print(X_coord)
            #print(Y_coord)
            #print(LAMESHMAP.shape)
            
            Image.fromarray((meshSlice).astype(np.uint8)).save(os.path.join(ROI_filtered_MASK_LA_dir,maskSaveName))
            next
        
        else:
            maskSaveName = str(IHCpatch)[:-len('.png')] + '_LAperc_' + str(float(LA_Values/meshSLICESize)) + '.png'
            Image.fromarray((meshSlice).astype(np.uint8)).save(os.path.join(LA_masks_dest,maskSaveName))
        
            keptCoorderTracker.append([X_coord,Y_coord])
            
            cur_patch = Image.open(IHC_patch)
            
            # some registered patches will have black bc they dont have corresponding tissue for fixed image in registration... these will just extract as black... need to filter these out as you cannot perform the IHC detection on them
            
            if black_space_check(cur_patch,cur_patch.size[0]) == 'NoKeep':
                next
            else:
                patchMask, trackerDict = detectStain(cur_patch, IHC_patch, pos_control_path, trackerDict, sample_renamed_IHC_dir, fn_coordinfo, target)
                
                if X_coord + SF2_PATCH_SIZE > prob_map.shape[1] and Y_coord + SF2_PATCH_SIZE > prob_map.shape[0]:
                    prob_map[Y_coord:,X_coord:] = patchMask[:prob_map.shape[0]-Y_coord,:prob_map.shape[1]-X_coord]
                elif X_coord + SF2_PATCH_SIZE > prob_map.shape[1]:
                    prob_map[Y_coord:Y_coord+SF2_PATCH_SIZE,X_coord:] = patchMask[:,:prob_map.shape[1]-X_coord]
                elif Y_coord + SF2_PATCH_SIZE > prob_map.shape[0]:
                    prob_map[Y_coord:,X_coord:X_coord+SF2_PATCH_SIZE] = patchMask[:prob_map.shape[0]-Y_coord,:]
                elif X_coord + SF2_PATCH_SIZE < prob_map.shape[1] and Y_coord + SF2_PATCH_SIZE < prob_map.shape[0]:
                    prob_map[Y_coord:Y_coord+SF2_PATCH_SIZE,X_coord:X_coord+SF2_PATCH_SIZE] = patchMask
    
    # set 0 to save memory
    LAmeshMap = None
    LAMESHMAP = None
    
    return prob_map, trackerDict, keptCoorderTracker

def generatePalette_fromIHCTargets(IHC_ORDER):
    
    num_targets = len(IHC_ORDER)
    
    # initialize the color palette dictionary
    colorPalette = {}
    
    for i in range(0,num_targets):
        colorPalette[str(IHC_ORDER[i]).lower()] = POSSIBLE_COLORS[i]
        
    return colorPalette

def overlayVisualization_multiTarget_ROI_newDetect_moreMarkers_sepDest_notbetrorgt_filterLA(config):
    global target
    global LA_MESH_MAPS_DIR
    global ROI_filtered_LA_dir
    global ROI_filtered_MASK_LA_dir
    global noLA_OVERLAYS_DIR
    
    Image.MAX_IMAGE_PIXELS = None
    
    # define IHC markers to keep
    TOKEEP =['cd3','cd4','cd8b']
    
    # define directories
    DEST_DIR = config['directories']['DEST_DIR']
    LA_MESH_MAPS_DIR = config['directories']['LA_meshMaps_dir']
    
    #track time
    start = time.time()
    
    # get IHC order
    IHC_ORDER = config['SlideInfo']['IHC_Order']
    
    # define paths
    BASE_DIR = config['directories']['BASE_DIR']
    
    # where scaled/cropped HE SF2 WSIs are
    ROI_DIR_PREREG_REGISTERED = config['directories']['preRegROI_registrationOutputs']
    ROI_DIR_POSTREG = config['directories']['ROI_postReg']
    
    HE_ROI_samples = [(s,os.path.join(ROI_DIR_PREREG_REGISTERED,s)) for s in os.listdir(ROI_DIR_PREREG_REGISTERED) if 'reformat' not in s]
    
    HE_ROI_samples_postReg = [(r,os.path.join(ROI_DIR_POSTREG,r)) for r in os.listdir(ROI_DIR_POSTREG) if not r.endswith('.csv')]
    
    HE_ROI_samples += HE_ROI_samples_postReg
    
    # where SF2 MOVING Patches are
    MOVING_DIR = os.path.join(os.path.join(BASE_DIR,'extractedPatches_ROI_SF2_MOVING'),'bySample')
    
    # define destination dir for overlays
    gen_dest_dir = os.path.join(DEST_DIR,'registrationOverlays_ROI_newDetect_moreMarkers_notbetrorgt_filterLA')
    if not os.path.exists(gen_dest_dir):
        os.mkdir(gen_dest_dir)
        
    # save renamed IHC patches with count and stain positivty area percent info in fiename
    MOVING_renamed_dir = os.path.join(DEST_DIR,'extractedPatches_ROI_SF2_MOVING_renamed_newDetect_moreMarkers_notbetrorgt_filterLA')
    if not os.path.exists(MOVING_renamed_dir):
        os.mkdir(MOVING_renamed_dir)

    # Where to save IHC counts info
    countCSVs_dir = os.path.join(DEST_DIR,'countCSVs_ROI_newDetect_moreMarkers_notbetrorgt_filterLA')
    if not os.path.exists(countCSVs_dir):
        os.mkdir(countCSVs_dir)
    
    # make directory to save LA filtered out patches for check
    LA_FILTERED_DIR = os.path.join(DEST_DIR,'SF2_filteredLA')
    
    if not os.path.exists(LA_FILTERED_DIR):
        os.mkdir(LA_FILTERED_DIR)
    
    # make directory to save LA filtered out patch masks
    LA_FILTERED_MASKS_DIR = os.path.join(DEST_DIR,'SF2_filteredLA_masks')
    
    if not os.path.exists(LA_FILTERED_MASKS_DIR):
        os.mkdir(LA_FILTERED_MASKS_DIR)
    
    # make directory to save downsized overlays of patches that are kept after LA_filter
    noLA_OVERLAYS_DIR = os.path.join(DEST_DIR,'noLA_overlays')
    if not os.path.exists(noLA_OVERLAYS_DIR):
        os.mkdir(noLA_OVERLAYS_DIR)
    
    config['directories']['VIS_OVERLAYS_DIR_ROI_newDetect'] = gen_dest_dir
    config['directories']['IHC_CSV_Counts_ROI_newDetect'] = countCSVs_dir
    
    # iterate samples
    # generally, create empty template of same shape as SF2 WSI --> perform stain detection --> update WSI-size map patch by patch --> save single marker masks --> create composite overlay onto the H&E of different markers
    # get sampleIDs
    samples = [s for s in os.listdir(MOVING_DIR)]
    
    ALREADY_DONE = [s for s in os.listdir(MOVING_renamed_dir)]
    
    samples = [s for s in samples if s not in ALREADY_DONE]
    print('Samples: ' + str(samples))
    
    # iterate the sample directories to identify all IHC markers and assign RGB values / generate a dictionary for this
    config['color_palette'] = generatePalette_fromIHCTargets(IHC_ORDER)
    with open(os.path.join(gen_dest_dir,'palette.txt'), 'w') as convert_file:
        convert_file.write(json.dumps(config['color_palette']))
    
    counter = 1
    tot_num = len(samples)

    for sample in samples:
        print('On %s... %d out of %d samples' %(sample, counter, tot_num))
        
        sampleDir = os.path.join(MOVING_DIR,sample)
        
         # define general dest dir and create
        sample_dest = os.path.join(gen_dest_dir, sample)
        if not os.path.exists(sample_dest):
            os.mkdir(sample_dest)
        
        # define directory to save IHC patches renamed with output info
        sample_renamed_Moving_dir = os.path.join(MOVING_renamed_dir,sample)
        if not os.path.exists(sample_renamed_Moving_dir):
            os.mkdir(sample_renamed_Moving_dir)
        
        # define directory for filtered LA
        sample_filtered_LA_dir = os.path.join(LA_FILTERED_DIR,sample)
        if not os.path.exists(sample_filtered_LA_dir):
            os.mkdir(sample_filtered_LA_dir)
            
        # define directory for filtered LA masks
        sample_filtered_masks_LA_dir = os.path.join(LA_FILTERED_MASKS_DIR,sample)
        if not os.path.exists(sample_filtered_masks_LA_dir):
            os.mkdir(sample_filtered_masks_LA_dir)
        
        # gather HE_ROI_dirs
        HE_ROI_DIR_SF2 = [a[1] for a in HE_ROI_samples if a[0]==sample][0]
            
        #HE_SF2_ROIS = [h for h in os.listdir(HE_ROI_DIR_SF2) if h.endswith('.tif')]
        
        # gather ROI dirs
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]

        for ROI in ROIS:
            # define path for filtered LA
            ROI_filtered_LA_dir = os.path.join(sample_filtered_LA_dir,ROI)
            if not os.path.exists(ROI_filtered_LA_dir):
                os.mkdir(ROI_filtered_LA_dir)
            
            # define path for filtered LA masks
            ROI_filtered_MASK_LA_dir = os.path.join(sample_filtered_masks_LA_dir,ROI)
            if not os.path.exists(ROI_filtered_MASK_LA_dir):
                os.mkdir(ROI_filtered_MASK_LA_dir)
                
            # define path to ROI DIR containing WSIs for this sample
            CUR_ROI_DIR = os.path.join(HE_ROI_DIR_SF2,ROI)
            
            # set path for sample SF2 Moving patches dir
            ROI_samplePatchDir = os.path.join(sampleDir,ROI)
            
            # set save paths
            ROI_sample_dest = os.path.join(sample_dest,ROI)
            ROI_sample_renamed_Moving_dir = os.path.join(sample_renamed_Moving_dir,ROI)
            
            if not os.path.exists(ROI_sample_dest):
                os.mkdir(ROI_sample_dest)
            if not os.path.exists(ROI_sample_renamed_Moving_dir):
                os.mkdir(ROI_sample_renamed_Moving_dir)
            
            # define directory to save IHC detection masks
            ROI_sample_mask_dest = os.path.join(ROI_sample_dest,'single_mark_WSI_masks')
            if not os.path.exists(ROI_sample_mask_dest):
                os.mkdir(ROI_sample_mask_dest)
        
            # open openslide object for WSI to get dimensions for now... need to conver to PIL later to generate the overlay
            #ROI_ims = [o for o in HE_ROIS if str(o).split('_')[1]==ROI]
            ROI_ims = [o for o in os.listdir(CUR_ROI_DIR) if o.endswith('.tif')]

            HE_BASE = openslide.open_slide([os.path.join(CUR_ROI_DIR,a) for a in ROI_ims if str(a).split('.')[0].split('_')[-1]=='1'][0])
        
            HE_WSI_dims = HE_BASE.dimensions
        
            # gather Moving patch dirs
            Moving_dirs = [d for d in os.listdir(ROI_samplePatchDir)]
            Moving_dirs = [d for d in Moving_dirs if 'HE' not in d]
        
            # initialize trackerDict to track stain positiivty for all patches across all targets
            trackerDict = {}
        
            # iterate IHC_dir/target and create IHC detection mask for each patch by patch
            for Moving_dir in Moving_dirs:
                # only do this if we care about this marker
                if Moving_dir.lower() in TOKEEP:
                    Moving_dir_path = os.path.join(ROI_samplePatchDir, Moving_dir)
                    Moving_patches = [os.path.join(Moving_dir_path,p) for p in os.listdir(Moving_dir_path) if p.endswith('.png')]
                
                    mask, trackerDict, keptCoords = generate_det_mask(sample,ROI,counter, tot_num,Moving_patches,Moving_dir,HE_WSI_dims, trackerDict, ROI_sample_renamed_Moving_dir, config['PatchInfo']['PATCH_SIZE_SF2'])
                
                    Image.fromarray(mask).save(os.path.join(ROI_sample_mask_dest,sample+'_'+Moving_dir+ '_' + ROI + '_singleMarkerMask.tif'))
                    
                    mask = None
                    
                    relevantIHC_ROI_path = [os.path.join(CUR_ROI_DIR,a) for a in ROI_ims if (str(a).split('_')[2].lower() in TOKEEP)][0]
                    
                    generate_noLA_overlay(sample,ROI,Moving_dir,keptCoords,relevantIHC_ROI_path)
                
                else:
                    # this is tbet or rorgt and we do not care right now
                    next
            
            #### AS OF NOW!!! ITERATE SAMPLES --> GATHER SF2 IHC PATCHES FOR EACH TARGET --> PERFORM IHC DETECTION PATCH BY PATCH AND CREATE A WSI-LEVEL MASK ARRAYT BY PATCHING TOGEHR (ALSO SAVE RENAMED IHC PATCHES WITH COUTNS DATA, ALSO HAVE TRACKER DICT TO LOG BOTH COUNT NUMBER AND STAIN POSITIVITY AREA PERCENT ###
        
            ### TO DO: GATHER SINGLE MARKER IHC TARGER MASKS FOR EACH SAMPLE... CREATE MULTICOLORED OVERLAYS ONTO H&E
            ### FOR NOW, DONT WORRY ABOUT OVERLAPS, JSUT GET IT DONE
            generate_overlay(HE_BASE,ROI_sample_mask_dest,ROI_sample_dest,sample,config['color_palette'])
        
            ## take a random patch to get list of all Moving targets
            all_targets = list(trackerDict[list(trackerDict.keys())[0]])
            all_targets_perc = [a+'_percPosArea' for a in all_targets]
        
            appendDict = {}
            saveName = 'Count_stainPerc_info_' + str(sample) + '_' + ROI + '.csv'
            savePath = os.path.join(countCSVs_dir,saveName)
        
            with open(savePath,'w') as csvfile:
                fieldnames = ['patch_fn'] + all_targets + all_targets_perc
                testwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                testwriter.writeheader()
                for key,value in trackerDict.items():
                    appendDict['patch_fn'] = key
                        
                    for k2, v2 in value.items():
                        appendDict[k2] = v2[0]
                        appendDict[str(k2)+'_percPosArea'] = v2[1]
                    testwriter.writerow(appendDict)

        counter += 1
        
    final_end = time.time()
    timeTracker = []
    timeTracker.append(['DONE',str(int((final_end-start)/60))])
    
    save_fn = "time_noCHUNKS.txt"
    summaryFilePath = os.path.join(os.getcwd(),save_fn)
    writeSummary = open(summaryFilePath, 'w')
    for i in range(len(timeTracker)):
        writeSummary.write(str(timeTracker[i][0]) + ': ' + str(timeTracker[i][1]) + '\n')
    writeSummary.close()
        
        
    return config
