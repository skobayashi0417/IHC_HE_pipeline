import os
import json
import sys
sys.path.insert(0,'./0_ScaleWSIs')
sys.path.insert(0,'./1_PatchExtraction')
sys.path.insert(0,'./2_Inference')
sys.path.insert(0,'./3_MeshMaps')
sys.path.insert(0,'./4_registrationVisualization')
from generateJSON import *
from scaleWSIs_vipssave_secondLevel_ROI import *
from extractmeshPatches_ROI import *
from meshPredictions_ROI import *
from meshPredictions_ROI_wLA_sepDest import *
from generateMeshMaps_ROI import *
from extractPatches_ROI import *
from extractPatches_firstLevel_ROI import *
from extractPatches_Moving_ROI import *
from multiColor_overlayVisualization_ROI_newDetect_moreMarkers_sepDest_notbetrorgt_filterLA import *
from extractOverlayVisPatches_ROI_newdetect_moreMarkers_sepDest import *

def prepare_directories_ROI(config):
    SCALED_ROI_DIR_SF8 = os.path.join(config['directories']['BASE_DIR'],'scaled_ROI_SF8')
    if not os.path.exists(SCALED_ROI_DIR_SF8):
        os.mkdir(SCALED_ROI_DIR_SF8)
    
    config['directories']['SCALED_ROI_DIR_SF8'] = SCALED_ROI_DIR_SF8
    
    config['directories']['DEST_DIR'] = config['directories']['BASE_DIR']
    
    if not os.path.exists(config['directories']['DEST_DIR']):
        os.mkdir(config['directories']['DEST_DIR'])
    
    return config

def runPipeline_part3(config):
    config = prepare_directories_ROI(config)
    print(config)

    ### Scale HE and registered IHC to sf8
    print('Scaling Registered Images to Second Level Scale (sf8)...')
    #scale_WSI_secondLevel_ROI(ROI_DIR_POSTREG = config['directories']['ROI_postReg'],
    #                          ROI_DIR_PREREG_REGISTERED = config['directories']['preRegROI_registrationOutputs'],
    #                          SCALED_ROI_DIR = config['directories']['SCALED_ROI_DIR_SF8'],
    #                          IHC_ORDER = config['SlideInfo']['IHC_Order'],
    #                          PATCH_SIZE = config['PatchInfo']['PATCH_SIZE_SF8'], #use sf8 patch size as that is final expected scale
    #                          WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION'],
    #                          SCALE_FACTOR = 4) #going from sf2 to sf8
    
    ### Extract patches
    print('Performing Mesh Patch Extraction on Second Level Scale (sf8)...')
    #config = meshpatchExtraction_ROI(config = config)
    
    ### Run Mesh Predictions
    print('Running Mesh Predictions...')
    #config = perform_Meshpredictions_ROI(config = config)
    
    ### Run Mesh Predictions_wLA
    print('Running Mesh Predictions with LA...')
    #config = perform_Meshpredictions_ROI_wLA_sepDest(config = config)
    
    ### Generating Mesh Maps
    print('Generating Mesh Maps...')
    
    config['directories']['meshPREDICTIONS_DIR_ROI_wLA'] = '/data01/shared/skobayashi/github_test/32pixelPatch_Predictions_ROI_wLA_sepDest'
    config['directories']['meshPREDICTIONS_DIR_ROI'] = '/data01/shared/skobayashi/github_test/32pixelPatch_Predictions_ROI'
    
    config = createMeshMaps_ROI(config)
    
    ### Extract Larger Patches
    print('Performing Patch Extraction at Second Level Scale (sf8)...')
    #config = patchExtraction_ROI(config)
    
    ### Extract Corresponding Patches at First Scale
    print('Performing Extraction of Corresponding Patches at First Level Scale (sf2)...')
    #config = patchExtraction_firstLevel_ROI(config)
    
    ### Extracting IHC Patches at Both Scales
    print('Performing Corresponding Moving Patch Extraction at both scales...')
    #config = patchExtraction_Moving_ROI(config)
    
    config['directories']['SCALED_ROI_DIR_SF8'] = '/data01/shared/skobayashi/github_test/scaled_ROI_SF8'
    config['directories']['DEST_DIR'] = config['directories']['BASE_DIR']
    config['directories']['bySampleMeshPatches_ROI']  = '/data01/shared/skobayashi/github_test/extracted_meshPatches_HE_ROI_SF8/bySample_Mesh'
    config['directories']['meshPREDICTIONS_DIR_ROI_wLA'] = '/data01/shared/skobayashi/github_test/32pixelPatch_Predictions_ROI_wLA_sepDest'
    config['directories']['meshPREDICTIONS_DIR_ROI'] = '/data01/shared/skobayashi/github_test/32pixelPatch_Predictions_ROI'
    config['directories']['MESH_MAPS_ROI_DIR'] = '/data01/shared/skobayashi/github_test/meshMaps_ROI'
    config['directories']['LA_meshMaps_dir']  = '/data01/shared/skobayashi/github_test/meshMaps_ROI_wLA'
    config['directories']['bySamplePatches_ROI_FIXED_sf8'] = '/data01/shared/skobayashi/github_test/extractedPatches_ROI_FIXED_sf8/bySample'
    config['directories']['bySamplePatches_ROI_FIXED_sf2'] = '/data01/shared/skobayashi/github_test/extractedPatches_ROI_FIXED_sf2/bySample'
    
    

    ### Generate Single marker IHC detection masks... Generate IHC counts per patch as well
    print('Generating vis Overlays...')
    config = overlayVisualization_multiTarget_ROI_newDetect_moreMarkers_sepDest_notbetrorgt_filterLA(config)
    
    ### Extract patches from these overlay outputs
    print('Extracting patches from vis Overlays...')
    #extract_OverlayVisPatches_ROI_newDetect_moreMarkers(config)
    
    with open('config_part3.json','w') as outfile:
        json.dump(config,outfile)

    
if __name__ == '__main__':
    config = json.load(open('./config_part2.json'))
    
    runPipeline_part3(config)
