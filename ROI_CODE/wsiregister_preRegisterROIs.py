import os
from wsireg.wsireg2d import WsiReg2D
import openslide
import numpy as np
import PIL
from PIL import Image
import math


def gen_HE_stack(HE_ims):
    wOrder = [(a,int(a.split('.')[0].split('_')[3])) for a in HE_ims]
    new = sorted(wOrder, key=lambda x: x[1], reverse=True)
    
    return [c[0] for c in new]
    
def gen_IHC_stack(IHC_WSIs, IHC_ORDER):
    IHC_ORDER = [a.lower() for a in IHC_ORDER]
    
    wTarget = [(b,b.split('.')[0].split('_')[2].lower()) for b in IHC_WSIs]
    new = sorted(wTarget, key = lambda x: IHC_ORDER.index(x[1]),reverse=True)
    
    return [d[0] for d in new]
    
    

def prepare_iteration_list(HE_WSIs,IHC_WSIs,IHC_ORDER):
    newList = []
    
    HE_STACK = gen_HE_stack(HE_WSIs)
    IHC_STACK = gen_IHC_stack(IHC_WSIs, IHC_ORDER)
    
    newList.append(HE_STACK.pop())
    print(newList)
    print(IHC_STACK)

    #while len(HE_STACK)>0 and len(IHC_STACK)>0:
    #    if len(newList)%4==0: # there are 4 slide here, so HE, 3 IHCS,... time for next HE
    #        newList.append(HE_STACK.pop())
    #    else:
    #        newList.append(IHC_STACK.pop())
    while len(IHC_STACK)>0
        newList.append(IHC_STACK.pop())
    newList.reverse()
    print(newList)
    return newList


def registerWSIs_preROI(config):
    global base_dir
    global img_saveDir
    WSI_EXTENSION = config['SlideInfo']['WSI_EXTENSION']
    save_ext = config['SlideInfo']['WSI_EXTENSION']
    BASE_DIR = config['directories']['BASE_DIR']
    SCALE_FACTOR = config['SlideInfo']['SCALE_FACTOR']
    IHC_ORDER = config['SlideInfo']['IHC_Order']
    
    dest_dir = os.path.join(BASE_DIR,'registered_preRegisterROIs_sf2')
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        
    config['directories']['preRegROI_registrationOutputs'] = dest_dir
    
    ROI_dir = os.path.join(config['directories']['ROI_INPUT_DIR'],'bad_reg_qual_ROIs')
        
    samples = [s for s in os.listdir(ROI_dir) if not s.endswith('.csv')]
    samples = [s for s in samples if not s.endswith('.py')]
    
    tot_num = len(samples)
    counter = 1
    for sample in samples:
        sampleDir = os.path.join(ROI_dir,sample)
        print('Performing Registration on %s.. ------ %d out of %d total samples.' % (sample,counter,tot_num))
        
        # create general sample dest_dir
        sample_dest_dir = os.path.join(dest_dir,sample)
        if not os.path.exists(sample_dest_dir):
            os.mkdir(sample_dest_dir)
        
        # gather all ROI images
        ROIS_IMS = [r for r in os.listdir(sampleDir) if r.endswith('.tif')]
        
        # separate out HE and IHC
        HE_ROIS = [r for r in ROIS_IMS if '_HE_' in r]
        IHC_ROIS = [r for r in ROIS_IMS if r not in HE_ROIS]
        
        # get all possible ROI
        ROIS = list(set([str(a).split('_')[1] for a in ROIS_IMS]))
        
        for ROI in ROIS:
            ROI_dest_dir = os.path.join(sample_dest_dir,ROI)
            if not os.path.exists(ROI_dest_dir):
                os.mkdir(ROI_dest_dir)
        
            spec_HE_ROIS = [h for h in HE_ROIS if str(h).split('_')[1]==ROI]
            spec_IHC_ROIS = [i for i in IHC_ROIS if str(i).split('_')[1]==ROI]
            
            iterationList = prepare_iteration_list(spec_HE_ROIS,spec_IHC_ROIS,IHC_ORDER)
            
            fixed_im_tracker = [0]
            counter = 2 # start at 2 because first HE is _1
        
            initialTrigger = 0 # want to keep track of HE_1
            while len(iterationList) > 0:
                if initialTrigger ==0:
                    fixed_im = iterationList.pop()
                    moving_im = iterationList.pop()
                
                    FIXED_PATH = os.path.join(sampleDir,fixed_im)
                    MOVING_PATH = os.path.join(sampleDir,moving_im)
                    #print(FIXED_PATH)
                    #print(MOVING_PATH)
                
                    IHC_specificID = str(moving_im).split('.')[0]
                    print(IHC_specificID)
                
                    curID = 'reg_' + str(IHC_specificID)
                
                    # initialize registration graph
                    #reg_graph = WsiReg2D("reg_"+IHC_specificID, output_dir)
                    reg_graph = WsiReg2D(curID, ROI_dest_dir)

                    # add registration images (modalities)
                    reg_graph.add_modality(
                        "HE",
                        FIXED_PATH,
                        image_res=0.34,
                        prepro_dict={"image_type": "BF", "as_uint8": True, "inv_int_opt": True},
                    )

                    reg_graph.add_modality(
                        "IHC",
                        MOVING_PATH,
                        image_res=0.34,
                        prepro_dict={"image_type": "BF", "as_uint8": True, "inv_int_opt": True},
                    )

                    reg_graph.add_reg_path(
                        "IHC",
                        "HE",
                        thru_modality=None,
                        reg_params=["nl","affine"],
                    )
                    reg_graph.register_images()
                    reg_graph.save_transformations()
                    reg_graph.transform_images(file_writer="ome.tiff")
                
                    # this is the first iteration... rename the HE image to desired filename
                    newName = sample + '_' + ROI + '_' + 'HE_1.tif'
                    HE_slide = [a for a in os.listdir(ROI_dest_dir) if '-HE_' in a][0]
                    os.rename(os.path.join(ROI_dest_dir,HE_slide),os.path.join(ROI_dest_dir,newName))
                
                    # reName moving too
                    newIHCname = IHC_specificID + '_' + str(counter) + '.tif'
                
                    slides = [s for s in os.listdir(ROI_dest_dir) if s.endswith('.tiff')]
                    Moving_slide = [a for a in slides if '-IHC_to_HE' in a][0]
                    print(Moving_slide)
                    os.rename(os.path.join(ROI_dest_dir,Moving_slide),os.path.join(ROI_dest_dir,newIHCname))
                
                    fixed_im_tracker[0] = newIHCname
                    counter += 1
                
                    initialTrigger = 1
            
                else:
                    fixed_im = fixed_im_tracker[0]
                    moving_im = iterationList.pop()
                    
                    moving_src_dir = sampleDir
                
                    IHC_specificID = str(moving_im).split('.')[0]
                
                    # initialize registration graph
                    reg_graph = WsiReg2D("reg_"+IHC_specificID, ROI_dest_dir)

                    # add registration images (modalities)
                    reg_graph.add_modality(
                        "FIXED",
                        os.path.join(ROI_dest_dir,fixed_im),
                        image_res=0.34,
                        prepro_dict={"image_type": "BF", "as_uint8": True, "inv_int_opt": True},
                    )

                    reg_graph.add_modality(
                        "MOVING",
                        os.path.join(moving_src_dir,moving_im),
                        image_res=0.34,
                        prepro_dict={"image_type": "BF", "as_uint8": True, "inv_int_opt": True},
                    )

                    reg_graph.add_reg_path(
                        "MOVING",
                        "FIXED",
                        thru_modality=None,
                        reg_params=["nl","affine"],
                    )

                    reg_graph.register_images()
                    reg_graph.save_transformations()
                    reg_graph.transform_images(file_writer="ome.tiff")
                
                    # dont need this fixed slide...
                    fixed = [a for a in os.listdir(ROI_dest_dir) if '-FIXED_' in a][0]
                    os.remove(os.path.join(ROI_dest_dir,fixed))
                
                    # rename IHC
                    slides = [s for s in os.listdir(ROI_dest_dir) if s.endswith('.tiff')]
                    Moving_slide = [t for t in slides if '-MOVING_to_FIXED' in t][0]

                    if str(moving_im).split('.')[0].split('_')[2] == 'HE':
                        newName = sample + '_' + ROI + '_' + 'HE' + '_' + str(counter) + '.tif'
                    else:
                        newName = IHC_specificID + '_' + str(counter) + '.tif'
                
                    os.rename(os.path.join(ROI_dest_dir,Moving_slide),os.path.join(ROI_dest_dir,newName))
                
                    fixed_im_tracker[0] = newIHCname
                
                    counter +=1
 
    return config
