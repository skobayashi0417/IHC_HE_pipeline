import os
import PIL
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import openslide
import io
import pyvips
import math

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

    scaled_WSI.save(os.path.join(saveDir, saveName[:-len(wsi_ext)] + '_sf' + str(SCALEFACTOR) + wsi_ext))

def generate_overlay(Fixed_img,IHC_array,target, orderNum, ROI):
    global sample_dest

    # Create an overlay
    IHC_mask = Image.fromarray(IHC_array).convert("RGB")
    
    IHC_mask.putalpha(100)
    Fixed_img.paste(im=IHC_mask,box=(0,0),mask=IHC_mask)
    
    saveName = 'OverlayCheck_' + ROI + '_' + target + '_' + str(orderNum) + '.tif'
    
    #save_img(img= Fixed_img, saveName = saveName, saveDir = sample_dest)
    save_downsized(img = Fixed_img, SCALEFACTOR = 10, saveDir = sample_dest, saveName = saveName)

def return_tissueDetection_Arrays(img):
    # convert to grayscale and then to numpy array
    grayscale_img = ImageOps.grayscale(img)
    grayscale_array = np.array(grayscale_img)
    
    # generally detect tissue regions
    tissueMap = grayscale_array < 200
      
    # convert into array of 1s and 0s
    tissue_num_map = tissueMap * 1
      
    # return GT True/False tissue map too for later overlay
    return tissue_num_map, tissueMap

def main(IHC_Path, Fixed_Path, target, orderNum, ROI):
    IHC = openslide.open_slide(IHC_Path)
    IHC = IHC.read_region((0,0),0,size=(IHC.dimensions[0],IHC.dimensions[1]))

    _, IHC_map = return_tissueDetection_Arrays(IHC)

    Fixed = openslide.open_slide(Fixed_Path)
    Fixed = Fixed.read_region((0,0),0,size=(Fixed.dimensions[0],Fixed.dimensions[1]))

    generate_overlay(Fixed,IHC_map,target, orderNum, ROI)


def generate_genOverlays_composite(config):
    global sample_dest
    
    registrationOutputsDir = config['directories']['preRegROI_registrationOutputs']
    baseDir = config['directories']['BASE_DIR']
    
    postRegROIDir = os.path.join(config['directories']['ROI_INPUT_DIR'],'goodreg_qual_ROIs')
    
    gen_dest_dir = os.path.join(config['directories']['BASE_DIR'],'generalOverlays_forCheck_Registered_ROIs')
    if not os.path.exists(gen_dest_dir):
        os.mkdir(gen_dest_dir)
    
    config['directories']['genOverlayCheck_PreAndPost_ROIs'] = gen_dest_dir
    config['directories']['ROI_postReg'] = os.path.join(config['directories']['ROI_INPUT_DIR'],'goodreg_qual_ROIs')
    
    samples = [os.path.join(registrationOutputsDir,s) for s in os.listdir(registrationOutputsDir) if 'reformat' not in s]
    
    samples_postReg = [os.path.join(postRegROIDir,r) for r in os.listdir(postRegROIDir) if not r.endswith('.csv')]
    
    samples += samples_postReg
    print(samples)
        
    for sampleDir in samples:
        sampleID = str(sampleDir).split('/')[-1]
        print('On %s' %(sampleID))
        sample_dest = os.path.join(gen_dest_dir, sampleID)
        if not os.path.exists(sample_dest):
            os.mkdir(sample_dest)
        
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        
        for ROI in ROIS:
            ROI_dest = os.path.join(sample_dest,ROI)
            if not os.path.exists(ROI_dest):
                os.mkdir(ROI_dest)
                
            ROIDir = os.path.join(sampleDir,ROI)
            
            slides = [s for s in os.listdir(ROIDir) if s.endswith('.tif')]
        
            slides_w_number = [(v,str(v).split('.')[0].split('_')[-1]) for v in slides]
        
            sorted_slides = sorted(slides_w_number, key=lambda x: x[1], reverse=True)
        
            fixed_im = sorted_slides.pop()[0]
            Fixed_path = os.path.join(ROIDir, fixed_im)
        
            while len(sorted_slides)>0:
                moving, orderNum = sorted_slides.pop()
                target = str(moving).split('_')[2]
                Moving_path = os.path.join(ROIDir,moving)

                main(Moving_path, Fixed_path, target, orderNum, ROI)
    
    return config


if __name__ == '__main__':
    generate_genOverlays()
