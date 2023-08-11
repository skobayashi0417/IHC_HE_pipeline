import json

def generate_JSON():

    data = {}

    data['directories'] = {}
    data['SlideInfo'] = {}
    data['PatchInfo'] = {}

    data['directories']['ROI_INPUT_DIR'] = '/data01/shared/skobayashi/ROI_tifs'
    data['directories']['BASE_DIR'] = '/data1/shared/skobayashi/github_test/'

    data['SlideInfo']['WSI_EXTENSION'] = '.tif'
    data['SlideInfo']['SCALE_FACTOR'] = 2
    data['SlideInfo']['SAVE_CROPPED_WSI'] = False

    data['PatchInfo']['PATCH_SIZE_SF8'] = 224
    data['PatchInfo']['PATCH_SIZE_SF2'] = 896
    data['PatchInfo']['meshPATCH_SIZE'] = 32
    data['SlideInfo']['IHC_Order'] = ['CD8b','CD3','CD4']

    data['DEVICE'] = 0

    with open('config.json','w') as outfile:
        json.dump(data,outfile)

if __name__=='__main__':
    generate_JSON()
