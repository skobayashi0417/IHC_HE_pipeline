import os
import json
import torch
import sys
import argparse
sys.path.insert(0,'./ROI_CODE')
from wsiregister_preRegisterROIs import *
from generalOverlayCheck_prePostROIs import *
from generateJSON import *

def prepare_directories(config):
    BASE_DIR = config['directories']['BASE_DIR']
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)


def runPipeline_part1(config):
    
    good_qual_reg_samples = ['980']
    bad_qual_reg_samples = ['24']
    
    # extract ROI from samples with bad registration quality
    print('registering ROI from bad registration quality samples...')
    config = registerWSIs_preROI(config)
    
    print('generating overlays of registered ROIs...')
    config = generate_genOverlays_composite(config)
    
    
    with open('config_part2.json','w') as outfile:
        json.dump(config,outfile)
    
if __name__ == '__main__':
    generate_JSON()
    config = json.load(open('config.json'))
    
    prepare_directories(config)
    
    runPipeline_part1(config)
