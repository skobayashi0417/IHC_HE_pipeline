import argparse
from torchvision import transforms
import time, os, sys, glob, copy
from time import strftime
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from utils_predict_2Class import *
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rand_seed = 26700
device = torch.device("cuda:0")


if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    
def get_data_transforms(mean, std):
    data_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return data_transforms

def get_model(net_depth, n_class):
    if net_depth == 34:
        model = models.resnet34(pretrained=True)
    elif net_depth == 50:
        model = models.resnet50(pretrained=True)
    elif net_depth == 101:
        model = models.resnet101(pretrained=True)
    elif net_depth == 152:
        model = models.resnet152(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, n_class)  # benign vs. tumor
    model = model.to(device)
    if torch.cuda.device_count() >= 2:  # use multiple GPUs
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        cudnn.benchmark = True

    return model
    
def string_noBrackets(x):
    x = str(x)
    x = x.replace('[','')
    x = x.replace(']','')
    return x

def test_fn_epoch(model, criterion, test_loader, patchType):
    global dest_dir

    model.eval()
    nline = 0
    running_loss = 0.0
    
    if patchType == 'Involved':
        save_all = os.path.join(dest_dir,'involved_ROI_wOverlap_RN_extractedFeatures.csv')
    elif patchType == 'Uninvolved':
        save_all = os.path.join(dest_dir,'UNinvolved_ROI_wOverlap_RN_extractedFeatures.csv')
    a = open(save_all,"w")

    with torch.no_grad():
        for i, (images, path, fn) in enumerate(test_loader,0):
            images = Variable(images.to(device))
            outputs = model(images)

            toSaveOutputs = outputs.cpu()
            a.write('\n'.join([
                ', '.join(x)
                for x in zip(fn,map(string_noBrackets,toSaveOutputs.tolist()))
                ]) + '\n')
                
def run_predictions(runinfo, test_loader, patchType):
    global model
    global classes
    #global runNumber

    #runNumber = runinfo[1]
    modelpath = runinfo[3]
    model = get_model(net_depth = 34, n_class = 2)

    checkpoint = torch.load(modelpath)
    model = checkpoint['model'].module

    model = torch.nn.Sequential(*list(model.children())[:-1]) # removes last layer for feature extraction
    
    criterion = nn.CrossEntropyLoss().to(device)
    start = time.time()

    test_fn_epoch(model=model, criterion=criterion, test_loader = test_loader, patchType = patchType)
 
def extract_top_model(checkpointPath):
    checkpoint = [a for a in os.listdir(checkpointPath) if a.endswith('.t7')][0]
    
    bestPath = os.path.join(checkpointPath,checkpoint)
    
    return bestPath
            
    
def RN_FeatureExtraction_ROI(config,patchType):
    global data_transforms
    global dest_dir
    
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    
    # using same RN model that performed Inv vs Uninv predictions as feature extractor. Thus, same Mean and std for either patch type and same checkpoint
    meansandstd = [[0.7480, 0.6054, 0.7506],[0.1190, 0.1661, 0.0663]]
        
    mean = torch.tensor(meansandstd[0], dtype=torch.float32)
    std = torch.tensor(meansandstd[1], dtype=torch.float32)
    CHECKPOINT_DIR = './2_Inference/model/'
    
    # extract checkpoint path and ready the trnsforms
    run_info = [[32,1,'',extract_top_model(CHECKPOINT_DIR)]]
    data_transforms = get_data_transforms(mean, std)
    
    if patchType == 'Involved':
        dest_dir = os.path.join(BASE_DIR,'involved_ROI_patch_RN_FeatureExtraction')
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        
        config['directories']['INVOLVED_ROI_RN_EXTRACTED_FEATURES_DIR'] = dest_dir
        
        testSource = config['directories']['INVOLVED_PATCHES_ROI_DIR']
                
        img_test = [f for f in glob.glob(os.path.join(testSource, '*png'))]
        test_set = data_loader(img_test, transform=data_transforms['test'])
        test_loader = DataLoader(test_set, batch_size = 32, num_workers = 8 )
        run_predictions(runinfo = run_info[0], test_loader = test_loader, patchType = patchType)
    
    elif patchType == 'Uninvolved':
        dest_dir = os.path.join(BASE_DIR,'UNinvolved_ROI_patch_RN_FeatureExtraction')
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        
        config['directories']['UNINVOLVED_ROI_RN_EXTRACTED_FEATURES_DIR'] = dest_dir
        
        testSource = config['directories']['UNINVOLVED_PATCHES_ROI_DIR']
                
        img_test = [f for f in glob.glob(os.path.join(testSource, '*png'))]
        test_set = data_loader(img_test, transform=data_transforms['test'])
        test_loader = DataLoader(test_set, batch_size = 32, num_workers = 8 )
        run_predictions(runinfo = run_info[0], test_loader = test_loader, patchType = patchType)
        
        
    
    return config
