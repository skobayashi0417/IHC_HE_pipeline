import argparse
from torchvision import transforms
import time, os, sys, glob, copy
from time import strftime
#from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from utils_predict_5Class import *
from torchvision import datasets, transforms, models

rand_seed = 26700

if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    
def get_data_transforms(mean, std, PATCH_SIZE):
    data_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop(PATCH_SIZE),
            transforms.Resize(PATCH_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return data_transforms

def get_model(DEVICE):
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, 2)  # pathology vs healthy
    model = model.to("cuda:" + str(DEVICE))
    if torch.cuda.device_count() >= 2:  # use multiple GPUs
        model = torch.nn.DataParallel(model, device_ids=[DEVICE, 1])
        cudnn.benchmark = True
        print('using multiple GPUs')

    return model

def adjust_label(x):
    x = int(x)+1
    x = str(x)
    return x
    
def find_max_prob(x):
    maximum = max(x)
    stringed = str(maximum)
    return stringed

def find_bg_prob(x):
    bgProb = x[0]
    bgProb = str(bgProb)
    return bgProb

def find_muscle_prob(x):
    muscleProb = x[1]
    muscleProb = str(muscleProb)
    return muscleProb

def find_tissue_prob(x):
    tissueProb = x[2]
    tissueProb = str(tissueProb)
    return tissueProb

def find_submucosa_prob(x):
    submucosaProb = x[3]
    submucosaProb = str(submucosaProb)
    return submucosaProb

def test_fn_epoch(model, criterion, test_loader):
    global ROI_save_dir
    global device

    model.eval()
    nline = 0

    savef = os.path.join(ROI_save_dir,'adjustlabel_predicts.csv')
    savea = os.path.join(ROI_save_dir,'origlabel_predicts.csv')

    f = open(savef,"w")
    a = open(savea,"w")

    preds_test = torch.zeros(0).type(torch.LongTensor).to(device)
    with torch.no_grad():
        for i, (images, path, fn) in enumerate(test_loader,0):
            images = Variable(images.to(device))
            outputs = model(images)
            
            _, preds = torch.max(outputs.data, 1) # get the argmax index along the axis 1
            preds_test = torch.cat((preds_test,preds))
            
            probabilities = F.softmax(outputs,dim=-1)
            pred = torch.max(outputs,1)[1]
            #labels_test = labels_test.cpu().tolist()
            f.write('\n'.join([
                ', '.join(x)
                for x in zip(map(find_bg_prob,probabilities.cpu().tolist()),map(find_muscle_prob,probabilities.cpu().tolist()),map(find_tissue_prob,probabilities.cpu().tolist()),map(find_submucosa_prob,probabilities.cpu().tolist()),map(find_max_prob,probabilities.cpu().tolist()),map(adjust_label,pred.cpu().tolist()),fn)
                ]) + '\n')
            a.write('\n'.join([
                ', '.join(x)
                for x in zip(map(find_bg_prob,probabilities.cpu().tolist()),map(find_muscle_prob,probabilities.cpu().tolist()),map(find_tissue_prob,probabilities.cpu().tolist()),map(find_submucosa_prob,probabilities.cpu().tolist()),map(find_max_prob,probabilities.cpu().tolist()),map(str,pred.cpu().tolist()),fn)
                ]) + '\n')
        f.close()
        a.close()
        
    preds_test2 = preds_test.cpu().numpy()
        
    return preds_test

def log_summary(Pr2):
    global timestr
    global ROI_save_dir
    global sample
        
    savePr = os.path.join(ROI_save_dir,'Pr.csv')

    np.savetxt(savePr,Pr2)


def run_predictions(testDataSet, DEVICE, checkpoint_path):
    global device
    
    model = get_model(DEVICE=DEVICE)
    
    checkpoint = torch.load(checkpoint_path)
    #print(checkpoint)
    
    model = checkpoint['model'].module

    criterion = nn.CrossEntropyLoss().to(device)
    start = time.time()

    Pr = test_fn_epoch(model=model, criterion=criterion, test_loader = testDataSet)

    Pr2 = Pr.to(device).cpu().numpy()
    
    log_summary(Pr2)
            
def perform_Meshpredictions_ROI_wLA_sepDest(config):
    global timestr
    global data_transforms
    global ROI_save_dir
    global sample
    global device
    
    BASE_DIR = config['directories']['BASE_DIR']
    DEVICE = config['DEVICE']
    PATCH_SIZE = config['PatchInfo']['meshPATCH_SIZE']
    testSource = config['directories']['bySampleMeshPatches_ROI']
    GEN_DEST_DIR = config['directories']['DEST_DIR']
    
    MESH_PREDICTIONS_DIR = os.path.join(GEN_DEST_DIR,'meshPatch_Predictions_ROI_wLA_sepDest')
    if not os.path.exists(MESH_PREDICTIONS_DIR):
        os.mkdir(MESH_PREDICTIONS_DIR)
        
    config['directories']['meshPREDICTIONS_DIR_ROI_wLA'] = MESH_PREDICTIONS_DIR
    
    device = torch.device("cuda:" + str(DEVICE))
    
    mean = torch.tensor([0.7861, 0.6368, 0.7648],dtype=torch.float32)
    std = torch.tensor([0.0740, 0.1025, 0.0425],dtype=torch.float32)
    
    data_transforms = get_data_transforms(mean, std, PATCH_SIZE)
    
    checkpoint_path = './2_Inference/smallPatch_model_wLA/smallPatch_wLA_chkpt.t7'
    
    timestr = time.strftime("%Y%m%d-%H%M%S")

    samples = [s for s in os.listdir(testSource) if not str(s).endswith('.csv')]
        
    tot_num = len(samples)
        
    counter = 1
        
    for sample in samples:
        #print('Performing predictions on sample  %s.. ------ %d out of %d total samples.' % (sample,counter,tot_num))
        
        sampleDir = os.path.join(testSource,sample)
        save_dir = os.path.join(MESH_PREDICTIONS_DIR,str(sample))
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        for ROI in ROIS:
            print('Performing mesh predictions on FIXED_HE_1 image for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total samples.' % (sample,ROI,ROICOUNTER,len(ROIS),counter,tot_num))
            ROI_save_dir = os.path.join(save_dir,ROI)
            if not os.path.exists(ROI_save_dir):
                os.mkdir(ROI_save_dir)
                
            img_test = [f for f in glob.glob(os.path.join(os.path.join(sampleDir,ROI), '*png'))]

            test_set = data_loader(img_test, transform=data_transforms['test'])
            
            run_predictions(testDataSet = DataLoader(test_set, batch_size = 32, num_workers = 8 ),DEVICE=DEVICE, checkpoint_path = checkpoint_path)
            
            ROICOUNTER += 1
            
        counter += 1
    
    return config
    
