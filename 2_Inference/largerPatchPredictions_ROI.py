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
from utils_predict_4Class import *
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

def find_healthy_prob(x):
    healthyProb = x[0]
    healthyProb = str(healthyProb)
    return healthyProb

def find_path_prob(x):
    pathProb = x[1]
    pathProb = str(pathProb)
    return pathProb

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

            #print(str(labels_test.cpu().tolist()))
            images = Variable(images.to(device))
            outputs = model(images)
            
            _, preds = torch.max(outputs.data, 1) # get the argmax index along the axis 1
            preds_test = torch.cat((preds_test,preds))
            
            probabilities = F.softmax(outputs,dim=-1)
            pred = torch.max(outputs,1)[1]
            #labels_test = labels_test.cpu().tolist()
            f.write('\n'.join([
                ', '.join(x)
                for x in zip(map(find_healthy_prob,probabilities.cpu().tolist()),map(find_path_prob,probabilities.cpu().tolist()),map(find_max_prob,probabilities.cpu().tolist()),map(adjust_label,pred.cpu().tolist()),fn)
                ]) + '\n')
            a.write('\n'.join([
                ', '.join(x)
                for x in zip(map(find_healthy_prob,probabilities.cpu().tolist()),map(find_path_prob,probabilities.cpu().tolist()),map(find_max_prob,probabilities.cpu().tolist()),map(str,pred.cpu().tolist()),fn)
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


def run_predictions(testDataSet, DEVICE):
    global device
    
    model = get_model(DEVICE=DEVICE)
    checkpoint = torch.load('./2_Inference/model/resnet34_chkpt.t7')
    model = checkpoint['model'].module

    criterion = nn.CrossEntropyLoss().to(device)
    start = time.time()

    Pr = test_fn_epoch(model=model, criterion=criterion, test_loader = testDataSet)

    Pr2 = Pr.to(device).cpu().numpy()
    
    log_summary(Pr2)
            
def perform_predictions_ROI(config):
    global timestr
    global data_transforms
    global ROI_save_dir
    global sample
    global device
    
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    
    DEVICE = config['DEVICE']
    PATCH_SIZE = config['PatchInfo']['PATCH_SIZE_SF8']
    
    device = torch.device("cuda:" + str(DEVICE))
    
    mean = torch.tensor([0.7480, 0.6054, 0.7506], dtype = torch.float32)
    std = torch.tensor([0.1190, 0.1661, 0.0663], dtype = torch.float32)
    
    data_transforms = get_data_transforms(mean, std, PATCH_SIZE)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
            
    testSource = config['directories']['extractedPatches_HE_sf8_wOverlaps_ROI']
    PREDICTIONS_DIR = os.path.join(BASE_DIR, 'predictions_wOverlaps_ROI')
    if not os.path.exists(PREDICTIONS_DIR):
        os.mkdir(PREDICTIONS_DIR)
    
    config['directories']['PREDICTIONS_ROI_DIR'] = PREDICTIONS_DIR

    samples = [s for s in os.listdir(testSource) if not str(s).endswith('.csv')]

    tot_num = len(samples)
    
    counter = 1
    
    for sample in samples:
        sample_save_dir = os.path.join(PREDICTIONS_DIR,sample)
        if not os.path.exists(sample_save_dir):
            os.mkdir(sample_save_dir)
    
        sampleDir = os.path.join(testSource,sample)
        ROIS = [r for r in os.listdir(sampleDir) if 'ROI' in r]
        ROICOUNTER = 1
        
        for ROI in ROIS:
            print('Performing sf8 patch predictions for sample %s: %s.. (%d/%d ROIs) ------ %d out of %d total samples.' % (sample,ROI,ROICOUNTER,len(ROIS),counter,tot_num))
            
            ROI_save_dir = os.path.join(sample_save_dir,ROI)
            if not os.path.exists(ROI_save_dir):
                os.mkdir(ROI_save_dir)
            
            ROI_patchDir = os.path.join(sampleDir,ROI)
    
            img_test = [f for f in glob.glob(os.path.join(ROI_patchDir, '*png'))]

            test_set = data_loader(img_test, transform=data_transforms['test'])
        
            run_predictions(testDataSet = DataLoader(test_set, batch_size = 32, num_workers = 8 ),DEVICE=DEVICE)
            
            ROICOUNTER += 1
        
        counter += 1
    
    return config
