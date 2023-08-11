import os
import pandas as pd
import numpy as np
import csv

def extract_sampleID_UNInvolved(fn):
    sampleID = str(fn)[1:].split('_')[0]
    return sampleID

def extract_sampleID(fn):
    sampleID = str(fn).split('_')[0]
    return sampleID

def extract_fncoordinfo(fn):
    if 'scaledFactor8' in fn:
        return ('_').join(str(fn).split('_')[5:])
    else:
        return ('_').join(str(fn).split('_')[1:])

def extract_ROI(fn):
    return str(fn).split('_')[1]
        
def generate_sf2_fncoordinfo(fn_sf8):
    Xcoord = str(int(str((fn_sf8)).split('_')[-4][:-1])*4)
    Ycoord = str(int(str(fn_sf8).split('_')[-3][:-1])*4)

    split_list = str(fn_sf8).split('_')
    split_list[-4] = Xcoord + 'X'
    split_list[-3] = Ycoord + 'Y'
    split_list[-2] = 'w896'
    split_list[-1] = 'h896.png'
    
    return ('_').join(split_list)

def mergeCounts_ROI(config):
    BASE_DIR = config['directories']['INV_UNV_wIHC_BASE_DIR_ROI']
    
    KMEANS_DIR = config['directories']['KMEANS_OUTPUT_ROI_DIR']
    
    IHC_COUNTS_DIR = config['directories']['IHC_CSV_Counts_ROI_newDetect']
    
    # load involved df
    involved_df = pd.read_csv(os.path.join(KMEANS_DIR,[a for a in os.listdir(KMEANS_DIR) if a.endswith('Involved_ROI_wOverlaps_test.csv')][0]), header=None)
    involved_df.columns = ['fn','Cluster']
    involved_df['sample'] = involved_df['fn'].map(extract_sampleID)
    
    # load uninvovled df
    #UNinvolved_df = pd.read_csv(os.path.join(config['directories']['GEN_PATCHES_DIR'],'merged_Uninvolved.csv'),header=None)
    #UNinvolved_df.columns = ['healthyProb','pathProb','maxProb','Pred','fn']
    #UNinvolved_df['sample'] = UNinvolved_df['fn'].map(extract_sampleID_UNInvolved)
    UNinvolved_df = pd.read_csv(os.path.join(KMEANS_DIR,[a for a in os.listdir(KMEANS_DIR) if a.endswith('Uninvolved_ROI_wOverlaps_test.csv')][0]), header=None)
    UNinvolved_df.columns = ['fn','Cluster']
    UNinvolved_df['sample'] = UNinvolved_df['fn'].map(extract_sampleID)
    
    # get involved and uninvolved clusterss
    invClusters = list(set(involved_df['Cluster']))
    UNinvClusters = list(set(UNinvolved_df['Cluster']))
    
    # get ROIs
    involved_df['ROI'] = involved_df['fn'].map(extract_ROI)
    UNinvolved_df['ROI'] = UNinvolved_df['fn'].map(extract_ROI)
    
    # get samples.. should be same in both involved and uninvolved dfs
    samples = list(set(list(involved_df['sample'])))
    
    samples = [s for s in samples if s!='23']

    tracker = []
    
    for sample in samples:
        print(sample)
        IHC_count_csvs = [a for a in os.listdir(IHC_COUNTS_DIR) if sample in a]
        
        initial_subset = involved_df[involved_df['sample']==str(sample)]
        initial_subset_UNinvolved = UNinvolved_df[UNinvolved_df['sample']==str(sample)]
        
        print(initial_subset.head)
        print(initial_subset_UNinvolved.head)
        
        ROIs = list(set(list(initial_subset['ROI']) + list(initial_subset_UNinvolved['ROI'])))
        
        print(ROIs)
        print('check')
        
        for ROI in ROIs:
            print(ROI)
            # start mouseTracker
            sampleTracker = [sample]
            sampleTracker.append(ROI)
            
            # subset involved_df for this mouse ROI
            subset = initial_subset[initial_subset['ROI']==ROI]
            print(subset.head)
            
            # subset UNinvolved_df for this mouse ROI
            subset_UNinvolved = initial_subset_UNinvolved[initial_subset_UNinvolved['ROI']==ROI]
            print(subset_UNinvolved.head)
            
            # log number of uninvolved patches... we still need this for LDA inference
            sampleTracker.append(len(subset_UNinvolved))
        
            # iterate and log involved kMeans cluster counts
            for invCluster in invClusters:
                sampleTracker.append(len(subset[subset['Cluster']==invCluster]))
            
            # iterate and log UNinvolved kMeans cluster counts
            for UNinvCluster in UNinvClusters:
                sampleTracker.append(len(subset_UNinvolved[subset_UNinvolved['Cluster']==UNinvCluster]))
            
            # append to overall tracker
            tracker.append(sampleTracker)
            
            # extract all patch fns
            all_fns = list(subset['fn'])
            all_fns_UNinvolved = list(subset_UNinvolved['fn'])
            #print(all_fns)
        
            # only get original patches
            all_fns = [f for f in all_fns if 'scaledFactor8' in f]
            all_fns_UNinvolved = [f for f in all_fns_UNinvolved if 'scaledFactor8' in f]
            #print(all_fns)
        
            # re-subset both subset to include only these original patches... we want to append this info to IHC_Counts df for these patches
            further_subset = subset[subset['fn'].isin(all_fns)]
            further_subset_UNinvolved = subset_UNinvolved[subset_UNinvolved['fn'].isin(all_fns_UNinvolved)]
        
            # generate fn_nocoordinfo IDs
            further_subset['fn_coordinfo'] = further_subset['fn'].map(extract_fncoordinfo)
            further_subset_UNinvolved['fn_coordinfo'] = further_subset_UNinvolved['fn'].map(extract_fncoordinfo)

            # convert to sf2 versions to match up with IHC counts fns later
            further_subset['fn_coordinfo_sf2'] = further_subset['fn_coordinfo'].map(generate_sf2_fncoordinfo)
            further_subset_UNinvolved['fn_coordinfo_sf2'] = further_subset_UNinvolved['fn_coordinfo'].map(generate_sf2_fncoordinfo)
        
            # gather the fns and also kMeans info
            kMeans_info = list(zip(list(further_subset['fn_coordinfo_sf2']),[str(c) for c in further_subset['Cluster']],['Involved']*len(further_subset)))
            #print(further_subset.head)
            #print(len(kMeans_info))
        
            # add on the uninvolved patches
            kMeans_info += list(zip(list(further_subset_UNinvolved['fn_coordinfo_sf2']),['Uninvolved']*len(further_subset_UNinvolved),[str(c) for c in further_subset_UNinvolved['Cluster']]))
            #print(len(kMeans_info))
            #print(further_subset_UNinvolved.head)
        
            # open relevant IHC count CSV
            print(further_subset.head)
            print(ROI)
            IHC_count_csv = pd.read_csv(os.path.join(IHC_COUNTS_DIR,[a for a in IHC_count_csvs if ROI in a][0]))
        
            # extract fns
            IHC_counts_fns = list(IHC_count_csv['patch_fn'])
            #print(len(IHC_counts_fns))
            # make sure length of file names matches
            print(len(kMeans_info))
            print(len(IHC_counts_fns))
            assert len(kMeans_info) == len(IHC_counts_fns)
        
            # sort kmeans info to match order in the IHC_count CSV
            print(kMeans_info[0:10])
            print(IHC_counts_fns[0:10])
            sorted_kmeans_info = sorted(kMeans_info, key = lambda x: IHC_counts_fns.index(x[0]))
        
            # update the IHC_counts_csv
            IHC_count_csv['Inv_kMeans_cluster'] = [a[1] for a in sorted_kmeans_info]
            IHC_count_csv['UNinv_kMeans_cluster'] = [a[2] for a in sorted_kmeans_info]
        
            # save it
            IHC_count_csv.to_csv(os.path.join(IHC_COUNTS_DIR,[a for a in IHC_count_csvs if ROI in a][0]),index=False)
        
         
    
    ## save info to df ##
    # convert involvedClusters into ColNames
    invClasses = ['Cluster_Inv_' + str(z) for z in invClusters]
    UNinvClasses = ['Cluster_Uninv_' + str(z) for z in UNinvClusters]
    
    appendDict = {}

    with open(os.path.join(config['directories']['GEN_PATCHES_ROI_DIR'],'invUninvolvedPatchCounts_test.csv'),'w') as csvfile:
        fieldnames = ['sample','ROI','UNinvCount'] + invClasses + UNinvClasses
                    
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for i in range(len(tracker)):
            toAppend = tracker[i]
            appendDict['sample']=toAppend[0]
            appendDict['ROI']=toAppend[1]
            appendDict['UNinvCount']=toAppend[2]
            appendDict[fieldnames[3]]=toAppend[3]
            appendDict[fieldnames[4]]=toAppend[4]
            appendDict[fieldnames[5]]=toAppend[5]
            appendDict[fieldnames[6]]=toAppend[6]
            appendDict[fieldnames[7]]=toAppend[7]
            appendDict[fieldnames[8]]=toAppend[8]
            appendDict[fieldnames[9]]=toAppend[9]
            csvwriter.writerow(appendDict)
    
    return config
        
if __name__=='__main__':
    main()
