import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
import json
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DSS = ['5949','29','24','332']
TAM = ['20','26','28','991','992']
CTRL = ['5952','23','21','27','980','981','785','786']


### GENERAL STEPS:
### 1) Combine the IHC Count CSVs while creating columns for sample, ROI, condition --> save
### 2) Go through both kMeans CSVs whiel creating columns for sample, ROI. Create SF2 equivalent of fn_coordinfo (patch_fn in IHC count CSVs) so they can be merged --> combine sanve save
### 3) merge the Count CSVs with kMeans

def extract_sampleID(fn):
    return str(fn).split('_')[0]

def extract_ROI(fn):
    return str(fn).split('_')[1]

def return_sf2_coordInfo(fn):
    Strsplit = str(fn).split('.')[0].split('_')
    Strsplit[-4] = str(int(Strsplit[-4][:-1])*4)+'X'
    Strsplit[-3] = str(int(Strsplit[-3][:-1])*4)+'Y'
    Strsplit[-2]='w896'
    Strsplit[-1]='h896'
    return ('_').join(Strsplit[-4:])+'.png'
    
def decode_cluster(cluster):
    clus = str(cluster)
    if clus == '0':
        return 'Inflammatory'
    elif clus == '1':
        return 'CryptDropout'
    elif clus == '2':
        return 'CryptDilation'
    elif clus == '3':
        return 'DistortedGlands'

def decode_cluster_UNINV(cluster):
    clus = str(cluster)
    if clus == '0':
        return 'Crypts'
    elif clus == '1':
        return 'Rosettes'
    elif clus == '2':
        return 'LightlyPacked'

def give_str(fn):
    return str(fn)

def generate_invUnv_graph(graphs_DEST,finalMerged_df):
    outputs_dir = os.path.join(graphs_DEST,'invUnv_normalizedIHCCounts_acrossConditions')
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    invUnv = list(set(finalMerged_df['InvUnInv']))

    muted    = ["#ff0000", "#008000","#A47449","#8031A7"]
    newPal   = dict(UnInv = muted[1], Inv = muted[0])

    IHC_Markers = ['CD3','CD8b','CD4']


    finalMerged_df['InvUnInv'] = pd.Categorical(finalMerged_df['InvUnInv'],
                                    categories=['UnInv', 'Inv'],
                                    ordered=True)

    finalMerged_df['Condition'] = pd.Categorical(finalMerged_df['Condition'],
                                    categories=['CTRL', 'TAM','DSS'],
                                    ordered=True)

    CONDITIONS = list(set(finalMerged_df['Condition']))
    countTracker_forStats = {}
    countTracker_forStats_withinConditions = {}
    for condition in CONDITIONS:
        countTracker_forStats_withinConditions[condition]={}

    for IHC_Marker in IHC_Markers:
        plt.figure(figsize=(20, 10))

        #df['Condition'] = pd.Categorical(df['InvUnInv'],
        #                            categories=['UnInv', 'Inv'],
        #                            ordered=True)

        ax = sns.boxplot(x=finalMerged_df['Condition'], y=finalMerged_df[IHC_Marker], color='white', hue=finalMerged_df['InvUnInv'], showfliers = False, linewidth=4)

        # grouped stripplot
        ax = sns.stripplot(
            x="Condition", 
            y=IHC_Marker, 
            hue="InvUnInv", 
            data=finalMerged_df,
            dodge=True,
            palette = newPal
            )

                #ax = sns.stripplot(
                #    x="strkMeansCluster", 
                #    y="CD3", 
                #    hue="Condition", 
                #    data=df,
                #    dodge=True
                #    )
        plt.xlabel('')
        plt.ylabel('IHC Count per Patch',fontsize=20)
        plt.title(IHC_Marker,fontsize=30)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
                
        plt.show()

        fig = ax.get_figure()

        saveName = IHC_Marker + '_stripplot.png'
        savePath = os.path.join(outputs_dir,saveName)
        fig.savefig(savePath,bbox_inches = "tight")

        CONDITIONS = list(set(finalMerged_df['Condition']))
        InvUnInv = list(set(finalMerged_df['InvUnInv']))
        
        ## collect data for statistics, this is now subset to Inv vs Uninv and on an IHC marker

        countTracker_forStats[IHC_Marker] = {}
        #countTracker_forStats_withinConditions[IHC_Marker] = {}

        for state in InvUnInv:
            countTracker_forStats[IHC_Marker][state] = {}

            stateSubset = df[df['InvUnInv']==state]

            for condition in CONDITIONS:
                condSubset = stateSubset[stateSubset['Condition']==condition]
                countTracker_forStats[IHC_Marker][state][condition] = condSubset[IHC_Marker].tolist()
        
        for condition in CONDITIONS:
            countTracker_forStats_withinConditions[condition][IHC_Marker] = {}
            conditionSubset = df[df['Condition']==condition]

            for state in InvUnInv:
                stateSubset = conditionSubset[conditionSubset['InvUnInv']==state]
                countTracker_forStats_withinConditions[condition][IHC_Marker][state] = stateSubset[IHC_Marker].tolist()
    
    # generate stats
    statsTracker_withinConditions = {}
    for condition in CONDITIONS:
        statsTracker_withinConditions[condition] = []
        statsTracker_withinConditions[condition].append(['CD3',stats.ttest_ind(countTracker_forStats_withinConditions[condition]['CD3']['Inv'], countTracker_forStats_withinConditions[condition]['CD3']['UnInv'])])
        statsTracker_withinConditions[condition].append(['CD8b',stats.ttest_ind(countTracker_forStats_withinConditions[condition]['CD8b']['Inv'], countTracker_forStats_withinConditions[condition]['CD8b']['UnInv'])])
        statsTracker_withinConditions[condition].append(['CD4',stats.ttest_ind(countTracker_forStats_withinConditions[condition]['CD4']['Inv'], countTracker_forStats_withinConditions[condition]['CD4']['UnInv'])])

    with open('significances_compareInvUninv.txt', 'w') as convert_file:
        convert_file.write(json.dumps(statsTracker_withinConditions))

def generate_kMeansweighted_mouseConditions_graph(graphs_DEST,inv_df):
    outputs_dir = os.path.join(graphs_DEST,'UnInvnormalied_kMeansWeighted_acrossConditions')
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    inv_df['Condition'] = pd.Categorical(inv_df['Condition'],
                                categories=['CTRL', 'TAM', 'DSS'],
                                ordered=True)
    correctedMarkers = [a for a in inv_df.columns if 'avgCorrected_kMeansWeighted' in a]

    muted    = ["#ff0000", "#008000","#A47449","#8031A7"]
    newPal   = dict(CTRL = muted[1], TAM = muted[0], DSS = muted[2])

    inv_df['strkMeansCluster'] = pd.Categorical(inv_df['strkMeansCluster'],
                                    categories=['Inflammatory', 'CryptDropout', 'CryptDilation','DistortedGlands'],
                                    ordered=True)

    for IHC_marker in correctedMarkers:
        plt.figure(figsize=(20, 10))

        g = sns.catplot(
        data=inv_df, kind="bar",
        x="strkMeansCluster", y=IHC_marker, hue="Condition",
        palette=newPal, alpha=.6, height=6
        )
        print(IHC_marker)
        plt.show()

        saveName = IHC_marker + '_plot.png'
        savePath = os.path.join(outputs_dir,saveName)

        #fig = g.figure
        g.savefig(savePath,bbox_inches = "tight")

    ### STATS!!!
    patchClasses = list(set(inv_df['strkMeansCluster']))
    invUnv = list(set(inv_df['InvUnInv']))
    CONDITIONS = list(set(inv_df['Condition']))

    IHC_Markers_toTrack = [z for z in inv_df.columns if 'avgCorrected' in z]

    countTracker_forStats = {}
    ## collect data for statistics, this is now subset to Inv vs Uninv and on an IHC marker
    for IHC_Marker in IHC_Markers_toTrack:
        countTracker_forStats[IHC_Marker] = {}
        for patchClass in patchClasses:
            countTracker_forStats[IHC_Marker][patchClass] = {}

            patchClassSubset = inv_df[inv_df['strkMeansCluster']==patchClass]

            for condition in CONDITIONS:
                condSubset = patchClassSubset[patchClassSubset['Condition']==condition]
                countTracker_forStats[IHC_Marker][patchClass][condition] = condSubset[IHC_Marker].tolist()


    COMP_DICT = {0:'CTRL-DSS',
                1:'CTRL-TAM',
                2:'DSS_TAM'}
    ALL_CLUSTERS = list(set(inv_df['strkMeansCluster']))
    statsTracker = {}
    for IHC_Marker in IHC_Markers_toTrack:
        statsTracker[IHC_Marker] = {}
        for patchClass in patchClasses:
            statsTracker[IHC_Marker][patchClass] = []
            #for condition in CONDITIONS:
            #    statsTracker[IHC_Marker][patchClass][conditio] = []

    for ihcMarker_toTrack, value in countTracker_forStats.items():
        curDict = countTracker_forStats[ihcMarker_toTrack]

        for patchClass, value in curDict.items():
            fvalue, pvalue = stats.f_oneway(curDict[patchClass]['CTRL'],curDict[patchClass]['TAM'],
                                                curDict[patchClass]['DSS'])
            
            if pvalue < 0.05:
                    #create DataFrame to hold data
                        #https://www.statology.org/tukey-test-python/
                        df = pd.DataFrame({'value': curDict[patchClass]['CTRL'] + curDict[patchClass]['TAM'] + curDict[patchClass]['DSS'],
                    'group': ['CTRL']*len(curDict[patchClass]['CTRL']) + ['TAM']*len(curDict[patchClass]['TAM']) + ['DSS']*len(curDict[patchClass]['DSS'])}) 

                        tukey = pairwise_tukeyhsd(endog=df['value'],
                            groups=df['group'],
                            alpha=0.05)
                            
                        significant_indices = [i for i,v in enumerate(tukey.pvalues) if v <.05]

                        for significance in significant_indices:
                            statsTracker[ihcMarker_toTrack][patchClass] += [(COMP_DICT[significance],tukey.pvalues[significance])]

    with open('significances_kMeansweighted_avgCorrected.txt', 'w') as convert_file:
        convert_file.write(json.dumps(statsTracker))

def IHC_HE_aggregatedDF_graphs_generation(config):

    IHC_HE_DEST_DIR = os.path.join(config['directories']['BASE_DIR'],'IHC_HE_aggDFs_andGraphs')
    if not os.path.exists(IHC_HE_DEST_DIR):
        os.mkdir(IHC_HE_DEST_DIR)

    # start by eliminating the additional LA patches from IHC counts
    addtlLA_CSV = pd.read_csv('./7_combineHE_IHC/addtlLA/addtl_LA_patches.csv')

    countCSVsDir = config['directories']['IHC_CSV_Counts_ROI_newDetect']

    csvs = [c for c in os.listdir(countCSVsDir) if c.endswith('.csv')]
    ROI_Tracker = {}

    initialTrigger = 0
    for csv in csvs:
        sampleNum = str(csv).split('_')[-2]
        ROINum = str(csv).split('.')[0].split('_')[-1]
        
        cur_df = pd.read_csv(os.path.join(countCSVsDir,csv))

        addtlLA_subset = addtlLA_CSV[addtlLA_CSV['sample']==int(sampleNum)]
        toElim = [str(a)+'.png' for a in list(addtlLA_subset['patch_fn'])]

        cur_df_LA_addtlLA_elim = cur_df[~cur_df['patch_fn'].isin(toElim)]
        
        if sampleNum in DSS:
            toUploadCondition = len(cur_df_LA_addtlLA_elim)*['DSS']
        elif sampleNum in TAM:
            toUploadCondition = len(cur_df_LA_addtlLA_elim)*['TAM']
        elif sampleNum in CTRL:
            toUploadCondition = len(cur_df_LA_addtlLA_elim)*['CTRL']
        
        toUploadSampleID = len(cur_df_LA_addtlLA_elim)*[str(sampleNum)]
        toUploadROINum = len(cur_df_LA_addtlLA_elim)*[str(ROINum)]
        cur_df_LA_addtlLA_elim.insert(0,'Sample',toUploadSampleID)
        cur_df_LA_addtlLA_elim.insert(1,'ROI',toUploadROINum)
        cur_df_LA_addtlLA_elim.insert(2,'Condition',toUploadCondition)
        
        if initialTrigger == 0:
            agg_df = cur_df_LA_addtlLA_elim
            initialTrigger = 1
        elif initialTrigger != 0:
            agg_df = pd.concat([agg_df,cur_df_LA_addtlLA_elim],axis=0)
        
        ROI_Tracker[sampleNum] = ROINum

    # gather Kmeans data across mice into aggregate df
    kMeans_dir = config['directories']['KMEANS_OUTPUT_ROI_DIR']
    
    kMeans_csvs = [c for c in os.listdir(kMeans_dir) if c.endswith('.csv')]
    
    kMeans_initialTrigger = 0
    
    for kMeans_csv in kMeans_csvs:
        cur_df = pd.read_csv(os.path.join(kMeans_dir,kMeans_csv),header=None)
        cur_df.columns = ['fn','kMeansCluster']
        cur_df['Sample'] = cur_df['fn'].map(extract_sampleID)
        cur_df['ROI'] = cur_df['fn'].map(extract_ROI)
        cur_df['patch_fn'] = cur_df['fn'].map(return_sf2_coordInfo)
        
        if '_Involved_' in kMeans_csv:
            cur_df['strkMeansCluster'] = cur_df['kMeansCluster'].map(decode_cluster)
            toUploadInvUnInv = len(cur_df)*['Inv']
        elif '_Uninvolved' in kMeans_csv:
            cur_df['strkMeansCluster'] = cur_df['kMeansCluster'].map(decode_cluster_UNINV)
            toUploadInvUnInv = len(cur_df)*['UnInv']
        
        cur_df['InvUnInv'] = toUploadInvUnInv
            
        if kMeans_initialTrigger == 0:
            kMeans_agg_df = cur_df
            kMeans_initialTrigger = 1
        elif kMeans_initialTrigger != 0:
            kMeans_agg_df = pd.concat([kMeans_agg_df,cur_df],axis=0)
    
    tot_samples = list(set(agg_df['Sample']))
    
    
    # combine the IHC with kmeans data
    agg_initialTrigger = 0
    
    for sample in tot_samples:
        onlySample = agg_df[agg_df['Sample']==sample]
        
        ROI_subset_k = kMeans_agg_df[kMeans_agg_df['ROI']==ROI_Tracker[sample]]
        onlySample_k = ROI_subset_k[ROI_subset_k['Sample']==sample]
        
        merged = pd.merge(onlySample, onlySample_k, on=['patch_fn'])
        
        print('Sample: ' + sample)
        
        if agg_initialTrigger == 0:
            finalMerged_df = merged
            agg_initialTrigger = 1
        elif agg_initialTrigger != 0:
            finalMerged_df = pd.concat([finalMerged_df,merged],axis=0)
    
    finalMerged_df.to_csv(os.path.join(IHC_HE_DEST_DIR,'aggregatedkMeans_IHCCounts_ROI.csv'),index=False)

    ## generate final df with weighted kmeans (normalize counts within each mouse to average uninvovled count for that mouse and eliminate any that have an involved kmeans patch frequency <10% per mouse model)
    IHC_Markers = ['CD3','CD8b','CD4']

    inv_df = finalMerged_df[finalMerged_df['InvUnInv']=='Inv']
    conditions = list(set(finalMerged_df['Condition']))
    sampleInfoTracker = {}
    samples = list(set(df['Sample_x']))

    all_patchClusters_inv = list(set(inv_df['strkMeansCluster']))

    for sample in samples:
        sampleInfoTracker[sample] = {}
        sample_subset = dfinalMerged_dff[finalMerged_df['Sample_x']==sample]
        uninv_subset = sample_subset[sample_subset['InvUnInv']=='UnInv']

        sampleInfoTracker[sample]['UnInv_averages']={}
        for IHC_marker in IHC_Markers:
            sum_quantification = sum(list(uninv_subset[IHC_marker]))
            num_total = len(list(uninv_subset[IHC_marker]))
            avg = sum_quantification/num_total

            sampleInfoTracker[sample]['UnInv_averages'][IHC_marker] = avg

    for condition in conditions:
        sampleInfoTracker[condition] = {}
        condition_subset = inv_df[inv_df['Condition']==condition]
        tot_num_inv = len(condition_subset)
        for inv_class in all_patchClusters_inv:
            cur_subset = condition_subset[condition_subset['strkMeansCluster']==inv_class]
            num_cur_inv_class = len(cur_subset)
            inv_class_freq = float(num_cur_inv_class/tot_num_inv)
            print(inv_class)
            print(inv_class_freq)
            if inv_class_freq < .1:
                inv_class_freq = 0
            else:
                inv_class_freq = 1
            sampleInfoTracker[condition][inv_class] = inv_class_freq

    num_patches = len(inv_df)
    inv_df['CD3_perMouse_avgCorrected'] = [0]*num_patches
    inv_df['CD4_perMouse_avgCorrected'] = [0]*num_patches
    inv_df['CD8b_perMouse_avgCorrected'] = [0]*num_patches

    inv_df['CD3_perMouse_avgCorrected_kMeansWeighted'] = [0]*num_patches
    inv_df['CD4_perMouse_avgCorrected_kMeansWeighted'] = [0]*num_patches
    inv_df['CD8b_perMouse_avgCorrected_kMeansWeighted'] = [0]*num_patches

    for index, row in inv_df.iterrows():
        mouseNum =  row['Sample_x']
        inv_patchClass = row['strkMeansCluster']
        condition = row['Condition']

        rel_avg_uninvCount_CD3 = sampleInfoTracker[mouseNum]['UnInv_averages']['CD3']
        rel_avg_uninvCount_CD4 = sampleInfoTracker[mouseNum]['UnInv_averages']['CD4']
        rel_avg_uninvCount_CD8b = sampleInfoTracker[mouseNum]['UnInv_averages']['CD8b']
        #print(rel_avg_uninvCount_CD3)
        #print(float(int(row['CD3'])/rel_avg_uninvCount_CD3))
        #row['CD3_avgCorrected'] = float(int(row['CD3'])/rel_avg_uninvCount_CD3)
        #rche_df.loc[index, 'wgs1984_latitude'] = dict_temp['lat']
        inv_df.loc[index,'CD3_perMouse_avgCorrected'] = float(int(row['CD3'])/rel_avg_uninvCount_CD3)
        inv_df.loc[index,'CD4_perMouse_avgCorrected'] = float(int(row['CD4'])/rel_avg_uninvCount_CD4)
        inv_df.loc[index,'CD8b_perMouse_avgCorrected'] = float(int(row['CD8b'])/rel_avg_uninvCount_CD8b)
        #row['CD3_avgCorrected'] = 5
        #row['CD4_avgCorrected'] = float(int(row['CD4'])/rel_avg_uninvCount_CD4)
        #row['CD8b_avgCorrected'] = float(int(row['CD8b'])/rel_avg_uninvCount_CD8b)
        inv_df.loc[index,'CD3_perMouse_avgCorrected_kMeansWeighted'] = float(int(row['CD3'])/rel_avg_uninvCount_CD3)*sampleInfoTracker[condition][inv_patchClass] 
        inv_df.loc[index,'CD4_perMouse_avgCorrected_kMeansWeighted'] = float(int(row['CD4'])/rel_avg_uninvCount_CD4)*sampleInfoTracker[condition][inv_patchClass] 
        inv_df.loc[index,'CD8b_perMouse_avgCorrected_kMeansWeighted'] = float(int(row['CD8b'])/rel_avg_uninvCount_CD8b)*sampleInfoTracker[condition][inv_patchClass]         

    inv_df.to_csv(os.path.join(IHC_HE_DEST_DIR,'invPatches_IHC_HE_kMeansWeighted.csv'),index=False)
    
    graphs_DEST = os.path.join(IHC_HE_DEST_DIR,'graphs')
    if not os.path.exists(graphs_DEST):
        os.mkdir(graphs_DEST)

    # generate graphs of IHC count between Uninvolved and Involved across mouse conditions
    generate_invUnv_graph(graphs_DEST,finalMerged_df)

    # generate graphs of kMeans normalized counts across patch classes across mouse conditions
    generate_kMeansweighted_mouseConditions_graph(graphs_DEST,inv_df)

if __name__ == '__main__':
    main()
