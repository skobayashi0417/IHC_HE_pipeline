import os
import pandas as pd


def return_sum(a):
    ### a should be a list of what you want to sum up
    assert isinstance(a,list)
    counts = [int(b) for b in a]
    return sum(counts)

def return_proportion(clusterCount, totalCount):
    clusterProportion = float((clusterCount/totalCount)*100)

    return clusterProportion


def generateProps_ROI(config):
    test_df = pd.read_csv(os.path.join(config['directories']['GEN_PATCHES_ROI_DIR'],'invUninvolvedPatchCounts_test.csv'))

    # sum up all patch counts in each
    test_df['totalCount'] = test_df.apply(lambda x: return_sum(a=[x['UNinvCount'],x['Cluster_Inv_0'],x['Cluster_Inv_1'],x['Cluster_Inv_2'],x['Cluster_Inv_3']]),axis=1)

    # calculate percentages
    for col in test_df.columns:
        if str(col) == 'sample':
            next
        elif str(col) == 'ROI':
            next
        elif str(col) == 'totalCount':
            next
        else:
            newColName = str(col) + '_percent'
            test_df[newColName] = test_df.apply(lambda x: return_proportion(clusterCount=x[col],totalCount=x['totalCount']),axis=1)
    

    test_df.to_csv(os.path.join(config['directories']['GEN_PATCHES_ROI_DIR'],'invUninvolvedPatchCounts_test_wProps.csv'))
    
    
    
    return config

if __name__ == '__main__':
    main()

