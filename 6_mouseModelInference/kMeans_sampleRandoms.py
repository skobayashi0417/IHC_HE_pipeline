import os
import shutil
import random

def prepare_dirs(new_dest, clusters):
    for cluster in clusters:
        if not os.path.exists(os.path.join(new_dest,cluster)):
            os.mkdir(os.path.join(new_dest,cluster))

def sampleRandoms(BASE_DIR):
    kMeans_output_dirs = [k for k in os.listdir(BASE_DIR) if not k.endswith('.csv')]
    kMeans_output_dirs = [os.path.join(BASE_DIR,k) for k in kMeans_output_dirs if 'random_sampled' not in k]
    
    for kMeans_output_dir in kMeans_output_dirs:
        newDest = os.path.join(BASE_DIR,kMeans_output_dir.split('/')[-1]+'_random_sampled')
        if not os.path.exists(newDest):
            os.mkdir(newDest)
            
        clusters = [c for c in os.listdir(os.path.join(BASE_DIR,kMeans_output_dir))]
        
        prepare_dirs(newDest, clusters)
    
        for cluster in clusters:
            patches = [os.path.join(os.path.join(kMeans_output_dir,cluster),p) for p in os.listdir(os.path.join(kMeans_output_dir,cluster)) if p.endswith('.png')]
            
            random.shuffle(patches)
            patches = patches[0:5000]
            
            for patch in patches:
                shutil.copy(patch,os.path.join(os.path.join(newDest,cluster),patch.split('/')[-1]))
        
        
    

if __name__=='__main__':
    main()
