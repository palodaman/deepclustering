import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler

def load_data(filepath, format = 'h5'):
    if format == 'h5':
        data = sc.read_10x_h5(filepath, genome = 'X')

        # with h5py.File(filepath, 'r') as hf:
        #     # print("Heys in file:", hf.keys)
        #     read = lambda name, obj : print(name)
        #     hf.visititems(read)
        #     print("")

    elif format == 'csv':
        data = pd.read_csv(
            filepath_or_buffer = filepath, 
            index_col = 0
        )
    
    return data

def preprocess(data, should_norm = True, should_scale = True,
               should_log = True, n_top_genes = 2000):
        
        print("=== DATA: ===\n", data, "\n\n\n", sep = '', end = '')

        if should_log:
             # Not sure why log1p vs log, but they use log1p
             data = np.log1p()

        # If we have more than our limit, we need to cut down.
        # This chooses the genes with highest variance,
        # but we may wish to use other methods...
        if n_top_genes is not None and data.shape[1] > n_top_genes:
             var = np.var(data, axis = 0)
             top_indices = np.argsort(var)[-n_top_genes:]
             data = data[:, top_indices]

        # This is "library size normalization", hopefully.
        if should_norm:
             sizes = np.sum(data, axis = 1)
             data = data / sum[:, None] * np.median(sizes)
            
        if should_scale:
             scaler = StandardScaler()
             data = scaler.fit_transform(data)

        print("=== DATA AFTER: ===\n", data, "\n\n\n", sep = '', end = '')

        return data, scaler


# d = load_data('scDeepClustering_Sample_Data/mouse_bladder_cell_select_2100.h5', 'h5')