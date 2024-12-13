import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import h5py

def load_data(filepath, format):
    data = None
    labels = None 
    if format == 'h5':
        try:
            with h5py.File(filepath, 'r') as hf:
                print("Keys in file:", list(hf.keys()))
                #extract 'X' dataset
                data = hf['X'][:] 
                if 'Y' in hf: #check if labels exist
                    labels = hf['Y'][:]
        except Exception as e:
            print(f"Error reading h5 file: {e}")
            return None, None #return None for both if error

    elif format == 'csv':
        print("inside csv format")
        try:
            data = pd.read_csv(filepath_or_buffer=filepath, index_col=0).values
        except Exception as e:
            print(f"Error reading csv file: {e}")
            return None, None
    else:
        print("Format not supported")
        return None, None
    
    if data is None:
        print("data wasnt loaded")
        return None, None
    return data, labels

def preprocess(data, should_norm = True, should_scale = True,
               should_log = True, n_top_genes = 2000):
        
        print("=== DATA: ===\n", data, "\n\n\n", sep = '', end = '')
        if should_log:
             data = np.log1p(data)


        if n_top_genes is not None and data.shape[1] > n_top_genes:
             var = np.var(data, axis = 0)
             top_indices = np.argsort(var)[-n_top_genes:]
             data = data[:, top_indices]

        if should_norm:
             sizes = np.sum(data, axis = 1)
             data = data / sizes[:, None] * np.median(sizes)
            
        if should_scale:
             scaler = StandardScaler()
             data = scaler.fit_transform(data)

        print("=== DATA AFTER: ===\n", data, "\n\n\n", sep = '', end = '')

        return data, scaler


#testing
# d, e = load_data('scDeepClustering_Sample_Data/mouse_bladder_cell_select_2100.h5', 'h5')
# print(d)
# print("this is Y", e)