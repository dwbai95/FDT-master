import numpy as np
import pandas as pd
import csv




def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)

        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]
        
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j], A[j, i] = 1, 1

    return A




def load_weighted_adjacency_matrix(file_path, n_vertex): 

    df = get_adjacency_matrix(file_path, n_vertex)
        

    return df


def load(file_path):
    
    try:
        data = pd.read_csv(file_path, header=None) 
        data = np.array(data)
    except:
        try:
            data = pd.read_hdf(file_path)
            data = np.array(data)
        except:
            try:
                data = np.load(file_path)
                data = data[data.files[0]]
            except:
                data = np.load(file_path)
    if data.ndim == 2:
        
        data = data.reshape(data.shape[0], data.shape[1], -1)
    
    
    
    return data

    

def data_transform(moudle_type, data, n_feature, n_his, n_pred, time_len, time_start, device):
    # produce data slices for x_near_data, x_future_data, and y_data
    
    
    
    day = 288
    
    
    
    
    n_vertex = data.shape[1] 
    len_record = len(data)
    num = len_record - n_his - n_pred# 
    
    
        
        
    if moudle_type == 'full':
        x = np.zeros([num, n_feature + n_feature, n_his, n_vertex])
    else:
        x = np.zeros([num, n_feature, n_his, n_vertex])
    y = np.zeros([num, n_pred, n_vertex])
        
    for i in range(num):
        head = i
        
        tail = i + n_his
        
        if moudle_type == 'full':
            # x_near_data:
            x[i, :n_feature, :, :] = data[head: tail].transpose(2, 0, 1)
            # x_future_data
            for j in range(n_pred):
          
                index = (tail + j) % (day * 7)
                data_transfer = data[range(index, len_record, (day * 7)), :, :]
                K = len(range(index, len_record, (day * 7)))
              
                data_F_e = (np.sum(data_transfer, axis=0) - data[tail + j]) / (K-1)
                data_F_t = np.mean(data_transfer, axis=0)
                residual_sample = data_F_e - data_F_t
                correction_factor = np.random.normal(0, np.random.random())
                residual_correction = residual_sample * correction_factor
                data_F = data_F_t + residual_correction 
    
                x[i, n_feature:, j, :] = data_F.transpose(1, 0)
        elif moudle_type == 'no-future':
            x[i] = data[head: tail].transpose(2, 0, 1)
        elif moudle_type == 'no-near':
            for j in range(n_pred):
                index = (tail + j) % (day * 7)
                data_transfer = data[range(index, len_record, (day * 7)), :, :]
                K = len(range(index, len_record, (day * 7)))
             
                data_F_e = (np.sum(data_transfer, axis=0) - data[tail + j]) / (K-1)
                data_F_t = np.mean(data_transfer, axis=0)
                residual_sample = data_F_e - data_F_t
                correction_factor = np.random.normal(0, np.random.random())
                residual_correction = residual_sample * correction_factor
                data_F = data_F_t + residual_correction 

                x[i, :, j, :] = data_F.transpose(1, 0)
        # y_data:
        y[i] = data[tail: tail + n_pred, :, 0]
                
    
    return x, y