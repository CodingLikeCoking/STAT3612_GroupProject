import pandas as pd
import numpy as np

def extract_X(csv_file):
    X_df = pd.read_csv(csv_file, index_col=[0], header=[0, 1, 2])
    X_arr = np.array(X_df)
    X_shape = X_arr.shape
    X_arr = np.reshape(X_arr, (X_shape[0], -1, 3, 24))
    X_arr = X_arr[:, :, 1, :]
    return X_arr

def extract_X_clean(csv_file):
    X_df = pd.read_csv(csv_file, index_col=[0], header=[0, 1, 2])
    X_arr = np.array(X_df)
    X_shape = X_arr.shape
    X_arr = np.reshape(X_arr, (X_shape[0], -1, 3, 24))
    def check_mask(sample):
        mask = sample[:, 0, :] # (104, 24)
        mask = np.sum(mask, axis=-1) # (104)
        mask = np.array(list(map(lambda item: max(0, 1-item), mask)))
        all_fake = np.average(mask, axis=-1)
        return all_fake < 0.7
    X_arr = np.array(list(filter(check_mask, X_arr)))
    X_arr = X_arr[:, :, 1, :]
    return X_arr

def extract_y(csv_file):
    y_df = pd.read_csv(csv_file, index_col=[0], header=[0])
    y_arr = np.array(y_df)[:, 0]
    return y_arr

def time_reduction(X_arr, factor):
    n, a, t = X_arr.shape
    reduced_arr = np.reshape(X_arr, (n, a, -1, factor))
    reduced_arr = np.average(reduced_arr, axis = -1)
    return reduced_arr

def attribute_reduction(X_arr, n_components):
    from sklearn.decomposition import PCA
    n, a, t = X_arr.shape
    processed_arr = np.transpose(X_arr, (0, 2, 1))
    processed_arr = np.reshape(processed_arr, (-1, a))
    pca = PCA(n_components = n_components)
    new_arr = pca.fit_transform(processed_arr)
    new_arr = np.reshape(new_arr, (n, t, n_components))
    new_arr = np.transpose(new_arr, (0, 2, 1))
    new_arr = np.reshape(new_arr, (n, n_components, t))
    return new_arr