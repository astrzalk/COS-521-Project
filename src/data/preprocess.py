#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA

def main():
    # Load Data
    raw_data_path = '../../data/raw/'
    X_raw = np.load(raw_data_path + 'examples.npy')
    y_raw = np.load(raw_data_path + 'labels.npy')

    # Whiten Data
    X_white = PCA(whiten=True).fit_transform(X_raw)

    # Rescale labels to make Frobenius norm equal to 1 (or very close)
    cross_cov = (1 / y_raw.shape[0]) * (y_raw.T.dot(X_raw))
    fro_norm = np.linalg.norm(cross_cov, ord='fro')
    y_scaled = y_raw / fro_norm # This is quite close to \fro_norm{cross_cov} \approx 1

    # Save Processed data
    proc_data_path = '../../data/processed/'
    np.save(proc_data_path + 'X.npy', X_white)
    np.save(proc_data_path + 'y.npy', y_scaled)

if __name__ == "__main__":
    main()

