"""
Module for loading UCI datasets (based on CIFAR-10/100 loading module by Serhiy Mytrovtsiy).

Author: Nadav Cohen
Project: https://github.com/cohennadav/overparam
"""

import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys


def get_dataset(ds_options={'task': 1}):
    """Load selected UCI dataset.

    Args:
        ds_name: String specifying name of dataset to load.  Supported choices:
            * 'drift' - "Gas Sensor Array Drift Dataset at Different Concentrations"
        ds_options: Dictionary holding key(string)-value pairs configuring different options for the dataset.  Options
            to configure depend on selected dataset:
            * For 'drift':
            ** 'task' - integer between 1 and 6 specifying which gas to use for labels.
    Returns:
        Tuple (examples, labels), where:
            examples: Matrix with rows holding examples.
            labels: Matrix with rows holding corresponding labels.
    """
    ds_dir = '../../data/raw/drift/'
    ds_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00270/driftdataset.zip'
    maybe_download_and_extract(ds_dir, ds_url)
    d = 128
    examples = np.empty(shape=(0, d))
    labels = np.empty(shape=(0, 1))
    for file_name in os.listdir(ds_dir):
        if file_name.endswith('.dat'):
            for line in open(os.path.join(ds_dir, file_name), 'r'):
                line_words = line.rstrip().split(' ')
                line_task = int(line_words[0].split(';')[0])
                if ds_options['task'] == line_task:
                    labels = np.append(labels, np.empty(shape=(1, 1)), axis=0)
                    examples = np.append(examples, np.empty(shape=(1, d)), axis=0)
                    labels[-1, 0] = float(line_words[0].split(';')[1])
                    for i in range(d):
                        assert int(line_words[i + 1].split(':')[0]) == i + 1
                        examples[-1, i] = float(line_words[i + 1].split(':')[1])
    return examples, labels


def maybe_download_and_extract(ds_dir, ds_url):
    """Download and extract dataset if not present.

    Args:
        ds_dir: String specifying folder for dataset download and extraction.
        ds_url: String specifying URL for dataset download.
    """
    if not os.path.exists(ds_dir):
        os.makedirs(ds_dir)
        file_name = ds_url.split('/')[-1]
        file_path = os.path.join(ds_dir, file_name)
        file_path, _ = urlretrieve(url=ds_url, filename=file_path, reporthook=_print_download_progress)
        print()
        print('Dataset download finished. Extracting files.')
        if file_path.endswith('.zip'):
            zipfile.ZipFile(file=file_path, mode='r').extractall(ds_dir)
        elif file_path.endswith(('.tar.gz', '.tgz')):
            tarfile.open(name=file_path, mode='r:gz').extractall(ds_dir)
        print('Done.')
        os.remove(file_path)


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = '\r- Download progress: {0:.1%}'.format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

if __name__ == "__main__":
    X_raw, y_raw = get_dataset(ds_options={'task': 1})
    np.save("../../data/raw/examples.npy", X_raw)
    np.save("../../data/raw/labels.npy", y_raw)
