"""
Functions for using a network to predict on data
"""

import torch
import numpy as np


def predict(network, dataset):
    network.eval()

    try:
        predictions = torch.max(network(dataset[:]['lightcurve'],
                                        dataset[:]['image']), 1)[1].data.numpy()
    except TypeError:  # Instead of ZIPPER, we are using CNN or RNN
        try:  # RNN
            predictions = torch.max(network(dataset[:]['lightcurve']),
                                    1)[1].data.numpy()
        except TypeError:  # CNN
            predictions = torch.max(network(dataset[:]['image']),
                                    1)[1].data.numpy()
    labels = dataset[:]['label'].data.numpy()

    print("Accuracy:", np.sum(predictions == labels) / len(labels))

    return predictions, labels
