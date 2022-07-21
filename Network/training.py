"""
Training for ZipperNet
"""
import sys

import numpy as np
import torch
import torch.nn as nn


def train_zipper(zipper, train_dataloader, train_dataset, test_dataset,
                 validation_size=None, monitor=False, outfile_prefix=""):

    zipper.train()

    number_of_training_epochs = 40
    if validation_size is None:
        validation_size = len(test_dataset)
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(zipper.parameters(), lr=learning_rate)

    losses, train_acc, validation_acc = [], [], []

    # Track best validation acc
    best_val_acc = 0.0

    for epoch in range(number_of_training_epochs):
        sys.stdout.write("\rEpoch {0}\r".format(epoch + 1))
        sys.stdout.flush()

        for i_batch, sample_batched in enumerate(train_dataloader):

            # Clear out all existing gradients on the loss surface to
            # reevaluate for this step
            optimizer.zero_grad()

            # Get the CNN's current prediction of the training data
            output = zipper(sample_batched['lightcurve'],
                            sample_batched['image'])

            # Calculate the loss by comparing the prediction to the truth
            loss = loss_function(output, sample_batched['label'])

            # Evaluate all gradients along the loss surface using back
            # propagation
            loss.backward()

            # Based on the gradients, take the optimal step in the weight space
            optimizer.step()

            # Performance monitoring if desired
            if monitor:
                if i_batch % 500 == 0:
                    train_output = zipper(
                        train_dataset[0:validation_size]['lightcurve'],
                        train_dataset[0:validation_size]['image'])
                    validation_output = zipper(
                        test_dataset[0:validation_size]['lightcurve'],
                        test_dataset[0:validation_size]['image'])

                    train_predictions = torch.max(
                        train_output, 1)[1].data.numpy()
                    validation_predictions = torch.max(
                        validation_output, 1)[1].data.numpy()

                    train_true = train_dataset[0:validation_size]['label']
                    train_true = train_true.numpy()
                    is_right = train_predictions == train_true
                    train_accuracy = np.sum(is_right) / validation_size

                    test_true = test_dataset[0:validation_size]['label']
                    test_true = test_true.numpy()
                    is_right = validation_predictions == test_true
                    validation_accuracy = np.sum(is_right) / validation_size

                    print("Epoch: {0} Batch: {1}  | Training Accuracy: {2:.3f}"
                          " -- Validation Accuracy: {3:.3f} -- Loss: {4:.3f}"
                          "".format(epoch + 1, i_batch + 1, train_accuracy,
                                    validation_accuracy, loss.data.numpy()))

                    losses.append(loss.data.numpy())
                    train_acc.append(train_accuracy)
                    validation_acc.append(validation_accuracy)

                    # Save best network
                    if validation_accuracy > best_val_acc:
                        torch.save(zipper.state_dict(),
                                   f"{outfile_prefix}_network.pt")
                        best_val_acc = validation_accuracy
                        best_state = zipper.state_dict()

    # Return the network with the best state
    zipper.load_state_dict(best_state)

    setattr(zipper, 'losses', losses)
    setattr(zipper, 'train_acc', train_acc)
    setattr(zipper, 'validation_acc', validation_acc)

    return zipper


def train_single(network, train_dataloader, train_dataset, test_dataset,
                 datatype, validation_size=None, monitor=False,
                 outfile_prefix=""):

    error_message = "datatype argument must be 'image' or 'lightcurve'"
    assert datatype in ['image', 'lightcurve'], error_message

    network.train()

    number_of_training_epochs = 20
    if validation_size is None:
        validation_size = len(test_dataset)
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    losses, train_acc, validation_acc = [], [], []

    # Track best validation acc
    best_val_acc = 0.0

    for epoch in range(number_of_training_epochs):
        sys.stdout.write("\rEpoch {0}\r".format(epoch + 1))
        sys.stdout.flush()

        for i_batch, sample_batched in enumerate(train_dataloader):

            # Clear out all existing gradients on the loss surface to
            # reevaluate for this step
            optimizer.zero_grad()

            # Get the CNN's current prediction of the training data
            output = network(sample_batched[datatype])

            # Calculate the loss by comparing the prediction to the truth
            loss = loss_function(output, sample_batched['label'])

            # Evaluate all gradients along the loss surface using back
            # propagation
            loss.backward()

            # Based on the gradients, take the optimal step in the weight space
            optimizer.step()

            # Performance monitoring if desired
            if monitor:
                if i_batch % 500 == 0:
                    train_output = network(
                        train_dataset[0:validation_size][datatype])
                    validation_output = network(
                        test_dataset[0:validation_size][datatype])

                    train_predictions = torch.max(
                        train_output, 1)[1].data.numpy()
                    validation_predictions = torch.max(
                        validation_output, 1)[1].data.numpy()

                    train_true = train_dataset[0:validation_size]['label']
                    train_true = train_true.numpy()
                    is_right = train_predictions == train_true
                    train_accuracy = np.sum(is_right) / validation_size

                    test_true = test_dataset[0:validation_size]['label']
                    test_true = test_true.numpy()
                    is_right = validation_predictions == test_true
                    validation_accuracy = np.sum(is_right) / validation_size

                    print("Epoch: {0} Batch: {1}  | Training Accuracy: {2:.3f}"
                          " -- Validation Accuracy: {3:.3f} -- Loss: {4:.3f}"
                          "".format(epoch + 1, i_batch + 1, train_accuracy,
                                    validation_accuracy, loss.data.numpy()))

                    losses.append(loss.data.numpy())
                    train_acc.append(train_accuracy)
                    validation_acc.append(validation_accuracy)

                    # Save best network
                    if validation_accuracy > best_val_acc:
                        torch.save(network.state_dict(),
                                   f"{outfile_prefix}_network.pt")
                        best_val_acc = validation_accuracy
                        best_state = network.state_dict()

    # Return the network with the best state
    network.load_state_dict(best_state)

    setattr(network, 'losses', losses)
    setattr(network, 'train_acc', train_acc)
    setattr(network, 'validation_acc', validation_acc)

    return network
