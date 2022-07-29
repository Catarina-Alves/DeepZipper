"""
Training for ZipperNet
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def plot_grad_flow(named_parameters):
    """Plot the gradient flow of the named parameters.

    This function is useful to investigate vanishing/exploding gradients.

    Code adapted from
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7 .
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            grads = p.grad
            ave_grads.append(np.log10(grads[grads != 0].abs().mean().item()))
    plt.plot(ave_grads, alpha=0.005, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Log average abs gradient")
    plt.title("Gradient flow")
    plt.grid(True)


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


def train_network_regression(network, train_dataloader, train_dataset,
                             test_dataset, validation_size=None, monitor=False,
                             outfile_prefix="", **kwargs):
    """Trains networks to estimate parameters.

    This is a regression task. The code was modified from `train_zipper` for a
    test of concept. Thus it should be further improved for actual use.

    `**kwargs` were introduced for parameters that could not be modified in
    `train_zipper`. In Catarina Alves opinion (28/July/2022) more parameters
    should be included as `**kwargs`, such as `outfile_prefix`.
    """
    # Whether to show gradient flow plots. These plots help to investigate
    # vanishing/exploding gradients
    show_grad_flow = kwargs.pop('show_grad_flow', True)

    # Start the training
    network.train()

    # After 200 epochs the loss reaches a plateau;
    # By default we use 20 to obtain results fast
    number_training_epochs = kwargs.pop('number_training_epochs', 20)
    if validation_size is None:
        validation_size = len(test_dataset)
    loss_function = kwargs.pop('loss_function', nn.MSELoss())
    learning_rate = kwargs.pop('number_training_epochs', 0.001)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    losses, train_acc, validation_acc = [], [], []

    # Track best validation acc
    best_val_acc = 0.0

    for epoch in range(number_training_epochs):
        sys.stdout.write("\rEpoch {0}\r".format(epoch + 1))
        sys.stdout.flush()

        loss_epoch = []
        train_acc_epoch = []
        validation_acc_epoch = []

        for i_batch, sample_batched in enumerate(train_dataloader):
            # Torch prefers to work works with torch.float32.
            # The rule of a thumb is to do explicit conversion of tensors to
            # torch.float32 before passing to a neural net.
            # You can just do type conversion in the beginning.
            for key in sample_batched:
                sample_batched[key] = sample_batched[key].type(torch.float32)

            # Clear out all existing gradients on the loss surface to
            # reevaluate for this step
            optimizer.zero_grad()

            # Get the current prediction of the training data
            # TODO: edit to also work with CNN and RNN regression. Catarina
            # Alves suggests modifying `data_utils.CombinedDataset` and use
            # dictionaries to make this change
            output = network(sample_batched['lightcurve'],
                             sample_batched['image'])

            # Calculate the loss by comparing the prediction to the truth
            loss = loss_function(output, sample_batched['label'])

            # Evaluate all gradients along the loss surface using back
            # propagation
#             loss.mean().backward()
            loss.backward()

            if show_grad_flow:
                plot_grad_flow(network.named_parameters())

            # Based on the gradients, take the optimal step in the weight space
            optimizer.step()

            # Save average performance per epoch
            train_output = network(
                train_dataset[0:validation_size]['lightcurve'],
                train_dataset[0:validation_size]['image'])
            train_true = train_dataset[0:validation_size]['label']
            train_loss = loss_function(train_output, train_true)

            validation_output = network(
                test_dataset[0:validation_size]['lightcurve'],
                test_dataset[0:validation_size]['image'])
            validation_true = test_dataset[0:validation_size]['label']
            validation_loss = loss_function(validation_output, validation_true)
            validation_accuracy = validation_loss.data.numpy()

            loss_epoch.append(loss.data.numpy())
            train_acc_epoch.append(train_loss.data.numpy())
            validation_acc_epoch.append(validation_accuracy)

            # Save best network
            if validation_accuracy > best_val_acc:
                torch.save(network.state_dict(),
                           f"{outfile_prefix}_regression_network.pt")
                best_val_acc = validation_accuracy
                best_state = network.state_dict()

        # Save average performance per epoch
        losses.append(np.mean(loss_epoch))
        train_acc.append(np.mean(train_acc_epoch))
        train_acc.append(np.mean(validation_acc_epoch))

        # Performance monitoring if desired
        if monitor:
            print("Epoch: {0}  | Training Accuracy: {2:.3f}"
                  " -- Validation Accuracy: {3:.3f} -- Loss: {4:.3f}"
                  "".format(epoch + 1, np.mean(train_acc_epoch),
                            np.mean(validation_acc_epoch),
                            np.mean(loss_epoch)))

    # Return the network with the best state
    network.load_state_dict(best_state)

    setattr(network, 'losses', losses)
    setattr(network, 'train_acc', train_acc)
    setattr(network, 'validation_acc', validation_acc)

    return network
