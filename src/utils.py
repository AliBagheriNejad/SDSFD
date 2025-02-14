# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


global device


# Function to train classifier model
def train_classifier(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=100, early_stopping='val_loss'):
    """
    Trains a classifier model.

    Args:
        model (nn.Module): The classifier model to train.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        epochs (int): Number of training epochs.
        early_stopping (str): Metric for early stopping ('val_loss', 'val_acc', 'train_loss', 'train_acc').

    Returns:
        dict: Dictionary containing training and validation losses and accuracies.
    """
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}')

        for i, (batch_data, batch_labels) in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch_data)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

            progress_bar.set_postfix(train_loss=train_loss / (i + 1), train_acc=100 * correct_train / total_train)

        train_losses.append(train_loss / len(train_dataloader))
        train_accs.append(100 * correct_train / total_train)

        # Validation
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_dataloader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid += batch_labels.size(0)
                correct_valid += (predicted == batch_labels).sum().item()

        valid_losses.append(valid_loss / len(val_dataloader))
        valid_accs.append(100 * correct_valid / total_valid)

        print(f'Validation Accuracy: {valid_accs[-1]:.1f}, Validation Loss: {valid_losses[-1]:.4f}')

        # Early stopping
        if early_stopping == 'val_acc':
            stop = model.early_stopping(valid_accs[-1], epoch)
        elif early_stopping == 'val_loss':
            stop = model.early_stopping(-valid_losses[-1], epoch)
        elif early_stopping == 'train_acc':
            stop = model.early_stopping(train_accs[-1], epoch)
        elif early_stopping == 'train_loss':
            stop = model.early_stopping(-train_losses[-1], epoch)

        if stop:
            break

    return {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': valid_losses, 'val_acc': valid_accs}



# Function to reassign labels after removing unwanted ones
def fix_labels(labels, to_remove):
    """
    Removes specified labels and reassigns remaining labels to be consecutive integers starting from 0.

    Args:
        labels (list): List of labels.
        to_remove (list): Labels to remove.

    Returns:
        torch.Tensor: Tensor of reassigned labels.
    """
    new_labels = []
    label_map = {}
    current_label = 0

    for label in labels:
        if label not in to_remove:
            if label not in label_map:
                label_map[label] = current_label
                current_label += 1
            new_labels.append(label_map[label])

    return torch.tensor(new_labels, dtype=torch.long).to(device)



# Function to enable or disable gradient computation for a model
def set_grad(model, requires_grad=True):
    """
    Sets the requires_grad attribute of all model parameters.

    Args:
        model (nn.Module): The model whose parameters are to be updated.
        requires_grad (bool): Whether gradients should be computed.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad



# Function to extract regressor weights for each input
def sensor_weights(model, input_tensor, device):
    """
    Extracts weights corresponding to each input using the regressor component of the model.

    Args:
        model (nn.Module): The model.
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        numpy.ndarray: Extracted weights.
    """
    model.eval()
    with torch.no_grad():
        x = input_tensor.to(device)
        if model.in_ch == 1:
            x_input = x.view(x.shape[0], 1, x.shape[1])
        else:
            x_input = x.view(x.shape[0], x.shape[2], x.shape[1])

        x = model.feature_extractor_1(x_input)
        x = model.regressor(x).unsqueeze(-1)
        return x.cpu().detach().numpy()



# Function to plot weight distribution
def plot_weight_distribution(df, label):
    """
    Plots the distribution of weights for each class using Seaborn.

    Args:
        df (DataFrame): DataFrame containing weight data.
        label (str): Column name of the weights to plot.
    """
    plt.figure(figsize=(20, 10))
    sns.kdeplot(
        data=df,
        x=label,
        hue='label',
        fill=True,
        palette='husl',
        linewidth=2.5
    )
    plt.xlabel('Regressor Output', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.title(f'Weight Distribution for {label}', fontsize=30)
    plt.legend(fontsize=15)
    plt.show()



# Function to extract second feature extractor outputs
def extract_second_features(model, input_tensor,device):
    """
    Extracts features from the second feature extractor in the model.

    Args:
        model (nn.Module): The model.
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Extracted features.
    """
    model.eval()
    with torch.no_grad():
        x = input_tensor.to(device)
        if model.in_ch == 1:
            x_input = x.view(x.shape[0], 1, x.shape[1])
        else:
            x_input = x.view(x.shape[0], x.shape[2], x.shape[1])

        x = model.feature_extractor_1(x_input)
        x = model.regressor(x).unsqueeze(-1)
        x = x_input * x
        return model.feature_extractor_2(x)




# Function to extract first feature extractor outputs
def extract_first_features(model, input_tensor):
    """
    Extracts features from the first feature extractor in the model.

    Args:
        model (nn.Module): The model.
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Extracted features.
    """
    model.eval()
    with torch.no_grad():
        x = input_tensor.to(device)
        if model.in_ch == 1:
            x_input = x.view(x.shape[0], 1, x.shape[1])
        else:
            x_input = x.view(x.shape[0], x.shape[2], x.shape[1])
        return model.feature_extractor_1(x_input)

# Function to extract first feature extractor outputs
def extract_features(model, input_tensor,device):
    """
    Extracts features from the first feature extractor in the model.

    Args:
        model (nn.Module): The model.
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Extracted features.
    """
    model.eval()
    with torch.no_grad():
        x = input_tensor.to(device)
        if model.in_ch == 1:
            x_input = x.view(x.shape[0], 1, x.shape[1])
        else:
            x_input = x.view(x.shape[0], x.shape[2], x.shape[1])
        return model.feature_extractor(x_input)


# Plot reduced features
def plot_reduced_features(n_classes, reduced_features, labels, title, class_dic):
    """
    Visualizes reduced features using a scatter plot.

    Args:
        n_classes (int): Number of distinct classes.
        reduced_features (numpy.ndarray): Array of reduced feature dimensions.
        labels (torch.Tensor): True labels for the data points.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(20, 20))
    for i in range(n_classes):
        indices = labels.cpu().numpy() == i
        plt.scatter(
            reduced_features[indices, 0], 
            reduced_features[indices, 1],
            s=200, 
            label=f'{class_dic.get(i, f"Class {i}")}', 
            alpha=0.6
        )

    plt.xlabel('t-SNE Dimension 1', fontsize=25)
    plt.ylabel('t-SNE Dimension 2', fontsize=25)
    plt.title(title, fontsize=40)
    plt.legend(fontsize=20, loc='best')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



# Plot training and validation metrics
def plot_training_metrics(metrics):
    """
    Plots training and validation accuracy and loss over epochs.

    Args:
        metrics (dict): Dictionary containing 'train_acc', 'val_acc', 'train_loss', and 'val_loss'.
    """
    train_accs = metrics.get('train_acc', [])
    val_accs = metrics.get('val_acc', [])
    train_losses = metrics.get('train_loss', [])
    val_losses = metrics.get('val_loss', [])

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(20, 8))
    plt.plot(train_accs, 'g-o', linewidth=3, markersize=8, label='Train Accuracy')
    plt.plot(val_accs, 'r-o', linewidth=3, markersize=8, label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.title('Training and Validation Accuracy', fontsize=25)
    plt.legend(fontsize=15, loc='best')
    plt.xticks(fontsize=15)
    plt.yticks(np.arange(0, 101, 10), fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

   
    # Plot Training and Validation Loss
    plt.figure(figsize=(20, 8))
    plt.plot(train_losses, 'g-o', linewidth=3, markersize=8, label='Train Loss')
    plt.plot(val_losses, 'r-o', linewidth=3, markersize=8, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=25)
    plt.legend(fontsize=15, loc='best')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def data_forward(model, val_loader):

    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            # batch_data = batch_data.unsqueeze(1)
            _, outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)

            y_pred.append(
                predicted.cpu().detach().numpy()
            )
            y_true.append(
                batch_labels.cpu().detach().numpy()
            )

    return np.hstack(y_pred), np.hstack(y_true)
