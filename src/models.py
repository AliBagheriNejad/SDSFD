# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


    
class FeatureExtractor(nn.Module):
    """
    A feature extraction module for 1D input data.

    Args:
        drop (float): Dropout rate for regularization.
        input_channels (int): Number of input channels.
    """
    def __init__(self, drop=0.1, input_channels=1, cons:int=2):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16*cons, kernel_size=128)
        self.bn1 = nn.BatchNorm1d(num_features=16*cons)
        self.dropout1 = nn.Dropout(p=drop)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=16*cons, out_channels=32*cons, kernel_size=64)
        self.bn2 = nn.BatchNorm1d(num_features=32*cons)
        self.dropout2 = nn.Dropout(p=drop)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(in_channels=32*cons, out_channels=64*cons, kernel_size=16)
        self.bn3 = nn.BatchNorm1d(num_features=64*cons)
        self.dropout3 = nn.Dropout(p=drop)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=64*cons, out_channels=128*cons, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(num_features=128*cons)
        self.dropout4 = nn.Dropout(p=drop)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=128*cons, out_channels=256*cons, kernel_size=2)
        self.bn5 = nn.BatchNorm1d(num_features=256*cons)
        self.dropout5 = nn.Dropout(p=drop)

    def forward(self, x):
        """
        Forward pass for the feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, seq_length).

        Returns:
            torch.Tensor: Flattened feature tensor.
        """
        x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.dropout3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout4(F.relu(self.bn4(self.conv4(x)))))
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, start_dim=1)
        return x
    



class Classifier(nn.Module):
    """
    A classification module with three fully connected layers.

    Args:
        num_classes (int): Number of output classes.
        drop (float): Dropout rate for regularization.
    """
    def __init__(self, num_classes: int, drop: float = 0.2, cons:int=2):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(1024*cons, 128)
        self.dropout1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(drop)

        self.fc3 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(drop)

        self.fcc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, num_classes).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fcc(x), dim=1)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute raw logits (before softmax).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Regressor(nn.Module):
    """
    A regression module with three fully connected layers.

    Args:
        num_channels (int): Number of output channels for regression.
        drop (float): Dropout rate for regularization.
    """
    def __init__(self, num_channels: int, drop: float = 0.2, cons:int=2):
        super(Regressor, self).__init__()
        
        self.reg1 = nn.Linear(1024*cons, 128)
        self.reg2 = nn.Linear(128,64)
        self.reg3 = nn.Linear(64,num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: Regression outputs of shape (batch_size, num_channels).
        """
        x = F.relu(self.reg1(x))
        x = F.relu(self.reg2(x))
        x = F.softmax(self.reg3(x), dim=1)
        return x
    


class Network(nn.Module):
    """
    A neural network that includes feature extraction, regression, and classification modules.
    Supports early stopping during training.

    Args:
        num_classes (int): Number of output classes for classification.
        in_channels (int): Number of input channels for the feature extractor.
    """
    def __init__(self, num_classes, in_channels=1, cons:int=2):
        super(Network, self).__init__()
        self.feature_extractor_1 = FeatureExtractor(input_channels=in_channels, cons=cons)
        self.feature_extractor_2 = FeatureExtractor(input_channels=in_channels, cons=cons)
        self.regressor = Regressor(2, cons=cons)
        self.classifier = Classifier(num_classes, cons=cons)

        # Early stopping parameters
        self.best_acc = 0
        self.save_path = '/content/drive/MyDrive/ADDA/pruning_01.pth'
        self.patience = 10
        self.e_ratio = 100
        self.in_ch = in_channels

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, num_classes).
        """
        if self.in_ch == 1:
            x_input = x.view(x.shape[0], 1, x.shape[1])
        else:
            x_input = x.view(x.shape[0], x.shape[2], x.shape[1])

        x = self.feature_extractor_1(x_input)
        x = self.regressor(x).unsqueeze(-1)
        x = x_input * x
        x = self.feature_extractor_2(x)
        x = self.classifier(x)
        return x

    def early_stopping(self, metric, epoch):
        """
        Implements early stopping based on validation metric.

        Args:
            metric (float): Current metric value to evaluate.
            epoch (int): Current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if (metric > self.best_acc) and (np.abs(metric - self.best_acc) > np.abs(self.best_acc) / self.e_ratio):
            self.best_acc = metric
            self.best_epoch = epoch
            self.current_patience = 0

            # Save the model's weights
            torch.save(self.state_dict(), self.save_path)
            print("<<<<<<<  Model saved! >>>>>>>")
            return False
        else:
            self.current_patience += 1
            if self.current_patience >= self.patience:
                print("Early stopping triggered!")
                return True
            return False


class NetworkNoReg(nn.Module):
    """
    A neural network with feature extraction and classification modules.
    Excludes the regression component.

    Args:
        num_classes (int): Number of output classes for classification.
        in_channels (int): Number of input channels for the feature extractor.
    """
    def __init__(self, num_classes, in_channels=1):
        super(NetworkNoReg, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channels=in_channels)
        self.classifier = Classifier(num_classes)

        # Early stopping parameters
        self.best_acc = 0
        self.save_path = '/content/drive/MyDrive/ADDA/model_weights.pth'
        self.patience = 10
        self.e_ratio = 100
        self.in_ch = in_channels

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, num_classes).
        """
        if self.in_ch == 1:
            x = x.view(x.shape[0], 1, x.shape[1])
        else:
            x = x.view(x.shape[0], x.shape[2], x.shape[1])

        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def early_stopping(self, metric, epoch):
        """
        Implements early stopping based on validation metric.

        Args:
            metric (float): Current metric value to evaluate.
            epoch (int): Current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if (metric > self.best_acc) and (np.abs(metric - self.best_acc) > np.abs(self.best_acc) / self.e_ratio):
            self.best_acc = metric
            self.best_epoch = epoch
            self.current_patience = 0

            # Save the model's weights
            torch.save(self.state_dict(), self.save_path)
            print("<<<<<<<  Model saved! >>>>>>>")
            return False
        else:
            self.current_patience += 1
            if self.current_patience >= self.patience:
                print("Early stopping triggered!")
                return True
            return False
        



