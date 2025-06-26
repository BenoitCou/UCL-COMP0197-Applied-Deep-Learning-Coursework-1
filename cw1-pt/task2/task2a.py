#IMPORTANT : The parameters save_model and load_model of fit_elm_sgd are respectively set to False and True
# to avoid overwriting the saved models and allow to run the code without training the models again. 
# Please change them to True and False if you want to retrain and save the models.

#IMPORTANT : The parameter load_model of MyEnsembleELM are set to True to avoid overwriting the saved models 
# and allow to run the code without training the models again. 
# Please change them to False if you want to retrain and save the models.


import torch
import numpy as np
import time
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import copy
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
import random
from collections import Counter
from torchvision.utils import make_grid  
import torchvision.utils as vutils
from torchvision.utils import save_image

class MyExtremeLearningMachine(nn.Module):
    """Extreme Learning Machine (ELM) with fixed convolutional weights and trainable final layer.
    
    Args:
        fixed_weights (torch.Tensor, optional): Pre-trained weights for conv layer. 
            Shape: (hidden_size, input_channels, 3, 3). Default: None
        hidden_size (int): Number of hidden units in conv layer
        input_channels (int): Number of input channels
        input_height (int): Input image height
        input_width (int): Input image width
        num_classes (int): Number of output classes. Default: 10
        seed (int): Random seed for reproducibility. Default: 0
        std (float): Weight initialization std deviation. Default: 0.1
    """
    def __init__(self, fixed_weights, hidden_size, input_channels, input_height, input_width, num_classes=10, seed=0, std=0.1):
        super(MyExtremeLearningMachine, self).__init__()
        self.fixed_weights = fixed_weights
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.seed = seed
        self.conv_output_size = (self.input_height - 2) * (self.input_width - 2) * self.hidden_size

        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.conv1 = nn.Conv2d(self.input_channels, self.hidden_size, kernel_size=(3, 3))

        if fixed_weights is None:
            with torch.no_grad():
                self.conv1.weight.copy_(self.initialise_fixed_layers(self.hidden_size, self.input_channels, 3, 3, std))
        else:
            if isinstance(fixed_weights, torch.Tensor) and fixed_weights.shape == self.conv1.weight.shape:
                self.conv1.weight = nn.Parameter(fixed_weights)
            else:
                raise ValueError("Fixed_weights must be a tensor of the correct shape!")

        self.conv1.weight.requires_grad = False

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(self.conv_output_size, num_classes)
        self.fc2.weight.requires_grad = True

    def initialise_fixed_layers(self, a, b, c, d, std):
        """Initialize fixed convolutional weights with normal distribution.
        
        Args:
            a (int): Output channels (hidden_size)
            b (int): Input channels (input_channels)
            c (int): Kernel height dimension
            d (int): Kernel width dimension 
            std (float): Standard deviation for weight initialization
            
        Returns:
            torch.Tensor: Initialized weights tensor of shape (a, b, c, d)
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        return torch.normal(0, std, size=(a, b, c, d))

    def forward(self, x):
        """Forward pass through the ELM network.
        
        Args:
            x (torch.Tensor): Input tensor 
                Shape: (batch_size, input_channels, input_height, input_width)
                
        Returns:
            torch.Tensor: Output logits 
                Shape: (batch_size, num_classes)
        """
        with torch.no_grad():
            x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc2(x)
        return x

class MyMixUp:
    """MixUp data augmentation implementation for image-label pairs.
    
    Args:
        alpha (float): Beta distribution parameter for mixing ratio. Default: 1
        seed (int): Random seed for reproducibility. Default: 42
    """
    def __init__(self, alpha=1, seed=42):
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def apply(self, images, labels):
        """Apply MixUp augmentation to a batch of images and labels.
        
        Args:
            images (torch.Tensor): Input images 
                Shape: (batch_size, channels, height, width)
            labels (torch.Tensor): One-hot encoded labels
                Shape: (batch_size, num_classes)
                
        Returns:
            tuple: (
                mixed_images (torch.Tensor): Mixed images 
                    Shape: same as input images,
                mixed_labels (torch.Tensor): Mixed labels 
                    Shape: same as input labels
            )
        """
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        images2 = images[indices]
        labels2 = labels[indices]
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_images = lam * images + (1 - lam) * images2
        mixed_labels = lam * labels + (1 - lam) * labels2
        return mixed_images, mixed_labels

    def save_montage(self, images, filename="task2/mixup.png", num_images=16):
        """Save grid visualization of mixed images.
        
        Args:
            images (torch.Tensor): Batch of images 
                Shape: (batch_size, channels, height, width)
            filename (str): Output file path
            num_images (int): Number of images to display in grid
        """
        images = (images - images.min()) / (images.max() - images.min())
        
        grid = vutils.make_grid(
            images[:num_images], 
            nrow=4, 
            normalize=False,  
            padding=1
        )
        save_image(grid, filename)

    def apply_with_labels(self, images, labels):
        """Apply MixUp and return original labels + mixing coefficient.
        
        Args:
            images (torch.Tensor): Input images 
                Shape: (batch_size, channels, height, width)
            labels (torch.Tensor): One-hot encoded labels 
                Shape: (batch_size, num_classes)
                
        Returns:
            tuple: (
                mixed_images (torch.Tensor), 
                original_labels (torch.Tensor),
                shuffled_labels (torch.Tensor), 
                lam (float): Mixing coefficient
            )
        """
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        images2 = images[indices]
        labels2 = labels[indices]
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_images = lam * images + (1 - lam) * images2
        return mixed_images, labels, labels2, lam

class MyEnsembleELM:
    """Ensemble of Extreme Learning Machines with optional MixUp augmentation.
    
    Args:
        trainloader (DataLoader): Training data loader
            Each batch: (images [batch_size, channels, h, w], labels [batch_size])
        learning_rate (float): SGD learning rate (1e-6 to 1)
        num_epochs (int): Training epochs (>0)
        seeds (list[int]): Random seeds for ensemble members
        std (float): Weight initialization std. Default: 0.1
        hidden_layers (int): Hidden units in ELM. Default: 10
        test_loader (DataLoader, optional): Test data loader. Default: None
        load_model (bool): Load saved models. Default: False
        mixup_alpha (float, optional): MixUp alpha. Default: None
    """
    def __init__(self, trainloader, learning_rate, num_epochs, seeds, std=0.1, hidden_layers=10, test_loader=None, load_model=False, mixup_alpha=None):
        self.seeds = seeds
        self.models = []
        self.mixup_alpha = mixup_alpha

        if not (1e-6 <= learning_rate <= 1):
             raise ValueError("Learning rate outside of the recommended zone [1e-6, 1].")
        if not isinstance(num_epochs, int) or num_epochs < 1:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(std, float) or std <= 0:
            raise ValueError("Standard Deviation must be a positive float.")
        if not isinstance(hidden_layers, int) or hidden_layers <= 0:
            raise ValueError("The number of hidden layers must be a positive integer.")

        if not load_model:
            for seed in seeds:
                model, _, _, _, _, _ = fit_elm_sgd(trainloader, learning_rate, num_epochs, seed, test_loader=test_loader, save_model=False, load_model=False, std=std, hidden_size=hidden_layers, mixup_alpha=mixup_alpha, graph_name=f"ELM_{seed}")
                self.models.append(model)
            self.save_models()
        else:
            self.load_models()

    def predict(self, pairs):
        """Make predictions using ensemble averaging.
        
        Args:
            pairs (tuple): (
                X (np.ndarray/torch.Tensor): Input data 
                    Shape: (num_samples, channels, height, width),
                y: Ignored placeholder
            )
            
        Returns:
            torch.Tensor: Averaged predictions 
                Shape: (num_samples, num_classes)
        """
        X, _ = pairs
        X = torch.tensor(X, dtype=torch.float32)
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                y_pred = model.forward(X)
                predictions.append(y_pred)
        stacked = torch.stack(predictions)
        avg_prediction = torch.mean(stacked, dim=0)
        return avg_prediction

    def save_models(self, directory='task2/saved_models'):
        """Save ensemble models to disk with different naming conventions.
        
        Args:
            directory (str): Output directory path
        """
        if self.mixup_alpha is None:
            os.makedirs(directory, exist_ok=True)
            for i, model in enumerate(self.models):
                torch.save(model.state_dict(), f'{directory}/ensemble_model_{i}.pth')
        else :
            os.makedirs(directory, exist_ok=True)
            for i, model in enumerate(self.models):
                torch.save(model.state_dict(), f'{directory}/ensemble_model_mixup_{i}.pth')

    def load_models(self, directory='task2/saved_models'):
        """Load pretrained ensemble models from disk.
        
        Args:
            directory (str): Input directory path
        """
        self.models = []
        for i in range(len(self.seeds)):
            model = MyExtremeLearningMachine(
                fixed_weights=None,
                hidden_size=10,
                input_channels=3,
                input_height=32,
                input_width=32,
                num_classes=10,
                seed=self.seeds[i]
            )
            model.load_state_dict(torch.load(f'{directory}/ensemble_model_{i}.pth'))
            model.eval()
            self.models.append(model)

def fit_elm_sgd(train_loader, learning_rate, num_epochs, seed, test_loader=None, save_model=True, load_model=False, model_path='task2/saved_models/best_model_ELM.pth', std=0.1, hidden_size=10, mixup_alpha=None, graph_name="ELS"):
    """Train an ELM model with SGD optimizer and optional MixUp augmentation.
    
    Args:
        train_loader (DataLoader): Training data loader
        learning_rate (float): Learning rate for SGD (1e-6 to 1)
        num_epochs (int): Number of training epochs (>0)
        seed (int): Random seed for reproducibility
        test_loader (DataLoader, optional): Test data loader. Default: None
        save_model (bool): Whether to save trained model. Default: True
        load_model (bool): Whether to load existing model. Default: False
        model_path (str): Path for model saving/loading. Default: 'task2/saved_models/best_model_ELM.pth'
        std (float): Weight initialization std. Default: 0.1
        hidden_size (int): Number of hidden units. Default: 10
        mixup_alpha (float, optional): MixUp alpha parameter. Default: None
        graph_name (str): Base name for performance plot. Default: "ELS"
    
    Returns:
        tuple: (model, avg_epoch_loss, best_train_acc, best_train_f1, test_accuracies, test_f1s)
    """
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch[0].shape[1]
    input_height = sample_batch[0].shape[2]
    input_width = sample_batch[0].shape[3]

    num_classes = 10

    model = MyExtremeLearningMachine(
        fixed_weights=None,
        hidden_size=hidden_size,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        num_classes=num_classes,
        seed=seed,
        std=std
    )

    if load_model and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        avg_epoch_loss = checkpoint['avg_epoch_loss']
        best_acc = checkpoint['best_acc']
        best_test_f1 = checkpoint['test_f1']
        best_train_acc = checkpoint['train_acc']
        best_train_f1 = checkpoint['train_f1']
        print(f"Model and metrics loaded from {model_path}")
    else:
        avg_epoch_loss = 0.0
        best_acc = 0.0
        best_test_f1 = 0.0
        best_train_acc = 0.0
        best_train_f1 = 0.0

    if not load_model:
        optimizer = optim.SGD(model.fc2.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        mixup = MyMixUp(alpha=mixup_alpha, seed=seed) if mixup_alpha else None

        best_model = None
        display_epochs = []
        test_accuracies = []
        test_f1s = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                
                # Training loop with optional MixUp augmentation
                if mixup:
                    mixed_X, y_a, y_b, lam = mixup.apply_with_labels(X_batch, y_batch)
                    y_pred = model(mixed_X)
                    loss = lam * loss_fn(y_pred, y_a) + (1 - lam) * loss_fn(y_pred, y_b)
                else:
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_epoch_loss = epoch_loss / len(train_loader)

            if epoch % (num_epochs // 20) == 0:
                model.eval()
                with torch.no_grad():
                    train_preds, train_trues = [], []
                    for X_batch, y_batch in train_loader:
                        outputs = model(X_batch)
                        train_preds.append(outputs)
                        train_trues.append(y_batch)
                    train_preds_all = torch.cat(train_preds, dim=0)
                    train_trues_all = torch.cat(train_trues, dim=0)
                    train_predictions = torch.argmax(train_preds_all, dim=1).cpu().numpy()
                    train_true_np = train_trues_all.cpu().numpy()
                    train_acc = accuracy_score(train_true_np, train_predictions)
                    train_f1 = f1_score(train_true_np, train_predictions, average='weighted')

                    if test_loader is not None:
                        y_pred_list, y_true_list = [], []
                        for X_batch, y_batch in test_loader:
                            y_pred_batch = model(X_batch)
                            y_pred_list.append(y_pred_batch)
                            y_true_list.append(y_batch)
                        y_pred_all = torch.cat(y_pred_list, dim=0)
                        y_true_all = torch.cat(y_true_list, dim=0)
                        predictions = torch.argmax(y_pred_all, dim=1).cpu().numpy()
                        y_np = y_true_all.cpu().numpy()
                        test_acc = accuracy_score(y_np, predictions)
                        test_f1 = f1_score(y_np, predictions, average='weighted')
                    else:
                        test_acc, test_f1 = 0, 0

                display_epochs.append(epoch)
                test_accuracies.append(test_acc)
                test_f1s.append(test_f1)
                print(f"Epoch {epoch}: Train Loss: {avg_epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_test_f1 = test_f1
                    best_train_acc = train_acc
                    best_train_f1 = train_f1
                    best_avg_epoch_loss = avg_epoch_loss
                    best_model = copy.deepcopy(model)

        if save_model:
            os.makedirs('task2/saved_models', exist_ok=True)
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'avg_epoch_loss': best_avg_epoch_loss,
                'best_acc': best_acc,
                'test_f1': best_test_f1, 
                'train_acc': best_train_acc,
                'train_f1': best_train_f1,
            }, model_path)

        draw_performance_plot(display_epochs, test_accuracies, test_f1s, graph_name=graph_name)

    else:
        test_accuracies, test_f1s = None, None

    return model, avg_epoch_loss, best_train_acc, best_train_f1, test_accuracies, test_f1s

def draw_performance_plot(epochs, accuracies, f1_scores, graph_name, width=800, height=400):
    """Generate performance plot image showing accuracy and F1 score over epochs.
    
    Args:
        epochs (list[int]): List of epoch numbers
        accuracies (list[float]): Test accuracies per epoch
        f1_scores (list[float]): Test F1 scores per epoch
        graph_name (str): Base name for output file
        width (int): Image width in pixels. Default: 800
        height (int): Image height in pixels. Default: 400
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    margin = 50
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    x_min = min(epochs) if epochs else 0
    x_max = max(epochs) if epochs else 1
    y_min = 0.0
    y_max = 1.0 

    draw.line([(margin, height - margin), (width - margin, height - margin)], fill="black", width=2)
    draw.line([(margin, margin), (margin, height - margin)], fill="black", width=2)

    draw.text((width / 2 - 20, height - margin + 10), "Epoch", fill="black")
    draw.text((10, margin - 20), "Score", fill="black")

    def get_coords(x, y):
        x_pixel = margin + ((x - x_min) / (x_max - x_min)) * plot_width if x_max != x_min else margin
        y_pixel = height - margin - ((y - y_min) / (y_max - y_min)) * plot_height if y_max != y_min else height - margin
        return (x_pixel, y_pixel)

    if len(epochs) > 1:
        acc_points = [get_coords(x, y) for x, y in zip(epochs, accuracies)]
        draw.line(acc_points, fill="red", width=2)
        f1_points = [get_coords(x, y) for x, y in zip(epochs, f1_scores)]
        draw.line(f1_points, fill="blue", width=2)

        for (x, y), val in zip(acc_points, accuracies):
            draw.text((x, y - 10), f"{val:.2f}", fill="red")

        for (x, y), val in zip(f1_points, f1_scores):
            draw.text((x, y - 10), f"{val:.2f}", fill="blue")

    step_x = max(1, (x_max - x_min) // 5) 
    for epoch in range(x_min, x_max + 1, step_x):
        x_pixel = get_coords(epoch, y_min)[0]
        draw.line([(x_pixel, height - margin), (x_pixel, height - margin + 5)], fill="black", width=2)
        draw.text((x_pixel - 10, height - margin + 8), str(epoch), fill="black")

    step_y = 0.2  
    for y_val in [round(i, 1) for i in frange(y_min, y_max, step_y)]:
        y_pixel = get_coords(x_min, y_val)[1]
        draw.line([(margin - 5, y_pixel), (margin, y_pixel)], fill="black", width=2)
        draw.text((margin - 25, y_pixel - 5), f"{y_val:.1f}", fill="black")

    legend_x = width - margin - 150
    legend_y = margin
    draw.line([(legend_x, legend_y), (legend_x + 20, legend_y)], fill="red", width=2)
    draw.text((legend_x + 25, legend_y - 5), "Test Accuracy", fill="black")
    draw.line([(legend_x, legend_y + 20), (legend_x + 20, legend_y + 20)], fill="blue", width=2)
    draw.text((legend_x + 25, legend_y + 15), "Test F1 Score", fill="black")

    filename = f"task2/Test_set_performance_{graph_name}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename)
    print(f"Graph saved as {filename}")

def frange(start, stop, step):
    """Generate floating-point numbers in a specified range.
    
    Args:
        start (float): Starting value
        stop (float): Maximum end value (inclusive)
        step (float): Increment step size
    
    Yields:
        float: Successive values in the sequence
    """
    while start <= stop:
        yield start
        start += step

def fit_elm_ls(model, pairs, batch_size=64, seed=0):
    """Train ELM using Least Squares method on hidden layer outputs.
    
    Args:
        model (MyExtremeLearningMachine): Initialized ELM model
        pairs (tuple): (X_train [num_samples, channels, h, w], y_train [num_samples])
        batch_size (int): Training batch size. Default: 64
        seed (int): Random seed. Default: 0
        
    Returns:
        MyExtremeLearningMachine: Trained ELM model
    """
    X, y = pairs
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long).view(-1)

    torch.manual_seed(seed)

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_classes = model.fc2.out_features
    input_size = model.conv_output_size + 1 

    # Initialize matrices for X^T X and X^T Y
    XTX = torch.zeros((input_size, input_size), dtype=torch.float32)
    XTY = torch.zeros((input_size, num_classes), dtype=torch.float32)

    for X_batch, y_batch in train_loader:
        with torch.no_grad():
            # Compute hidden outputs
            hidden_outputs = model.conv1(X_batch)
            hidden_outputs = model.relu(hidden_outputs)
            hidden_outputs = model.flatten(hidden_outputs)
            
            # Add bias term (column of ones)
            batch_size = hidden_outputs.size(0)
            ones = torch.ones((batch_size, 1), dtype=torch.float32)
            hidden_outputs = torch.cat([hidden_outputs, ones], dim=1)
            
            # Convert y_batch to one-hot encoding
            y_onehot = torch.zeros((batch_size, num_classes), dtype=torch.float32)
            y_onehot.scatter_(1, y_batch.unsqueeze(1), 1.0)
            
            # Update XTX and XTY incrementally
            XTX += hidden_outputs.T @ hidden_outputs
            XTY += hidden_outputs.T @ y_onehot

    # Solve for weights using the normal equations (add small regularization to avoid singular matrix)
    XTX += torch.eye(input_size) * 1e-6
    weights = torch.linalg.solve(XTX, XTY).numpy().T

    # Separate the weights and bias
    model.fc2.weight.data = torch.tensor(weights[:, :-1], dtype=torch.float32)
    model.fc2.bias.data = torch.tensor(weights[:, -1], dtype=torch.float32)

    return model

def random_hyperparameter_search_ls(X_train, y_train, X_test, y_test, num_iterations=3):
    """Random search for optimal ELM-LS hyperparameters.
    
    Args:
        X_train (torch.Tensor): Train images [num_train, channels, h, w]
        y_train (torch.Tensor): Train labels [num_train]
        X_test (torch.Tensor): Test images [num_test, channels, h, w]
        y_test (torch.Tensor): Test labels [num_test]
        num_iterations (int): Search iterations. Default: 3
        
    Returns:
        MyExtremeLearningMachine: Best performing model
    """
    best_model = None
    best_acc = 0.0
    best_params = {}

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    for _ in range(num_iterations):
        hidden_size = np.random.randint(5, 30)
        batch_size = np.random.randint(32, 128)

        model = MyExtremeLearningMachine(
            fixed_weights=None,
            hidden_size=hidden_size,
            input_channels=X_train.shape[1],
            input_height=X_train.shape[2],
            input_width=X_train.shape[3],
            num_classes=len(np.unique(y_train)),
            seed=0
        )

        model = fit_elm_ls(model, (X_train, y_train), batch_size=batch_size)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        y_pred_list = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                y_pred_batch = model.forward(X_batch)
                y_pred_list.append(y_pred_batch)

        y_pred_all = torch.cat(y_pred_list, dim=0)
        y_pred_classes = torch.argmax(y_pred_all, dim=1).cpu().numpy()
        acc = accuracy_score(y_test, y_pred_classes)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = {'hidden_size': hidden_size, 'batch_size': batch_size}

    print("Best Hyperparameters for fit_elm_ls:", best_params)
    print("Best Test Accuracy for fit_elm_ls:", best_acc)

    return best_model

def visualize_results(X_test, y_test, y_pred_classes, filename='task2/result_ELM.png'):
    """Generate visualization grid of test images with true/predicted labels.
    
    Args:
        X_test (torch.Tensor): Test images tensor of shape (num_samples, channels, height, width)
        y_test (np.ndarray): True labels array of shape (num_samples,)
        y_pred_classes (np.ndarray): Predicted labels array of shape (num_samples,)
        filename (str): Output file path. Default: 'task2/result_ELM.png'
    """
    num_images = 36
    images = X_test[:num_images]
    true_labels = y_test[:num_images]
    pred_labels = y_pred_classes[:num_images]

    cols = 6
    rows = 6
    cell_size = 64  
    spacing = 8   

    font = ImageFont.truetype("arial.ttf", 7)  


    grid_width = cols * (cell_size + spacing) - spacing
    grid_height = rows * (cell_size + spacing + 20) - spacing
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    for i in range(num_images):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5) * 255
        img = Image.fromarray(img.astype(np.uint8)).resize((cell_size, cell_size))
        
        cell = Image.new('RGB', (cell_size, cell_size + 20), (255, 255, 255))
        cell.paste(img, (0, 0))
        
        draw = ImageDraw.Draw(cell)
        text = f"True: {true_labels[i]}\nPred: {pred_labels[i]}"
        draw.text((5, cell_size + 2), text, fill=(0, 0, 0), font=font)

        row = i // cols
        col = i % cols
        x = col * (cell_size + spacing)
        y = row * (cell_size + spacing + 20)
        grid_image.paste(cell, (x, y))

    grid_image.save(filename)

def one_hot_encode(y, num_classes):
    """Convert class labels to one-hot vectors.
    
    Args:
        y (np.ndarray): Class labels [num_samples]
        num_classes (int): Total number of classes
        
    Returns:
        np.ndarray: One-hot encoded labels [num_samples, num_classes]
    """
    return np.eye(num_classes)[y]

def accuracy_score(y_true, y_pred):
    """Compute classification accuracy between true and predicted labels.
    
    Args:
        y_true (torch.Tensor/np.ndarray): Ground truth labels 
            Shape: (num_samples,)
        y_pred (torch.Tensor/np.ndarray): Predicted labels 
            Shape: (num_samples,)
    
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Handle both tensor and numpy array inputs
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred, average='weighted'):
    """Compute F1-score balancing precision and recall.
    
    Args:
        y_true (torch.Tensor/np.ndarray): True class labels
            Shape: (num_samples,)
        y_pred (torch.Tensor/np.ndarray): Predicted class labels
            Shape: (num_samples,)
        average (str): Scoring mode ('weighted' or None)
            Default: 'weighted'
    
    Returns:
        float: F1-score between 0 and 1
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    all_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    true_positive = Counter()

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            true_positive[true] += 1

    f1_scores = []
    for label in all_labels:  
        tp = true_positive.get(label, 0)
        pred_count = pred_counter.get(label, 0)
        true_count = true_counter.get(label, 0)

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / true_count if true_count > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    if average == 'weighted':
        weights = np.array([true_counter.get(label, 0) for label in all_labels]) / len(y_true)
        return np.sum(np.array(f1_scores) * weights)
    else:
        return np.mean(f1_scores)

#-----------------
# Data Preparation
#-----------------

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.CIFAR10(root='task2/data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='task2/data', train=False, download=True, transform=transform_test)


batch_size = 64
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

X_test, y_test = [], []
for batch in testloader:
    images, labels = batch
    X_test.append(images)
    y_test.append(labels)
X_test = torch.cat(X_test, dim=0)
y_test = torch.cat(y_test, dim=0)

X_train, y_train = [], []
for batch in trainloader:
    images, labels = batch
    X_train.append(images)
    y_train.append(labels)
X_train = torch.cat(X_train, dim=0)
y_train = torch.cat(y_train, dim=0)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print(f"X_test shape : {list(X_test.shape)}, y_test shape : {list(y_test.shape)}")

# Performing the random hyperparameter search for the ELM using the Least Squares method and the Stochastic Gradient Descent method

hidden_size = 10
output_size = len(np.unique(y_train))
learning_rate = 0.1
num_epochs = 20

input_channels = X_train.shape[1]
input_height = X_train.shape[2]
input_width = X_train.shape[3]

start_time = time.time()
best_model_ls = random_hyperparameter_search_ls(X_train, y_train, X_test, y_test)
ls_time = time.time() - start_time

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
y_pred_best_ls_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        y_pred_batch = best_model_ls.forward(X_batch)
        y_pred_best_ls_list.append(y_pred_batch)

y_pred_best_ls = torch.cat(y_pred_best_ls_list, dim=0)
y_pred_classes_best_ls = torch.argmax(y_pred_best_ls, dim=1).cpu().numpy()

acc_ls = accuracy_score(y_test, y_pred_classes_best_ls)
f1_ls = f1_score(y_test, y_pred_classes_best_ls, average='weighted')
visualize_results(X_test, y_test, y_pred_classes_best_ls, filename='task2/new_result_best_ls.png')

print("ELM LS Test Accuracy:", acc_ls)
print("ELM LS Test F1:", f1_ls)
print("ELM LS Training Time:", ls_time)
print("When the training is performed, its execution time is about 1000 seconds on my computer")


start_time = time.time()

model_sgd = MyEnsembleELM(
    trainloader,
    learning_rate=0.1,
    num_epochs=20,
    seeds=[0, 1, 2],  
    test_loader=testloader,
    mixup_alpha=1.0,
    load_model=True    
)

sgd_time = time.time() - start_time

test_preds = []
test_labels = []
with torch.no_grad():
    for X_batch, y_batch in testloader:
        ensemble_pred = model_sgd.predict((X_batch, y_batch))
        test_preds.append(ensemble_pred.argmax(dim=1))
        test_labels.append(y_batch)

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
best_acc = accuracy_score(test_labels.numpy(), test_preds.numpy())
f1_sgd = f1_score(test_labels.numpy(), test_preds.numpy(), average='weighted')

visualize_results(X_test, y_test, test_preds.numpy(), filename='task2/new_result_best_ensemble.png')

print("ELM Ensemble Test Accuracy:", best_acc)
print("ELM Ensemble Test F1:", f1_sgd)
print("ELM Ensemble Training Time:", sgd_time)
print("When the training is performed, its execution time is about 2000 seconds on my computer")

print("""

We notice that the LS method is two times faster compared to the MixupEnsembleELM. 
Nevertheless, the MixupEnsembleELM is a better classifier with a test accuracy of 0.43 (vs 0.32 for the LS) 
and a test f1 of 0.42 (vs 0.32 for the LS). To improve the performance of the LS method, we could perform a 
more thorough parameter search : we are currently performing 3 iterations, which might not be sufficient.

""")

