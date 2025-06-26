import numpy as np
import torch
from itertools import combinations_with_replacement
import torch.nn as nn
import torch.optim as optim
from collections import Counter

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

class MyCrossEntropy:
    """A class to compute Binary Cross Entropy loss.
    
    Attributes:
        eps (float): Small value to avoid log(0). Default is 1e-10.
    """
    def __init__(self):
        self.eps = 1e-10

    def binary_cross_entropy(self, truth, predictions):
        """Compute BCE loss between predictions and ground truth.
        
        Args:
            truth (torch.Tensor): Ground truth labels. Shape (batch_size, 1). Values in [0, 1].
            predictions (torch.Tensor): Model predictions. Shape (batch_size, 1). Values in (0, 1).
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Add epsilon to avoid logarithm of zero (numerical stability)
        loss = - (truth * torch.log(predictions + self.eps) + (1 - truth) * torch.log(1 - predictions + self.eps))
        return loss.mean()
    
class MyRootMeanSquare:
    """A class to compute Root Mean Square Error (RMSE)."""

    def __init__(self):
        pass

    def rmse(self, truth, predictions):
        """Compute RMSE between predictions and ground truth.
        
        Args:
            truth (torch.Tensor): Ground truth values. Shape (batch_size, 1).
            predictions (torch.Tensor): Model predictions. Shape (batch_size, 1).
            
        Returns:
            torch.Tensor: Scalar RMSE value.
        """
        mse = torch.mean((truth - predictions) ** 2)
        return torch.sqrt(mse)

class LogisticRegressionModel(nn.Module):
    """Logistic Regression model using polynomial feature expansion.
    
    Args:
        input_dim (int): Number of input features after polynomial expansion.
    """
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """Forward pass with sigmoid activation.
        
        Args:
            x (torch.Tensor): Input tensor. Shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Probability outputs. Shape (batch_size, 1).
        """
        return torch.sigmoid(self.linear(x))
 

def comb(n, k):
    """Compute binomial coefficient C(n, k) efficiently.
    
    Args:
        n (int): Total number of elements.
        k (int): Number of elements to choose.
        
    Returns:
        int: Binomial coefficient value.
    """
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k) 
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

def accuracy_score(y_true, y_pred):
    """Compute classification accuracy.
    
    Args:
        y_true (np.ndarray): True labels. Shape (n_samples,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted labels. Shape (n_samples,). Values in {0, 1}.
        
    Returns:
        float: Accuracy between 0 and 1.
    """
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred, average='weighted'):
    """Compute F1 score for binary/multiclass classification.
    
    Args:
        y_true (np.ndarray): True labels. Shape (n_samples,). Integer values.
        y_pred (np.ndarray): Predicted labels. Shape (n_samples,). Integer values.
        average (str): Averaging method ('weighted' or None). Default 'weighted'.
        
    Returns:
        float: Weighted or macro F1 score between 0 and 1.
    """
    true_counter = Counter(int(label) for label in y_true)
    pred_counter = Counter(int(label) for label in y_pred)
    true_positive = Counter()

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            true_positive[true] += 1

    f1_scores = []
    for label in range(max(int(key) for key in true_counter.keys()) + 1):
        precision = true_positive[label] / pred_counter[label] if pred_counter[label] > 0 else 0
        recall = true_positive[label] / true_counter[label] if true_counter[label] > 0 else 0
        if precision + recall > 0:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
        else:
            f1_scores.append(0.0)

    if average == 'weighted':
        weights = np.array([true_counter[label] for label in range(len(f1_scores))]) / sum(true_counter.values())
        return np.sum(np.array(f1_scores) * weights)
    else:
        return np.mean(f1_scores)

def logistic_fun(w, x, M):
    """Compute logistic function with polynomial feature expansion.
    
    Args:
        w (torch.Tensor): Weight vector. Shape (p,).
        x (np.ndarray/torch.Tensor): Input vector. Shape (D,).
        M (int): Maximum polynomial degree.
        
    Returns:
        torch.Tensor: Probability output after sigmoid. Scalar value.
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    prod = [torch.tensor([1.0], dtype=torch.float32, device=x.device)]
    
    # Generate all polynomial terms up to degree M
    for r in range(1, M + 1):
        for comb in combinations_with_replacement(x, r):
            term = torch.prod(torch.stack([item.clone().detach() for item in comb]))
            prod.append(term.unsqueeze(0))
    prod = torch.cat(prod)

    if w.shape[0] != prod.shape[0]:
        raise ValueError(f"The size of w ({w.shape[0]}) doesn't correspond to the number of terms of the polynomial ({prod.shape[0]}).")

    f = torch.dot(w, prod)
    return torch.sigmoid(f)
   
def transform_X(X, M):
    """Generate polynomial features up to degree M for all samples.
    
    Args:
        X (np.ndarray/torch.Tensor): Input data. Shape (n_samples, D).
        M (int): Maximum polynomial degree.
        
    Returns:
        torch.Tensor: Transformed features. Shape (n_samples, p).
    """
    X_transformed = []
    for x in X:
        prod = [torch.tensor([1.0], dtype=torch.float32)]
        for r in range(1, M + 1):
            for comb in combinations_with_replacement(x, r):
                term = torch.prod(torch.stack([torch.tensor(item).clone().detach() for item in comb]))
                prod.append(term.unsqueeze(0))
        prod = torch.cat(prod)
        X_transformed.append(prod)
    return torch.stack(X_transformed)

def fit_logistic_sgd(pairs, loss_function, minibatch_size, learning_rate, num_epochs, M):
    """Train logistic regression model with SGD.
    
    Args:
        pairs (tuple): (X_train, y_train) where:
            X_train (np.ndarray): Training data. Shape (n_samples, D).
            y_train (np.ndarray): Labels. Shape (n_samples,).
        loss_function (str): "BinaryCrossEntropy" or "RMSE".
        minibatch_size (int): Size of mini-batches.
        learning_rate (float): SGD learning rate.
        num_epochs (int): Number of training epochs.
        M (int): Polynomial degree for feature expansion.
        
    Returns:
        tuple: (model, loss, accuracy, f1_score) trained model and metrics.
    """
    X, y = pairs

    # Convert numpy arrays to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    X_transformed = transform_X(X, M)
    input_dim = X_transformed.shape[1]
    model = LogisticRegressionModel(input_dim)

    # Weight decay for L2 regularization (implicit via optimizer)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)

    if loss_function == "BinaryCrossEntropy":
        loss_fn = MyCrossEntropy().binary_cross_entropy
    elif loss_function == "RMSE":
        loss_fn = MyRootMeanSquare().rmse
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    for epoch in range(num_epochs):
        for i in range(0, len(X_transformed), minibatch_size):
            X_batch = X_transformed[i:i+minibatch_size]
            y_batch = y[i:i+minibatch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_batch, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Print metrics every 10% of total epochs
        if epoch % (num_epochs // 10) == 0:
            with torch.no_grad():
                y_pred_all = model(X_transformed)
                predictions = (y_pred_all >= 0.5).float()

                y_np = y.numpy().flatten()
                pred_np = predictions.numpy().flatten()

                acc = accuracy_score(y_np, pred_np)
                f1 = f1_score(y_np, pred_np)

                w, b = model.linear.weight, model.linear.bias
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                #print(f"w = {w.flatten().tolist()}")
                #print(f"b = {b.item():.4f}")
                print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    return model, loss.item(), acc, f1, w.flatten().tolist()

def compute_p(D, M):
    """Compute number of polynomial terms for D features up to degree M.
    
    Args:
        D (int): Original feature dimension.
        M (int): Maximum polynomial degree.
        
    Returns:
        int: Total number of polynomial terms (p).
    """
    return int(sum(comb(D + m - 1, m) for m in range(M + 1)))

def compute_w(p):
    """Generate weight vector with alternating signs.
    
    Args:
        p (int): Number of weights (polynomial terms).
        
    Returns:
        torch.Tensor: Weight vector. Shape (p,).
    """
    return torch.tensor([(-1)**(p - i) * torch.sqrt(torch.tensor(p - i)) / p for i in range(p)], dtype=torch.float32)

def pretty_print_dict(d, indent=0):
    """Display a dictionary in a readable way with indentations.

    Args:
        d (dict): Dictionary to display.
        indent (int): Initial indentation level.
    """
    for key, value in d.items():
        print(' ' * indent + str(key), end=': ')
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 4)
        else:
            print(value)

#----------
#Main code
#----------

dict_results = {}

# Generate synthetic data with 5 features
X_true = np.random.uniform(-5, 5, size=(300, 5))

for lossfunc in ["BinaryCrossEntropy", "RMSE"]:
    for m in range(1, 4):
        p = compute_p(D=5, M=m)
        w = compute_w(p)
        print(p, w.shape)

        Y_true_true = np.array([logistic_fun(w, x, M=m).item() for x in X_true])
        Y_true = np.array(Y_true_true) + np.random.normal(0, 1, (300,))

        # Convert probabilities to binary labels with threshold 0.5
        Y_true_true = np.where(Y_true_true < 0.5, 0, 1)
        Y_true = np.where(Y_true < 0.5, 0, 1)

        X_train, X_test = X_true[:200], X_true[200:]
        y_train_noisy = Y_true[:200]          
        y_train_true = Y_true_true[:200]      
        y_test_noisy = Y_true[200:]
        y_test_true = Y_true_true[200:]

        model, training_loss, training_accuracy, training_f1, w = fit_logistic_sgd(
            (X_train, y_train_noisy), lossfunc, 64, 0.01, 3000, m
        )
        print("Final training loss:", training_loss)

        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(transform_X(X_train, m), dtype=torch.float32)
            X_test_tensor = torch.tensor(transform_X(X_test, m), dtype=torch.float32)

            y_train_pred = model(X_train_tensor)
            y_test_pred = model(X_test_tensor)

            y_train_pred_labels = (y_train_pred > 0.5).int().numpy().flatten()
            y_test_pred_labels = (y_test_pred > 0.5).int().numpy().flatten()

        #print("y_train predicted:", y_train_pred_labels)
        #print("y_test predicted:", y_test_pred_labels)

        if lossfunc == "BinaryCrossEntropy":
            test_loss = MyCrossEntropy().binary_cross_entropy(
                torch.tensor(y_test_noisy, dtype=torch.float32).view(-1, 1),
                torch.tensor(y_test_pred_labels, dtype=torch.float32).view(-1, 1)
            )
        elif lossfunc == "RMSE":
            test_loss = MyRootMeanSquare().rmse(
                torch.tensor(y_test_noisy, dtype=torch.float32).view(-1, 1),
                torch.tensor(y_test_pred_labels, dtype=torch.float32).view(-1, 1)
            )
        else:
            raise ValueError(f"Unknown loss function: {lossfunc}")

        print("Test Loss: ", test_loss.detach().item())

        train_accuracy_noisy = accuracy_score(y_train_noisy, y_train_pred_labels)
        train_f1_noisy = f1_score(y_train_noisy, y_train_pred_labels)
        test_accuracy_noisy = accuracy_score(y_test_noisy, y_test_pred_labels)
        test_f1_noisy = f1_score(y_test_noisy, y_test_pred_labels)
        test_accuracy_true = accuracy_score(y_test_true, y_test_pred_labels)
        test_f1_true = f1_score(y_test_true, y_test_pred_labels)
        train_accuracy_true = accuracy_score(y_train_true, y_train_pred_labels)
        train_f1_true = f1_score(y_train_true, y_train_pred_labels)

        print("Train Accuracy (noisy):", train_accuracy_noisy)
        print("Train Accuracy (true): ", train_accuracy_true)
        print("Train F1 (noisy):      ", train_f1_noisy)
        print("Train F1 (true):       ", train_f1_true)
        print("Test Accuracy (noisy):", test_accuracy_noisy)
        print("Test F1 Score (noisiy):", test_f1_noisy)
        print("Test Accuracy (true):", test_accuracy_true)
        print("Test F1 Score (true):", test_f1_true)

        key = f"LossFunction : {lossfunc} M : {m}"
        dict_results[key] = {
            "Training Loss": training_loss,
            "Test Loss (noisy)": test_loss.item(),
            "Train Accuracy (noisy)": training_accuracy,
            "Test Accuracy (noisy)": test_accuracy_noisy,
            "Test Accuracy (true)": test_accuracy_true,
            "Test F1 (noisy)": test_f1_noisy,
            "Test F1 (true)": test_f1_true,      
        }

print("\n --- Results --- \n")
pretty_print_dict(dict_results)

print("""\n I chose accuracy and F1 score to evaluate my model because accuracy 
measures the overall correctness of predictions, while F1 score balances precision and 
recall, providing a better assessment for imbalanced datasets. Together, they offer a 
comprehensive view of model performance, ensuring both high correctness and robustness across different classes.""")

print("""\n Using the final report, we notice that : 
    - Underfitting is observed in models with M=1 for both loss functions, as indicated by high test losses and poor test accuracies.
    - Overfitting is observed in models with M=3 for both loss functions, as indicated by low training losses but higher test losses and discrepancies in accuracy.
    - The model with M=2 using RMSE loss shows signs of overfitting but performs better than the binary cross-entropy counterpart.
    - Overall, the RMSE loss function with M=2 or M=3 provides a better balance, though careful tuning is needed to avoid overfitting.""")

print("""\n Cross-Entropy loss optimizes predicted probabilities with a logarithmic penalty for misclassifications, making 
      it ideal for binary classification by emphasizing accurate class probability estimation. Conversely, RMSE loss views 
      outputs as continuous values, calculating the square root of average squared differences, and measures overall error 
      magnitude without being tailored for classification. When comparing model predictions and training data against true classes, 
      Cross-Entropy typically yields sharper decision boundaries, whereas RMSE results in less precise classifications.""")