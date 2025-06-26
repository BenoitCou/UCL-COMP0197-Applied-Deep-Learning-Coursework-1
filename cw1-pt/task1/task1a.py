import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations_with_replacement
import numpy as np
from collections import Counter

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
    """Logistic Regression with adaptive polynomial feature selection.
    
    Args:
        max_degree (int): Maximum allowed polynomial degree
        input_dim (int): Number of input features in original data
    """
    def __init__(self, max_degree, input_dim):
        super().__init__()
        self.max_degree = max_degree
        self.input_dim = input_dim
        self.M_raw = nn.Parameter(torch.tensor(0.0)) 
        
        self.linear = nn.Linear(self._get_feature_count(), 1)
        
    def _get_feature_count(self):
        """Calculate total number of possible polynomial features up to max_degree.
        
        Returns:
            int: Total number of polynomial terms (p)
        """
        count = 1
        for d in range(1, self.max_degree + 1):
            count += len(list(combinations_with_replacement(range(self.input_dim), d)))
        return count

    @property
    def M(self):
        """Learnable polynomial degree (soft threshold between 1 and max_degree)
        
        Returns:
            torch.Tensor: Effective polynomial degree (scalar value)
        """
        return torch.sigmoid(self.M_raw) * (self.max_degree - 1) + 1

    def transform_x(self, x):
        """Generate adaptive polynomial features with degree weighting.
        
        Args:
            x (torch.Tensor): Input tensor. Shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Transformed features with adaptive weights. Shape (batch_size, p)
        """
        batch_size = x.size(0)
        features = [torch.ones(batch_size, 1, device=x.device)]
        
        # Generate polynomial terms with soft degree thresholding
        for d in range(1, self.max_degree + 1):
            # Soft thresholding: features get exponentially smaller weights as d exceeds learned M
            weight = torch.sigmoid(10 * (self.M - d + 0.5))
            
            # Generate all d-degree combinations
            for comb in combinations_with_replacement(range(self.input_dim), d):
                prod = torch.prod(x[:, comb], dim=1, keepdim=True)
                weighted_prod = prod * weight
                features.append(weighted_prod)
        
        return torch.cat(features, dim=1)

    def forward(self, x):
        """Forward pass with adaptive feature transformation.
        
        Args:
            x (torch.Tensor): Input tensor. Shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Probability outputs. Shape (batch_size, 1)
        """
        x_trans = self.transform_x(x)
        return torch.sigmoid(self.linear(x_trans))

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
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    true_positive = Counter()

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            true_positive[true] += 1

    f1_scores = []
    max_label = max(max(true_counter.keys(), default=0), max(pred_counter.keys(), default=0))
    for label in range(max_label + 1):
        precision_numerator = true_positive.get(label, 0)
        precision_denominator = pred_counter.get(label, 0)
        precision = precision_numerator / precision_denominator if precision_denominator else 0.0
        
        recall_numerator = true_positive.get(label, 0)
        recall_denominator = true_counter.get(label, 0)
        recall = recall_numerator / recall_denominator if recall_denominator else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    if average == 'weighted':
        total = len(y_true)
        weights = np.array([true_counter.get(label, 0) for label in range(max_label + 1)]) / total
        return np.sum(np.array(f1_scores) * weights)
    return np.mean(f1_scores)

def fit_logistic_sgd(pairs, loss_function, learning_rate, num_epochs, max_degree=5):
    """Train logistic regression with adaptive polynomial features.
    
    Args:
        pairs (tuple): (X_train, y_train) where:
            X_train (np.ndarray): Training data. Shape (n_samples, D)
            y_train (np.ndarray): Labels. Shape (n_samples,)
        loss_function (str): "BinaryCrossEntropy" or "RMSE"
        learning_rate (float): SGD learning rate
        num_epochs (int): Number of training epochs
        max_degree (int): Maximum polynomial degree to consider. Default 5
        
    Returns:
        LogisticRegressionModel: Trained model with learned feature weights
    """
    X, y = pairs
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LogisticRegressionModel(max_degree=max_degree, input_dim=X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    if loss_function == "BinaryCrossEntropy":
        loss_fn = MyCrossEntropy().binary_cross_entropy
    elif loss_function == "RMSE":
        loss_fn = MyRootMeanSquare().rmse
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        # Regularization term encouraging smaller M values (simpler models)
        loss += 0.1 * torch.sigmoid(model.M_raw)
        
        loss.backward()
        optimizer.step()

        if epoch % (num_epochs // 10) == 0:
            with torch.no_grad():
                predictions = (outputs > 0.5).float()
                y_np = y.numpy().flatten()
                pred_np = predictions.numpy().flatten()
                
                acc = accuracy_score(y_np, pred_np)
                f1 = f1_score(y_np, pred_np)
                w, b = model.linear.weight, model.linear.bias
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, M: {model.M.item():.2f}")
                print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}\n")

    return model, w.flatten().tolist()

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
    return torch.tensor([np.sqrt(p_)/p*(-1)**p_ for p_ in range(p, 0, -1)], dtype=torch.float32)

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

#----------
#Main code
#----------

# Generate synthetic data with 5 features (D=5) and M=3
D = 5
M_true = 3
X_true = np.random.uniform(-5, 5, size=(300, D))
p = compute_p(D=D, M=M_true)
w = compute_w(p)

Y_true_true = np.array([logistic_fun(w, x, M=M_true).item() for x in X_true])
Y_true = Y_true_true + np.random.normal(0, 1, (300,))

# Convert probabilities to binary labels with threshold 0.5
Y_true_true = np.where(Y_true_true < 0.5, 0, 1)
Y_true = np.where(Y_true < 0.5, 0, 1)

X_train, X_test = X_true[:200], X_true[200:]
y_train = Y_true[:200]           
y_test = Y_true[200:]           
y_test_true = Y_true_true[200:]

model, w = fit_logistic_sgd(
    (X_train, y_train),
    "BinaryCrossEntropy",
    learning_rate=0.01,
    num_epochs=3000,
    max_degree=5
)

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_pred = model(X_test_tensor)
    y_test_pred_labels = (y_test_pred > 0.5).float().numpy().flatten()

    test_acc_noisy = accuracy_score(y_test, y_test_pred_labels)
    test_f1_noisy = f1_score(y_test, y_test_pred_labels)

    test_acc_true = accuracy_score(y_test_true, y_test_pred_labels)
    test_f1_true = f1_score(y_test_true, y_test_pred_labels)
    
    print(f"Test Accuracy (noisy targets) = {test_acc_noisy:.4f}")
    print(f"Test F1 (noisy targets) = {test_f1_noisy:.4f}")
    print(f"Test Accuracy (true targets) = {test_acc_true:.4f}")
    print(f"Test F1 (true targets) = {test_f1_true:.4f}")
    print(f"Optimized M: {model.M.item():.2f}")