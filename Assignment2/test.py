import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import logging
import numpy.linalg as LA

# Set global font for plots
font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

#########################################################
# 1. Define Layers / Network / Loss / Evaluation Functions
#########################################################

class SigmoidCrossEntropy:
    def __init__(self):
        self.logits = None
        self.labels = None
        self.probs = None
        self.eps = 1e-12

    def forward(self, logits, labels):
        """
        Compute the forward pass for sigmoid cross-entropy.
        Reshape labels to (batch_size, 1) to ensure proper element-wise operations.
        """
        self.logits = logits
        self.labels = labels.reshape(-1, 1)
        self.probs = 1.0 / (1.0 + np.exp(-logits))
        loss = - np.mean(
            self.labels * np.log(self.probs + self.eps) +
            (1 - self.labels) * np.log(1 - self.probs + self.eps)
        )
        return loss

    def backward(self):
        """
        Compute the gradient of the loss with respect to logits.
        The gradient is: (sigmoid(logits) - labels) / batch_size.
        """
        batch_size = self.logits.shape[0]
        dlogits = (self.probs - self.labels) / batch_size
        return dlogits  # Shape: (batch_size, 1)


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        mask = (self.input > 0).astype(float)
        return grad * mask

    def step(self, step_size, momentum=0, weight_decay=0):
        # ReLU has no parameters to update.
        pass


class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # Initialize weights with small random numbers and biases with zeros.
        # (你也可以根據需要調整初始化方法)
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.vel_w = np.zeros_like(self.weights)
        self.vel_b = np.zeros_like(self.bias)
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, dZ):
        # Compute gradients for weights and biases.
        self.grad_weights = self.input.T @ dZ
        self.grad_bias = np.sum(dZ, axis=0, keepdims=True)
        # Propagate gradient to previous layer.
        dX = dZ @ self.weights.T
        return dX

    def step(self, step_size, momentum=0.8, weight_decay=0.0):
        # Update weights with momentum and weight decay.
        self.vel_w = momentum * self.vel_w + step_size * (self.grad_weights + weight_decay * self.weights)
        self.weights -= self.vel_w
        # Update biases with momentum.
        self.vel_b = momentum * self.vel_b + step_size * self.grad_bias
        self.bias -= self.vel_b


class FeedForwardNeuralNetwork:
    """
    Constructs a feed-forward neural network.
    If num_layers == 1, creates a single linear layer (input_dim -> 1).
    Otherwise, constructs a network: [Linear(input_dim, hidden_dim), ReLU, ..., Linear(hidden_dim, 1)]
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        self.layers = []
        output_dim = 1  # For binary classification
        if num_layers == 1:
            self.layers.append(LinearLayer(input_dim, output_dim))
        else:
            # First layer
            self.layers.append(LinearLayer(input_dim, hidden_dim))
            self.layers.append(ReLU())
            # Additional hidden layers (if any)
            for _ in range(num_layers - 2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(ReLU())
            # Final output layer: hidden_dim -> 1
            self.layers.append(LinearLayer(hidden_dim, output_dim))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size, momentum=0.8, weight_decay=0.0, freeze_first=False):
        """
        Update each layer's parameters.
        If freeze_first is True, skip updating the first layer.
        """
        for i, layer in enumerate(self.layers):
            if freeze_first and i == 0:
                continue
            layer.step(step_size, momentum, weight_decay)

def evaluate(model, X_val, Y_val, batch_size):
    """
    Evaluate the model on the validation/test set.
    Returns the average loss and accuracy.
    """
    criterion = SigmoidCrossEntropy()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for i in range(0, len(X_val), batch_size):
        X_batch = X_val[i:i+batch_size]
        Y_batch = Y_val[i:i+batch_size]
        logits = model.forward(X_batch)
        batch_loss = criterion.forward(logits, Y_batch)
        total_loss += batch_loss * len(X_batch)
        p = 1.0 / (1.0 + np.exp(-logits))
        preds = (p > 0.5).astype(int).reshape(-1)
        total_correct += np.sum(preds == Y_batch)
        total_samples += len(X_batch)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

#########################################################
# 2. Main Program: CIFAR Training and Hyperparameter Sweeping
#########################################################

def train_one_model(X_train, Y_train, X_test, Y_test,
                    input_dim,
                    hidden_dim=100,
                    num_layers=2,
                    batch_size=100,
                    max_epochs=5,
                    lr=1e-2,
                    momentum=0.8,
                    weight_decay=0.0):
    """
    Train a model with the specified hyperparameters and return the final test accuracy.
    """
    net = FeedForwardNeuralNetwork(input_dim, hidden_dim, num_layers)
    criterion = SigmoidCrossEntropy()
    num_examples = X_train.shape[0]
    for epoch in range(max_epochs):
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        for start_idx in range(0, num_examples, batch_size):
            end_idx = min(start_idx + batch_size, num_examples)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_train[batch_idx]
            Y_batch = Y_train[batch_idx]
            logits = net.forward(X_batch)
            loss = criterion.forward(logits, Y_batch)
            dlogits = criterion.backward()
            net.backward(dlogits)
            net.step(lr, momentum, weight_decay)  # 此處不用 freeze_first
        # (每個 epoch 後可評估訓練狀況)
    _, test_acc = evaluate(net, X_test, Y_test, batch_size)
    return test_acc

def sweep_parameters(X_train, Y_train, X_test, Y_test, input_dim):
    """
    Sweep through hyperparameters: batch_size, learning rate, and hidden_dim.
    Returns tuples of parameter values and corresponding test accuracies.
    """
    # Sweep batch_size
    batch_list = [16, 32, 64, 128, 256]
    acc_list_bs = []
    for bs in batch_list:
        acc = train_one_model(X_train, Y_train, X_test, Y_test,
                              input_dim=input_dim, hidden_dim=100, num_layers=2,
                              batch_size=bs, max_epochs=5, lr=1e-2,
                              momentum=0.8, weight_decay=0.0)
        acc_list_bs.append(acc)
        print(f"[batch_size={bs}] => test_acc={acc:.4f}")
    # Sweep learning rate
    lr_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    acc_list_lr = []
    for lr_val in lr_list:
        acc = train_one_model(X_train, Y_train, X_test, Y_test,
                              input_dim=input_dim, hidden_dim=100, num_layers=2,
                              batch_size=100, max_epochs=5, lr=lr_val,
                              momentum=0.8, weight_decay=0.0)
        acc_list_lr.append(acc)
        print(f"[learning_rate={lr_val}] => test_acc={acc:.4f}")
    # Sweep hidden_dim
    hidden_list = [50, 100, 200, 300, 500]
    acc_list_hd = []
    for hd in hidden_list:
        acc = train_one_model(X_train, Y_train, X_test, Y_test,
                              input_dim=input_dim, hidden_dim=hd, num_layers=2,
                              batch_size=100, max_epochs=5, lr=1e-2,
                              momentum=0.8, weight_decay=0.0)
        acc_list_hd.append(acc)
        print(f"[hidden_dim={hd}] => test_acc={acc:.4f}")
    return (batch_list, acc_list_bs), (lr_list, acc_list_lr), (hidden_list, acc_list_hd)

def main():
    # Load CIFAR 2-class data
    data = pickle.load(open('cifar_2class_py3.p', 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test  = data['test_data']
    Y_test  = data['test_labels']
    # Check and recode labels to 0 and 1 if necessary
    unique_train = np.unique(Y_train)
    print("Unique training labels before recoding:", unique_train)
    if not np.array_equal(unique_train, [0, 1]):
        label_map = {unique_train[0]: 0, unique_train[1]: 1}
        Y_train = np.array([label_map[y] for y in Y_train])
        Y_test = np.array([label_map[y] for y in Y_test])
        print("Labels recoded to 0 and 1.")
    # Normalize data to [0,1]
    X_train = X_train / 255.0
    X_test  = X_test / 255.0
    num_examples, input_dim = X_train.shape
    batch_size = 100
    max_epochs = 5
    step_size = 1e-2  # Learning rate
    hidden_dim = 100
    num_layers = 2
    momentum = 0.8
    weight_decay = 0.0

    # Build and train a single model (for demonstration)
    net = FeedForwardNeuralNetwork(input_dim, hidden_dim, num_layers)
    criterion = SigmoidCrossEntropy()
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    epoch_x = []
    batches_per_epoch = int(np.ceil(num_examples / batch_size))
    for epoch in range(max_epochs):
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_count = 0
        for start_idx in range(0, num_examples, batch_size):
            end_idx = min(start_idx + batch_size, num_examples)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_train[batch_idx]
            Y_batch = Y_train[batch_idx]
            logits = net.forward(X_batch)
            loss = criterion.forward(logits, Y_batch)
            dlogits = criterion.backward()
            net.backward(dlogits)
            net.step(step_size, momentum, weight_decay)
            losses.append(loss)
            epoch_loss_sum += loss * len(X_batch)
            p = 1.0 / (1.0 + np.exp(-logits))
            preds = (p > 0.5).astype(int).reshape(-1)
            epoch_correct += np.sum(preds == Y_batch)
            epoch_count += len(X_batch)
            accs.append(epoch_correct / epoch_count)
        epoch_avg_loss = epoch_loss_sum / epoch_count
        epoch_avg_acc = epoch_correct / epoch_count
        val_loss, val_acc = evaluate(net, X_test, Y_test, batch_size)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epoch_x.append((epoch+1) * batches_per_epoch)
        logging.info(f"[Epoch {epoch+1}/{max_epochs}] TrainLoss={epoch_avg_loss:.4f} TrainAcc={epoch_avg_acc:.4f} TestLoss={val_loss:.4f} TestAcc={val_acc:.4f}")
    # Hyperparameter sweeping (繪圖略)
    sweep_bs, sweep_lr, sweep_hd = sweep_parameters(X_train, Y_train, X_test, Y_test, input_dim)
    fig, axs = plt.subplots(2,2, figsize=(18,12))
    ax1 = axs[0,0]
    color_loss = 'tab:red'
    ax1.set_xlabel("Iterations (per batch)")
    ax1.set_ylabel("Train Loss", color=color_loss)
    ax1.plot(range(len(losses)), losses, c=color_loss, alpha=0.5, label="Train Loss")
    ax1.plot(epoch_x, val_losses, 'o-', c='darkred', label="Test Loss")
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.legend(loc="upper left")
    ax1.set_title("Training Curves (Single Run)")
    ax1b = ax1.twinx()
    color_acc = 'tab:blue'
    ax1b.set_ylabel("Accuracy", color=color_acc)
    ax1b.plot(range(len(accs)), accs, c=color_acc, alpha=0.3, label="Train Acc")
    ax1b.plot(epoch_x, val_accs, 'o-', c='blue', label="Test Acc")
    ax1b.tick_params(axis='y', labelcolor=color_acc)
    ax1b.set_ylim([0,1.0])
    ax1b.legend(loc="upper right")
    ax2 = axs[0,1]
    batch_list, acc_list_bs = sweep_bs
    ax2.plot(batch_list, acc_list_bs, marker='o')
    ax2.set_title("Test Accuracy vs. Batch Size")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Test Accuracy")
    ax2.grid(True)
    ax3 = axs[1,0]
    lr_list, acc_list_lr = sweep_lr
    ax3.plot(lr_list, acc_list_lr, marker='o')
    ax3.set_title("Test Accuracy vs. Learning Rate")
    ax3.set_xlabel("Learning Rate")
    ax3.set_ylabel("Test Accuracy")
    ax3.grid(True)
    ax4 = axs[1,1]
    hidden_list, acc_list_hd = sweep_hd
    ax4.plot(hidden_list, acc_list_hd, marker='o')
    ax4.set_title("Test Accuracy vs. Hidden Units")
    ax4.set_xlabel("Hidden Units")
    ax4.set_ylabel("Test Accuracy")
    ax4.grid(True)
    plt.tight_layout()
    plt.show()

#########################################################
# XOR Testing Section (for debugging gradient and layer updates)
#########################################################

def train_xor(freeze_first=True, epochs=5000, lr=0.1):
    """
    Train the network on XOR data.
    If freeze_first is True, do not update the first layer.
    """
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    Y = np.array([0,1,1,0])
    net = FeedForwardNeuralNetwork(input_dim=2, hidden_dim=4)
    criterion = SigmoidCrossEntropy()
    for epoch in range(epochs):
        logits = net.forward(X)
        loss = criterion.forward(logits, Y)
        dlogits = criterion.backward()
        net.backward(dlogits)
        net.step(lr, momentum=0.8, weight_decay=0.0, freeze_first=freeze_first)
        if epoch % 500 == 0:
            preds = (1.0/(1.0+np.exp(-net.forward(X))) > 0.5).astype(int).reshape(-1)
            accuracy = np.mean(preds == Y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    preds = (1.0/(1.0+np.exp(-net.forward(X))) > 0.5).astype(int).reshape(-1)
    accuracy = np.mean(preds == Y)
    print("Final predictions:", preds)
    print("Final accuracy:", accuracy*100, "%")

def train_xor_with_grad_check(freeze_first=False, epochs=5000, lr=0.1):
    """
    Train the network on XOR data with gradient norm checking.
    Every 500 epochs, print the loss and gradient norms (for each layer with grad_weights).
    """
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    Y = np.array([0,1,1,0])
    net = FeedForwardNeuralNetwork(input_dim=2, hidden_dim=4)
    criterion = SigmoidCrossEntropy()
    for epoch in range(epochs):
        logits = net.forward(X)
        loss = criterion.forward(logits, Y)
        dlogits = criterion.backward()
        net.backward(dlogits)
        # Check gradient norms for each layer that has grad_weights.
        grad_norms = []
        for i, layer in enumerate(net.layers):
            if hasattr(layer, 'grad_weights') and layer.grad_weights is not None:
                norm = LA.norm(layer.grad_weights)
                grad_norms.append(norm)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Gradient norms: {grad_norms}")
        net.step(lr, momentum=0.8, weight_decay=0.0, freeze_first=freeze_first)
    preds = (1.0/(1.0+np.exp(-net.forward(X))) > 0.5).astype(int).reshape(-1)
    accuracy = np.mean(preds == Y)
    print("Final predictions:", preds)
    print("Final accuracy:", accuracy*100, "%")

if __name__ == "__main__":
    # 先進行 CIFAR 的訓練與超參數掃描
    main()
    print("\nTraining XOR with frozen first layer:")
    train_xor(freeze_first=True, epochs=5000, lr=0.1)
    print("\nTraining XOR with all layers trainable and grad check:")
    train_xor_with_grad_check(freeze_first=False, epochs=5000, lr=0.1)
