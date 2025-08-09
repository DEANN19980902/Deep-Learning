"""
INSERT YOUR NAME HERE
Yi Chiun Chang
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################

class SigmoidCrossEntropy:
    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels.reshape(-1, 1)
        # Compute the sigmoid output
        self.s = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        loss = - (self.labels * np.log(self.s + eps) + (1 - self.labels) * np.log(1 - self.s + eps))
        return np.mean(loss)
    
    def backward(self):
        n = self.labels.shape[0]
        grad_logits = (self.s - self.labels) / n
        return grad_logits

class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
      
    def backward(self, grad):
        grad_input = grad.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
    
    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size, momentum=0, weight_decay=0):
        pass

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize weights with small random values and biases with zeros
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        # Initialize momentum terms for weights and bias
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)
    
    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias
    
    def backward(self, grad):
        self.grad_weights = np.dot(self.input.T, grad)
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)
        grad_input = np.dot(grad, self.weights.T)
        return grad_input
    
    ######################################################
    # Q2 Implement SGD with Weight Decay
    ######################################################
    def step(self, step_size, momentum=0.8, weight_decay=0.0):
        # Update weights using momentum SGD with weight decay
        self.v_weights = momentum * self.v_weights - step_size * (self.grad_weights + weight_decay * self.weights)
        self.weights += self.v_weights
        self.v_bias = momentum * self.v_bias - step_size * (self.grad_bias + weight_decay * self.bias)
        self.bias += self.v_bias


# FeedForward Neural Network
class FeedForwardNeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            self.layers = []
            # First hidden layer: from input to hidden_dim
            self.layers.append(LinearLayer(input_dim, hidden_dim))
            self.layers.append(ReLU())
            # For multiple hidden layers, add (num_layers-2) pairs of Linear + ReLU layers
            for _ in range(num_layers - 2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(ReLU())
            # Last layer: from hidden layer to output layer
            self.layers.append(LinearLayer(hidden_dim, output_dim))
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def step(self, step_size, momentum, weight_decay):
        for layer in self.layers:
            layer.step(step_size, momentum, weight_decay)

######################################################
# Q4 Implement Evaluation for Monitoring Training
######################################################

def evaluate(model, X_val, Y_val, batch_size):
    loss_layer = SigmoidCrossEntropy()
    total_loss = 0
    correct = 0
    num_examples = X_val.shape[0]
    num_batches = int(np.ceil(num_examples / batch_size))
    
    for i in range(num_batches):
        start = i * batch_size
        end = min(num_examples, (i + 1) * batch_size)
        X_batch = X_val[start:end]
        Y_batch = Y_val[start:end]
        
        logits = model.forward(X_batch)
        loss = loss_layer.forward(logits, Y_batch)
        total_loss += loss * (end - start)
        
        preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
        correct += np.sum(preds.flatten() == Y_batch.flatten())
    
    avg_loss = total_loss / num_examples
    accuracy = correct / num_examples
    return avg_loss, accuracy


# Training Function for a Single Run (records batch and epoch data)
def train_single_run(X_train, Y_train, X_test, Y_test, input_dim, batch_size, step_size, hidden_units, max_epochs, momentum=0.8, weight_decay=1e-4):
    output_dim = 1
    net = FeedForwardNeuralNetwork(input_dim, output_dim, hidden_units, num_layers=2)
    loss_layer = SigmoidCrossEntropy()
    
    losses = []     # Train loss for each batch
    accs = []       # Train accuracy for each batch
    epoch_x = []    # Cumulative batch count at the end of each epoch (for x-axis plotting)
    val_losses = [] # Test loss at each epoch
    val_accs = []   # Test accuracy at each epoch
    
    num_examples = X_train.shape[0]
    batch_counter = 0
    
    for epoch in range(max_epochs):
        perm = np.random.permutation(num_examples)
        X_train_epoch = X_train[perm]
        Y_train_epoch = Y_train[perm]
        num_batches = int(np.ceil(num_examples / batch_size))
        
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(num_examples, (i + 1) * batch_size)
            X_batch = X_train_epoch[start:end]
            Y_batch = Y_train_epoch[start:end]
            
            logits = net.forward(X_batch)
            loss = loss_layer.forward(logits, Y_batch)
            losses.append(loss)
            
            preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
            batch_correct = np.sum(preds.flatten() == Y_batch.flatten())
            epoch_train_loss += loss * (end - start)
            epoch_train_correct += batch_correct
            
            batch_acc = batch_correct / (end - start)
            accs.append(batch_acc)
            
            grad_loss = loss_layer.backward()
            net.backward(grad_loss)
            net.step(step_size, momentum, weight_decay)
            
            batch_counter += 1
        
        epoch_avg_loss = epoch_train_loss / num_examples
        epoch_avg_acc = epoch_train_correct / num_examples
        
        # Evaluate on the test set at the end of each epoch
        epoch_x.append(batch_counter)
        test_loss, test_acc = evaluate(net, X_test, Y_test, batch_size)
        val_losses.append(test_loss)
        val_accs.append(test_acc)

        logging.info("Epoch %d: Train Loss=%.4f, Train Acc=%.4f, Test Loss=%.4f, Test Acc=%.4f",
                     epoch+1, epoch_avg_loss, epoch_avg_acc, test_loss, test_acc)
    
    return losses, accs, epoch_x, val_losses, val_accs


# Hyperparameter Sweep Function
def sweep_parameters(X_train, Y_train, X_test, Y_test, input_dim):
    # Sweep over Batch Size
    candidate_batch_sizes = [32, 64, 128, 256]
    acc_list_bs = []
    for bs in candidate_batch_sizes:
        # Using fixed learning rate and hidden units; train for 10 epochs
        _, _, _, _, test_accs = train_single_run(X_train, Y_train, X_test, Y_test, input_dim, bs, 0.01, 100, max_epochs=10)
        acc_list_bs.append(test_accs[-1])
        print(f"Batch Size {bs}: Final Test Acc = {test_accs[-1]:.4f}")
    sweep_bs = (candidate_batch_sizes, acc_list_bs)
    
    # Sweep over Learning Rate
    candidate_learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    acc_list_lr = []
    for lr in candidate_learning_rates:
        _, _, _, _, test_accs = train_single_run(X_train, Y_train, X_test, Y_test, input_dim, 128, lr, 100, max_epochs=10)
        acc_list_lr.append(test_accs[-1])
        print(f"Learning Rate {lr}: Final Test Acc = {test_accs[-1]:.4f}")
    sweep_lr = (candidate_learning_rates, acc_list_lr)
    
    # Sweep over Hidden Units
    candidate_hidden_units = [50, 100, 200, 400]
    acc_list_hd = []
    for hu in candidate_hidden_units:
        _, _, _, _, test_accs = train_single_run(X_train, Y_train, X_test, Y_test, input_dim, 128, 0.01, hu, max_epochs=10)
        acc_list_hd.append(test_accs[-1])
        print(f"Hidden Units {hu}: Final Test Acc = {test_accs[-1]:.4f}")
    sweep_hd = (candidate_hidden_units, acc_list_hd)
    
    return sweep_bs, sweep_lr, sweep_hd

######################################################
# Main Function: Load data, run training and hyperparameter sweep, and plot 4 graphs together
######################################################

def main():
    # Load data (ensure the path to 'cifar_2class_py3.p' is correct)
    data = pickle.load(open('cifar_2class_py3.p', 'rb'))
    X_train = data['train_data'].astype(np.float32) / 255.0
    Y_train = data['train_labels']
    X_test = data['test_data'].astype(np.float32) / 255.0
    Y_test = data['test_labels']
    num_examples, input_dim = X_train.shape

    # Run hyperparameter sweep experiments
    sweep_bs, sweep_lr, sweep_hd = sweep_parameters(X_train, Y_train, X_test, Y_test, input_dim)
    
    # Single training run: record batch and epoch data 
    losses, accs, epoch_x, val_losses, val_accs = train_single_run(X_train, Y_train, X_test, Y_test, input_dim, 128, 0.001, 100, max_epochs=50)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Subplot 1: Training Curves (Single Run)
    ax1 = axs[0, 0]
    color_loss = 'tab:red'
    ax1.set_xlabel("Iterations (per batch)")
    ax1.set_ylabel("Train Loss", color=color_loss)
    ax1.plot(range(len(losses)), losses, c=color_loss, alpha=0.5, marker='.', label="Train Loss")
    ax1.plot(epoch_x, val_losses, 'o-', c='darkred', label="Test Loss")
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.legend(loc="upper left")
    ax1.set_title("Training Curves (Single Run)")
    
    ax1b = ax1.twinx()
    color_acc = 'tab:blue'
    ax1b.set_ylabel("Accuracy", color=color_acc)
    ax1b.plot(range(len(accs)), accs, c=color_acc, alpha=0.5, marker='.', label="Train Acc")
    ax1b.plot(epoch_x, val_accs, 'o-', c='blue', label="Test Acc")
    ax1b.tick_params(axis='y', labelcolor=color_acc)
    ax1b.set_ylim([0, 1.0])
    ax1b.legend(loc="upper right")
    best_epoch_idx = np.argmax(val_accs)
    best_epoch = best_epoch_idx + 1
    best_test_acc = val_accs[best_epoch_idx]
    ax1b.annotate(f"Best Epoch {best_epoch}\nTest Acc: {best_test_acc:.2f}",
                  xy=(epoch_x[best_epoch_idx], val_accs[best_epoch_idx]),
                  xytext=(epoch_x[best_epoch_idx], val_accs[best_epoch_idx] + 0.05),
                  arrowprops=dict(facecolor='blue', shrink=0.02, width=1, headwidth=6),
                  fontsize=10, color='blue')
    
    # Subplot 2: Test Accuracy vs. Batch Size
    ax2 = axs[0, 1]
    batch_list, acc_list_bs = sweep_bs
    ax2.plot(batch_list, acc_list_bs, marker='o')
    ax2.set_title("Test Accuracy vs. Batch Size")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Test Accuracy")
    ax2.grid(True)
    best_bs_idx = np.argmax(acc_list_bs)
    best_bs = batch_list[best_bs_idx]
    best_bs_acc = acc_list_bs[best_bs_idx]
    ax2.annotate(f"Best: {best_bs}\nAcc: {best_bs_acc:.2f}",
                 xy=(best_bs, best_bs_acc),
                 xytext=(best_bs, best_bs_acc + 0.03),
                 arrowprops=dict(facecolor='black', shrink=0.02, width=1, headwidth=6),
                 fontsize=10, color='black')
    
    # Subplot 3: Test Accuracy vs. Learning Rate
    ax3 = axs[1, 0]
    lr_list, acc_list_lr = sweep_lr
    ax3.plot(lr_list, acc_list_lr, marker='o')
    ax3.set_title("Test Accuracy vs. Learning Rate")
    ax3.set_xlabel("Learning Rate")
    ax3.set_ylabel("Test Accuracy")
    ax3.grid(True)
    best_lr_idx = np.argmax(acc_list_lr)
    best_lr = lr_list[best_lr_idx]
    best_lr_acc = acc_list_lr[best_lr_idx]
    ax3.annotate(f"Best: {best_lr}\nAcc: {best_lr_acc:.2f}",
                 xy=(best_lr, best_lr_acc),
                 xytext=(best_lr, best_lr_acc + 0.03),
                 arrowprops=dict(facecolor='black', shrink=0.02, width=1, headwidth=6),
                 fontsize=10, color='black')
    
    # Subplot 4: Test Accuracy vs. Hidden Units
    ax4 = axs[1, 1]
    hidden_list, acc_list_hd = sweep_hd
    ax4.plot(hidden_list, acc_list_hd, marker='o')
    ax4.set_title("Test Accuracy vs. Hidden Units")
    ax4.set_xlabel("Hidden Units")
    ax4.set_ylabel("Test Accuracy")
    ax4.grid(True)
    best_hd_idx = np.argmax(acc_list_hd)
    best_hd = hidden_list[best_hd_idx]
    best_hd_acc = acc_list_hd[best_hd_idx]
    ax4.annotate(f"Best: {best_hd}\nAcc: {best_hd_acc:.2f}",
                 xy=(best_hd, best_hd_acc),
                 xytext=(best_hd, best_hd_acc + 0.03),
                 arrowprops=dict(facecolor='black', shrink=0.02, width=1, headwidth=6),
                 fontsize=10, color='black')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
