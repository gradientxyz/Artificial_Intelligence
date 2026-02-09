import numpy as np

# def sigmoid(z):
#     return 1. / (1. + np.exp(-z))

def sigmoid(x):
    result = np.zeros_like(x)
    pos_mask = (x >= 0)
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    neg_mask = (x < 0)
    result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
    return result


def sigmoid_derivative(x):
    return x * (1. - x)


def softmax(z):
    exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


# def int_to_onehot(y, num_labels):
#     ary = np.zeros((y.shape[0], num_labels))
#     ary[np.arange(y.shape[0]), y] = 1
#     return ary


def batch_generator(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - batch_size + 1, 
                           batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]


def loss_crossEntropy(logits, y):
    # cross entropy loss
    # logits: [num_examples, num_classes]
    m = y.shape[0]
    p = softmax(logits)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_crossEntropy(logits, y):
    m = y.shape[0]
    grad = softmax(logits)
    # dL/dout = p_i - y_i
    grad[range(m), y] -= 1
    grad = grad/m
    return grad


def accuracy(logits, targets):
    predicted = np.argmax(logits, axis=-1)
    return np.mean(predicted == targets)


### @TODO
### Implement Adam optimizer from scratch


class NeuralNetMLP:
    def __init__(self, num_features, num_hidden,
                 num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes

        # hidden weight init
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output weight init
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # Hidden layer
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        # Output layer
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        # a_h is needed for backpropagation
        return a_h, z_out
    
    def backward(self, x, a_h, z_out, y):
        # Output layer
        # delta_z_out: [num_examples, num_classes]
        # a_h        : [num_examples, num_hidden]
        # delta_w_out: [num_classes, num_hidden]
        # weight_out : [num_classes, num_hidden]
        delta_z_out = delta_crossEntropy(z_out, y)
        delta_w_out = np.dot(delta_z_out.T, a_h)
        delta_bias_out = np.sum(delta_z_out, axis=0)
        delta_a_h = np.dot(delta_z_out, self.weight_out)
        # Hidden layer:
        delta_z_h = delta_a_h * sigmoid_derivative(a_h)
        delta_w_h = np.dot(delta_a_h.T, x)
        delta_bias_h = np.sum(delta_z_h, axis=0)
        return delta_w_out, delta_bias_out, delta_w_h, delta_bias_h
    
    def predict(self, X):
        _, logits = self.forward(X)
        return np.argmax(logits, axis=-1)

    
def train(model, X_train, y_train, X_valid, y_valid, num_epochs=20,
          batch_size=100, learning_rate=0.1):
    train_loss_lst = []
    train_acc_lst = []
    valid_loss_lst = []
    valid_acc_lst = []

    for e in range(num_epochs):
        batch_gen = batch_generator(
            X_train, y_train, batch_size)
        train_loss, is_correct = 0, 0
        for X_batch, y_batch in batch_gen:
            hidden, logits = model.forward(X_batch)
            delta_w_out, delta_bias_out, delta_w_h, delta_bias_h = \
                model.backward(X_batch, hidden, logits, y_batch)
            ## update weight (SGD optimization)
            model.weight_h -= learning_rate * delta_w_h
            model.bias_h -= learning_rate * delta_bias_h
            model.weight_out -= learning_rate * delta_w_out
            model.bias_out -= learning_rate * delta_bias_out
            train_loss += loss_crossEntropy(logits, y_batch) * y_batch.shape[0]
            is_correct += accuracy(logits, y_batch) * y_batch.shape[0]
        train_loss = train_loss / y_train.shape[0]
        train_acc = is_correct / y_train.shape[0]
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)

        _, valid_logits = model.forward(X_valid)
        valid_loss = loss_crossEntropy(valid_logits, y_valid)
        valid_acc = accuracy(valid_logits, y_valid)
        valid_loss_lst.append(valid_loss)
        valid_acc_lst.append(valid_acc)
        if e % 5 == 0:
            print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
                f'| Train loss: {train_loss:.3f} '
                f'| Train acc: {train_acc:.3f} '
                f'| Valid acc: {valid_acc}')
    return train_loss_lst, train_acc_lst, valid_loss_lst, valid_acc_lst

    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    X, y = fetch_openml('mnist_784',
                        version=1,
                        as_frame=False,
                        cache=True,
                        return_X_y=True)
    X = ((X.astype(np.float32) / 255.) - .5) * 2
    y = y.astype(np.int8)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=10000, random_state=123, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=5000,
        random_state=123, stratify=y_temp
    )

    model = NeuralNetMLP(num_features=784,
                         num_hidden=100,
                         num_classes=10)
    train_loss_lst, train_acc_lst, valid_loss_lst, valid_acc_lst = train(
        model, X_train, y_train, X_valid, y_valid, 
        num_epochs=20, batch_size=64, learning_rate=0.1)
    
    plt.plot(range(len(train_loss_lst)), train_loss_lst)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
    plt.plot(range(len(train_acc_lst)), train_acc_lst,
             label='Training')
    plt.plot(range(len(valid_acc_lst)), valid_acc_lst,
             label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.show()

    _, test_logits = model.forward(X_test)
    test_acc = accuracy(test_logits, y_test)
    print(f'Test accuracy: {test_acc:.3f}')


