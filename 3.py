import numpy as np
# from sklearn.preprocessing import normalize


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def init_two_layer_model(input_size, hidden_size, output_size):
    """Initialize the weights and biases for a two-layer fully connected neural
    network. The net has an input dimension of D, a hidden layer dimension of H,
    and performs classification over C classes. Weights are initialized to small
    random values and biases are initialized to zero.

    Inputs:
    - input_size: The dimension D of the input data
    - hidden_size: The number of neurons H in the hidden layer
    - ouput_size: The number of classes C

    Returns:
    A dictionary mapping parameter names to arrays of parameter values. It has
    the following keys:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)

    """
    # initialize a model
    model = {}
    model['W1'] = np.random.uniform(-0.05, 0.05, (input_size, hidden_size))
    model['b1'] = np.zeros(hidden_size)
    model['W2'] = np.random.uniform(-0.05, 0.05, (hidden_size, output_size))
    model['b2'] = np.zeros(output_size)
    return model


def two_layer_net(X, model, y=None, reg=0.0):
    """Compute the loss and gradients for a two layer fully connected neural
    network. The net has an input dimension of D, a hidden layer dimension of H,
    and performs classification over C classes. We use a softmax loss function
    and L2 regularization the the weight matrices. The two layer net should use a
    ReLU nonlinearity after the first affine layer.

    The two layer net has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - model: Dictionary mapping parameter names to arrays of parameter values.
      It should contain the following:
      - W1: First layer weights; has shape (D, H)
      - b1: First layer biases; has shape (H,)
      - W2: Second layer weights; has shape (H, C)
      - b2: Second layer biases; has shape (C,)
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
    is the score for class c on input X[i].

    If y is not passed, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function. This should have the same keys as model.

    """

    # Unlike the original assignment, this function uses sigmoid activation instead of ReLU.

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    N, D = X.shape
    H, C = W2.shape

    scores = None

    H_in = np.dot(X, W1) + b1
    # H_in[H_in < 0] = 0
    # H_out = H_in
    H_out = sigmoid(H_in)
    scores = np.dot(H_out, W2) + b2

    if y is None:
        return scores

    loss = None
    scores_hat = scores - np.max(scores, axis=1).reshape(-1, 1)
    softmax_res = np.exp(scores_hat) / np.sum(np.exp(scores_hat), axis=1).reshape(-1, 1)
    loss = -np.mean(np.log(softmax_res[range(N), list(y)]))

    loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    grads = {}

    delta_scores = softmax_res.copy()
    delta_scores[range(N), list(y)] -= 1

    grads['W2'] = H_out.T.dot(delta_scores.reshape(N, -1)) + reg * W2
    grads['W2'] /= N
    grads['b2'] = np.mean(delta_scores, axis=0)

    delta_H = delta_scores.dot(W2.T) * H_out * (1 - H_out)

    grads['W1'] = X.T.dot(delta_H) + reg * W1
    grads['W1'] /= N
    grads['b1'] = np.mean(delta_H, axis=0)

    # grads['b2'] = np.mean(grads['b2'], axis=0)
    # grads['W2'] = np.mean(grads['W2'], axis=0) + W2 * reg
    # grads['b1'] = np.mean(grads['b1'], axis=0)
    # grads['W1'] = np.mean(grads['W1'], axis=0) + W1 * reg

    return loss, grads


def train(X, y=None, reg=0.0, lr=0.1, epoch=1000, batchsize=5, gamma=1.0, stepsize=np.array([0])):
    stage = 0
    model = init_two_layer_model(400, 100, 10)
    iter = int(X.shape[0] / batchsize)
    for i in range(epoch):
        for j in range(iter):
            loss, grads = two_layer_net(X[j * batchsize:(j + 1) * batchsize, ...],
                                        model, y[j * batchsize:(j + 1) * batchsize], reg)
            model['b1'] -= grads['b1'] * lr
            model['W1'] -= grads['W1'] * lr
            model['b2'] -= grads['b2'] * lr
            model['W2'] -= grads['W2'] * lr
        if stage < stepsize.size and i == stepsize[stage]:
            lr *= gamma
            stage += 1
            # print("Epoch " + str(i) + ": Loss: " + str(loss))
    return model


def predict(X, model):
    score = two_layer_net(X, model[0])
    score += two_layer_net(X, model[1])
    score += two_layer_net(X, model[2])
    score += two_layer_net(X, model[3])
    score += two_layer_net(X, model[4])
    y_hat = np.argmax(score, axis=1)
    # print(y_hat.shape)
    return y_hat


def prepare_data():
    y = np.genfromtxt('train_targets.csv', delimiter=',', dtype=int)
    X = np.genfromtxt('train_data.csv', delimiter=',')
    test = np.genfromtxt('test_data.csv', delimiter=',')
    return X, y, test


def output_res(y_hat):
    with open('test_predictions.csv', 'w') as file:
        for i in range(y_hat.size):
            file.write(str(y_hat[i]) + "\r\n")


X, y, test = prepare_data()
# X = normalize(X)
# test = normalize(test)
model_0 = train(X, y, reg=5e-5, lr=0.5, epoch=50)
model_1 = train(X, y, reg=5e-5, lr=1.0, epoch=50)
model_2 = train(X, y, reg=5e-5, lr=0.3, epoch=50)
model_3 = train(X, y, reg=5e-5, lr=0.5, epoch=60)
model_4 = train(X, y, reg=1e-5, lr=0.5, epoch=30)
y_hat = predict(test, [model_0, model_1, model_2, model_3, model_4])
output_res(y_hat)
