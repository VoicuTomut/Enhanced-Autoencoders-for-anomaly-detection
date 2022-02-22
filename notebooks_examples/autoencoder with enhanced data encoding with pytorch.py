import time
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
import sklearn.datasets as datasets
import torch

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)

num_trash_bits = 2
num_data_bits = 4
num_wires = num_trash_bits + num_data_bits
num_layers = 4
trash_bits_encoding = list(range(num_trash_bits))
data_bits_encoding = list(range(num_trash_bits, num_wires))

dev = qml.device('default.qubit', wires=num_wires)

digits = datasets.load_digits(n_class=2)
data = digits['data']
labels = digits['target']
digits_zero = data[labels == 0]
digits_one = data[labels == 1]
each_digit = 10
X_train = np.concatenate([digits_zero[:each_digit], digits_one[:each_digit]], axis=0)
y_train = np.array([[0] * each_digit, [1] * each_digit]).ravel()
y_ = []
for i in y_train:
    y_.append([i] * 6)
X_train, y_train = torch.tensor(X_train, requires_grad=False, device=device_torch), torch.tensor(np.array(y_),
                                                                                                 requires_grad=False,
                                                                                                 device=device_torch)


def layer(theta, x, trash, data):
    t0 = theta[0][::2]
    t1 = theta[0][1::2]
    p0 = t0 * x + t1
    qml.broadcast(qml.RY, wires=trash + data, pattern='single',
                  parameters=p0)
    qml.broadcast(qml.CZ, wires=trash, pattern='all_to_all')
    qml.broadcast(qml.CZ, wires=trash + data, pattern=pattern[0::2])
    t0 = theta[1][::2]
    t1 = theta[1][1::2]
    p1 = t0 * x + t1
    qml.broadcast(qml.RY, wires=trash + data, pattern='single',
                  parameters=p1)
    qml.broadcast(qml.CZ, wires=trash, pattern='all_to_all')
    qml.broadcast(qml.CZ, wires=trash + data, pattern=pattern[1::2])


pattern = []
for i in range(num_trash_bits):
    for j in range(num_trash_bits, num_wires):
        if i < j:
            pattern.append([i, j])


@qml.qnode(dev, interface='torch')
def circuit(params, x, state):
    # x = torch.tensor([x] * num_wires, requires_grad=False, device=device_torch)
    theta0 = params[:-num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1 = params[-num_trash_bits * 2:].reshape([1, -1])
    qml.AmplitudeEmbedding(state, wires=range(num_wires), normalize=True)
    U(theta0, x, trash_bits_encoding, data_bits_encoding)
    t0 = theta1[0][::2]
    t1 = theta1[0][1::2]
    p0 = t0 * x[:2] + t1
    qml.broadcast(qml.RY, wires=range(num_trash_bits), pattern='single',
                  parameters=p0)
    # return qml.density_matrix(wires=range(num_trash_bits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_trash_bits)]


def U(theta0, x, trash, data):
    for l in range(num_layers):
        layer(theta0[l], x, trash, data)


num_weights = num_layers * num_wires * 2 * 2 + num_trash_bits * 2
weights = np.random.uniform(0, 2 * np.pi, num_weights, requires_grad=True)
weights = torch.tensor(weights, requires_grad=True, device=device_torch)
weights_old = torch.clone(weights)
"""
def per_cost(params, x, state):
    matrix = circuit(params, x, state)
    real = jnp.real(matrix)
    imag = jnp.imag(matrix)
    err = 1 - real[0][0]
    err = err + sum(jnp.abs(jnp.ravel(real)[1:]))
    err = err + sum(jnp.abs(jnp.ravel(imag)))
    return err
"""


def per_cost(params, x, state):
    exp_values = circuit(params, x, state)
    return torch.sum(1 - exp_values) / 2


def total_cost(params, X, Y):
    cost = 0
    for state, x in zip(X, Y):
        cost = cost + per_cost(params, x, state)
    cost = (cost / len(X)).clone().detach().requires_grad_(True)
    return cost


cost_pre = total_cost(weights, X_train, y_train)
print(cost_pre)

batch = 3

t1 = time.time()

opt = torch.optim.LBFGS([weights], lr=0.1)

steps = 3
for i in range(steps):
    subset = torch.tensor(np.random.choice(list(range(len(X_train))), batch, replace=False))
    x_batch = X_train[subset]
    y_batch = y_train[subset]


    def closure():
        opt.zero_grad()
        loss = total_cost(weights, x_batch, y_batch)
        loss.backward()
        print(weights.grad)
        return loss


    opt.step(closure)

weights_final = opt.param_groups[0]['params'][0]
cost_post = total_cost(weights_final, X_train, y_train)
print(cost_post)

print(time.time() - t1, 'seconds')
