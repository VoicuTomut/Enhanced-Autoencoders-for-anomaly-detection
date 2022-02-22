import time

import matplotlib.pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)

import pennylane as qml
import pennylane.numpy as np
import jax
import jax.numpy as jnp
import optax
import sklearn.datasets as datasets

np.random.seed(42)

num_trash_bits = 2
num_data_bits = 4
num_wires = num_trash_bits + num_data_bits
num_layers = 4
trash_bits_encoding = list(range(num_trash_bits))
data_bits_encoding = list(range(num_trash_bits, num_wires))

dev = qml.device('default.qubit.jax', wires=num_wires, shots=None)

digits = datasets.load_digits(n_class=2)
data = digits['data']
labels = digits['target']
digits_zero = data[labels == 0]
digits_one = data[labels == 1]
each_digit = 10
X_train = jnp.concatenate([digits_zero[:each_digit], digits_one[:each_digit]], axis=0)
y_train = jnp.array([[1] * each_digit, [2] * each_digit]).ravel()


def layer(theta, x, trash, data, pattern):
    qml.broadcast(qml.RY, wires=trash + data, pattern='single',
                  parameters=jnp.array([theta[0][j] * x[i] + theta[0][j + 1] for i, j in
                                        zip(range(num_wires), range(0, 2 * num_wires, 2))]))
    qml.broadcast(qml.CZ, wires=trash, pattern='all_to_all')
    qml.broadcast(qml.CZ, wires=trash + data, pattern=pattern[0])
    qml.broadcast(qml.RY, wires=trash + data, pattern='single',
                  parameters=jnp.array([theta[1][j] * x[i] + theta[1][j + 1] for i, j in
                                        zip(range(num_wires), range(0, 2 * num_wires, 2))]))
    qml.broadcast(qml.CZ, wires=trash, pattern='all_to_all')
    qml.broadcast(qml.CZ, wires=trash + data, pattern=pattern[1])


def get_pattern(trash, data):
    pattern = []
    for i in trash:
        for j in data:
            if i < j:
                pattern.append([i, j])
    pattern0 = pattern[:len(pattern) // 2]
    pattern1 = pattern[len(pattern) // 2:]
    new = []
    for i, j in zip(pattern1[::2], pattern1[1::2]):
        new.extend([j, i])
    pattern0_ = []
    for i, j in zip(pattern0[::2], new[::2]):
        pattern0_.extend([i, j])
    pattern1_ = []
    for i, j in zip(pattern0[1::2], new[1::2]):
        pattern1_.extend([i, j])
    if len(pattern0) % 2 != 0:
        pattern0_.append(pattern0[-1])
    if len(pattern1) % 2 != 0:
        pattern1_.append(pattern1[-1])
    pattern = [pattern0_, pattern1_]
    return pattern


pattern_encode = get_pattern(trash_bits_encoding, data_bits_encoding)


@jax.jit
@qml.qnode(dev, interface='jax')
def circuit(params, x, state):
    x = jnp.array([x] * num_wires)
    theta0 = params[:-num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1 = params[-num_trash_bits * 2:].reshape([1, -1])
    qml.AmplitudeEmbedding(state, wires=range(num_wires), normalize=True)
    U(theta0, x, trash_bits_encoding, data_bits_encoding, pattern_encode)
    qml.broadcast(qml.RY, wires=range(num_trash_bits), pattern='single',
                  parameters=jnp.array([theta1[0][j] * x[i] + theta1[0][j + 1] for i, j in
                                        zip(range(num_trash_bits), range(0, 2 * num_trash_bits, 2))]))
    # return qml.density_matrix(wires=range(num_trash_bits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_trash_bits)]


def U(theta0, x, trash, data, pattern):
    for l in range(num_layers):
        layer(theta0[l], x, trash, data, pattern)


num_weights = num_layers * num_wires * 2 * 2 + num_trash_bits * 2
weights = np.random.uniform(0, 2 * np.pi, num_weights, requires_grad=True)
weights = jnp.array(weights)
"""
fig, ax = qml.draw_mpl(circuit)(weights, y_train[0], X_train[0])
fig.show()
"""
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
    return jnp.sum(1 - exp_values) / 2


def total_cost(params, X, Y):
    return jnp.mean(jax.vmap(per_cost, in_axes=[None, 0, 0])(params, Y, X))


cost_pre = total_cost(weights, X_train, y_train)
print(cost_pre, 'cost before train')

steps = 1000


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        loss_value, grads = jax.value_and_grad(total_cost)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    costs = []
    for i in range(steps):
        subset = np.random.choice(list(range(len(X_train))), batch, replace=False)
        x_batch = X_train[subset]
        y_batch = y_train[subset]
        params, opt_state, loss_value = step(params, opt_state, x_batch, y_batch)
        costs.append(loss_value)

    return params, costs


batch = 3
optimizer = optax.adafactor(0.01)
t1 = time.time()
weights, costs = fit(weights, optimizer)
cost_post = total_cost(weights, X_train, y_train)
print(cost_post, 'cost after train')

plt.plot(costs)
plt.yscale('log')
plt.show()

print(time.time() - t1, 'seconds')


@jax.jit
@qml.qnode(dev, interface='jax')
def get_init_states(state):
    qml.AmplitudeEmbedding(state, wires=range(num_wires), normalize=True)
    return qml.density_matrix(wires=range(num_wires))


dev = qml.device('default.qubit.jax', wires=2 * num_trash_bits + num_data_bits)

trash_bits_decoding = list(range(num_trash_bits, 2 * num_trash_bits))
data_bits_decoding = [i + num_trash_bits for i in data_bits_encoding]

pattern_encode = get_pattern(trash_bits_encoding, data_bits_decoding)
pattern_decode = get_pattern(trash_bits_decoding, data_bits_decoding)


@jax.jit
@qml.qnode(dev, interface='jax')
def circuit_decoding(params, x, state):
    x = jnp.array([x] * num_wires)
    theta0 = params[:-num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1 = params[-num_trash_bits * 2:].reshape([1, -1])
    qml.AmplitudeEmbedding(state, wires=trash_bits_encoding + data_bits_decoding, normalize=True)
    U(theta0, x, trash_bits_encoding, data_bits_decoding, pattern_encode)
    qml.broadcast(qml.RY, wires=trash_bits_encoding, pattern='single',
                  parameters=jnp.array([theta1[0][j] * x[i] + theta1[0][j + 1] for i, j in
                                        zip(range(num_trash_bits), range(0, 2 * num_trash_bits, 2))]))
    qml.Barrier(wires=range(num_wires + num_trash_bits))
    qml.broadcast(qml.adjoint(qml.RY), wires=trash_bits_decoding, pattern='single',
                  parameters=jnp.array([theta1[0][j] * x[i] + theta1[0][j + 1] for i, j in
                                        zip(range(num_trash_bits), range(0, 2 * num_trash_bits, 2))]))
    qml.adjoint(U)(theta0, x, trash_bits_decoding, data_bits_decoding, pattern_decode)
    return qml.density_matrix(wires=trash_bits_decoding + data_bits_decoding)


def _funm_svd(matrix, func):
    import scipy.linalg as la

    unitary1, singular_values, unitary2 = la.svd(matrix)
    diag_func_singular = np.diag(func(singular_values))
    return unitary1.dot(diag_func_singular).dot(unitary2)


def check_fidelity(params, x, state):
    original = get_init_states(state)
    decoded = circuit_decoding(params, x, state)
    s1sq = _funm_svd(original, np.sqrt)
    s2sq = _funm_svd(decoded, np.sqrt)
    fid = np.linalg.norm(s1sq.dot(s2sq), ord="nuc") ** 2
    return float(np.real(fid))


def fid_score(params, X, Y):
    fids = []
    for state, x in zip(X, Y):
        fids.append(check_fidelity(params, x, state))
    return np.mean(np.array(fids))


print(fid_score(weights, X_train, y_train), 'average fidelity after decoding')
"""
fig, ax = qml.draw_mpl(circuit_decoding)(weights, y_train[0], X_train[0])
fig.show()
"""
