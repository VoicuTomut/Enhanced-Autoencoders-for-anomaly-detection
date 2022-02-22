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

# Parameters
num_trash_bits = 5
num_data_bits = 1
num_entangler_bits = 0
num_layers = 4
train_digits_each = 10
test_digits_each = 30
steps = 1000
batch = 3
lr = 0.01

num_wires = num_trash_bits + num_data_bits
trash_bits_encoding = list(range(num_trash_bits))
data_bits_encoding = list(range(num_trash_bits, num_wires))
entangler_bits_encoding = list(range(num_wires, num_wires + num_entangler_bits))

digits = datasets.load_digits(n_class=2)
data = digits["data"]
labels = digits["target"]
digits_zero = data[labels == 0]
digits_one = data[labels == 1]

X_train = jnp.concatenate(
    [digits_zero[:train_digits_each], digits_one[:train_digits_each]], axis=0
)
y_train = jnp.array([[1] * train_digits_each, [2] * train_digits_each]).ravel()
X_test = jnp.concatenate(
    [
        digits_zero[train_digits_each: train_digits_each + test_digits_each],
        digits_one[train_digits_each: train_digits_each + test_digits_each],
    ],
    axis=0,
)
y_test = jnp.array([[1] * test_digits_each, [2] * test_digits_each]).ravel()

num_weights = 2 * (
        num_layers * (num_wires + num_entangler_bits // 2) * 2 * 2 + num_trash_bits * 2
)
weights = np.random.uniform(0, 2 * np.pi, num_weights, requires_grad=True)
weights = jnp.array(weights)


def layer(theta, x, trash, data, entangler, pattern):
    qml.broadcast(
        qml.RY,
        wires=trash + data + entangler,
        pattern="single",
        parameters=jnp.array(
            [
                theta[0][j] * x[i] + theta[0][j + 1]
                for i, j in zip(
                range(num_wires + num_entangler_bits // 2),
                range(0, 2 * (num_wires + num_entangler_bits // 2), 2),
            )
            ]
        ),
    )
    qml.broadcast(qml.CZ, wires=trash, pattern="all_to_all")
    qml.broadcast(qml.CZ, wires=trash + data + entangler, pattern=pattern[0])
    qml.broadcast(
        qml.RY,
        wires=trash + data + entangler,
        pattern="single",
        parameters=jnp.array(
            [
                theta[1][j] * x[i] + theta[1][j + 1]
                for i, j in zip(
                range(num_wires + num_entangler_bits // 2),
                range(0, 2 * (num_wires + num_entangler_bits // 2), 2),
            )
            ]
        ),
    )
    qml.broadcast(qml.CZ, wires=trash, pattern="all_to_all")
    qml.broadcast(qml.CZ, wires=trash + data + entangler, pattern=pattern[1])


def make_entangled_bits(entangler):
    for i, j in zip(entangler[::2], entangler[1::2]):
        qml.Hadamard(i)
        qml.CNOT(wires=[i, j])


def get_pattern(trash, data):
    pattern = []
    for i in trash:
        for j in data:
            if i < j:
                pattern.append([i, j])
    pattern0 = pattern[: len(pattern) // 2]
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


def U(theta0, x, trash, data, entangler, pattern):
    for l in range(num_layers):
        layer(theta0[l], x, trash, data, entangler, pattern)


dev1 = qml.device("default.qubit.jax", wires=num_wires, shots=None)


@jax.jit
@qml.qnode(dev1, interface="jax")
def get_init_states(state):
    qml.AmplitudeEmbedding(state, wires=range(num_wires), normalize=True)
    return qml.state()


dev2 = qml.device(
    "default.qubit.jax",
    wires=2 * num_trash_bits + num_data_bits + num_entangler_bits,
    shots=None,
)

trash_bits_decoding = list(range(num_trash_bits, 2 * num_trash_bits))
data_bits_decoding = [i + num_trash_bits for i in data_bits_encoding]
entangler_bits_decoding = [i + num_trash_bits for i in entangler_bits_encoding]
pattern_encode = get_pattern(
    trash_bits_encoding, data_bits_decoding + entangler_bits_decoding[::2]
)
pattern_decode = get_pattern(
    trash_bits_decoding, data_bits_decoding + entangler_bits_decoding[1::2]
)


@jax.jit
@qml.qnode(dev2, interface="jax")
def circuit_decoding(params, x, state):
    x = jnp.array([x] * num_wires)
    params0 = params[: num_weights // 2]
    params1 = params[num_weights // 2:]
    theta0_encode = params0[: -num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1_encode = params0[-num_trash_bits * 2:].reshape([1, -1])
    theta0_decode = params1[: -num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1_decode = params1[-num_trash_bits * 2:].reshape([1, -1])

    qml.AmplitudeEmbedding(
        state, wires=trash_bits_encoding + data_bits_decoding, normalize=True
    )

    make_entangled_bits(entangler_bits_decoding)

    U(
        theta0_encode,
        x,
        trash_bits_encoding,
        data_bits_decoding,
        entangler_bits_decoding[::2],
        pattern_encode,
    )
    qml.broadcast(
        qml.RY,
        wires=trash_bits_encoding,
        pattern="single",
        parameters=jnp.array(
            [
                theta1_encode[0][j] * x[i] + theta1_encode[0][j + 1]
                for i, j in zip(range(num_trash_bits), range(0, 2 * num_trash_bits, 2))
            ]
        ),
    )

    qml.Barrier(wires=range(num_wires + num_trash_bits + num_entangler_bits))

    U(
        theta0_decode,
        x,
        trash_bits_decoding,
        data_bits_decoding,
        entangler_bits_decoding[1::2],
        pattern_decode,
    )
    qml.broadcast(
        qml.RY,
        wires=trash_bits_decoding,
        pattern="single",
        parameters=jnp.array(
            [
                theta1_decode[0][j] * x[i] + theta1_decode[0][j + 1]
                for i, j in zip(range(num_trash_bits), range(0, 2 * num_trash_bits, 2))
            ]
        ),
    )
    return qml.density_matrix(wires=trash_bits_decoding + data_bits_decoding)


def check_fidelity(params, x, state):
    original = get_init_states(state)
    decoded = circuit_decoding(params, x, state)
    fid = jnp.dot(jnp.conj(original), jnp.dot(decoded, original))
    return jnp.real(fid)


def fid_score(params, X, Y):
    return -jnp.mean(jax.vmap(check_fidelity, in_axes=[None, 0, 0])(params, Y, X))


cost_pre = fid_score(weights, X_train, y_train)
print(-cost_pre, "fidelity before train - train data")
print(-fid_score(weights, X_test, y_test), "fidelity before train - test data")


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        loss_value, grads = jax.value_and_grad(fid_score)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    costs_train = []
    costs_test = []
    for i in range(steps):
        subset = np.random.choice(list(range(len(X_train))), batch, replace=False)
        x_batch = X_train[subset]
        y_batch = y_train[subset]
        params, opt_state, loss_value = step(params, opt_state, x_batch, y_batch)
        costs_train.append(-loss_value)
        # costs_test.append(-fid_score(params, X_test, y_test))

    return params, costs_train, costs_test


optimizer = optax.adam(lr)
t1 = time.time()
weights, costs_train, costs_test = fit(weights, optimizer)

# plt.plot(costs_test)
plt.plot(costs_train)
# plt.legend(['Train', 'Test'])
# plt.yscale('log')
plt.show()

print(time.time() - t1, "seconds - training time")


def get_fid_scores(params, X, Y):
    return jax.vmap(check_fidelity, in_axes=[None, 0, 0])(params, Y, X)


fid_scores_train = get_fid_scores(weights, X_train, y_train)
fid_scores_test = get_fid_scores(weights, X_test, y_test)
print(jnp.mean(fid_scores_train), "average fidelity after decoding - train data")
print(jnp.mean(fid_scores_test), "average fidelity after decoding - test data")

plt.scatter(y=fid_scores_train[:train_digits_each], x=range(train_digits_each), marker='*', c='r')
plt.scatter(y=fid_scores_train[train_digits_each:], x=range(train_digits_each), marker='+', c='r')
plt.scatter(y=fid_scores_test[:test_digits_each], x=range(test_digits_each), marker='*', c='b')
plt.scatter(y=fid_scores_test[test_digits_each:], x=range(test_digits_each), marker='+', c='b')
plt.legend(['Train 0', 'Train 1', 'Test 0', 'Test 1'])
plt.show()

"""
fig, ax = qml.draw_mpl(circuit_decoding)(weights, y_train[0], X_train[0])
fig.show()
"""
