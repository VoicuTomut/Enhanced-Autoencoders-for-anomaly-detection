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

seed = 42
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

# Parameters
num_trash_bits = 5
num_data_bits = 1
num_entangler_bits = 0
num_layers = 1
train_digits_each = 50
test_digits_each = 100
batch = 10
epochs = 100
lr = 0.01

PARAMETERS = {
    "Trash Bits": num_trash_bits,
    "Data Bits": num_data_bits,
    "EPR Pairs": num_entangler_bits // 2,
    "Layers": num_layers,
    "Training Digits Each": train_digits_each,
    "Testing Digits Each": test_digits_each,
    "Training Epochs": epochs,
    "Batch Size": batch,
    "Learning Rate": lr,
}

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

num_weights = (
        num_layers * (num_wires + num_entangler_bits // 2) * 2 * 2 + num_trash_bits * 2
)

weights_encoder = np.random.uniform(-np.pi, np.pi, num_weights, requires_grad=True)
weights_encoder = jnp.array(weights_encoder)

weights_decoder = np.random.uniform(-np.pi, np.pi, num_weights, requires_grad=True)
weights_decoder = jnp.array(weights_decoder)


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


def iterate_batches(X, Y, batch_size):
    global key
    key, subkey = jax.random.split(key)
    X = jax.random.shuffle(subkey, X)
    Y = jax.random.shuffle(subkey, Y)
    batch_list_x = []
    batch_x = []
    for x in X:
        if len(batch_x) < batch_size:
            batch_x.append(x)

        else:
            batch_list_x.append(batch_x)
            batch_x = []
    if len(batch_x) != 0:
        batch_list_x.append(batch_x)
    batch_list_y = []
    batch_y = []
    for y in Y:
        if len(batch_y) < batch_size:
            batch_y.append(y)

        else:
            batch_list_y.append(batch_y)
            batch_y = []
    if len(batch_y) != 0:
        batch_list_y.append(batch_y)
    return batch_list_x, batch_list_y


pattern_encode_0 = get_pattern(
    trash_bits_encoding, data_bits_encoding + entangler_bits_encoding[::2]
)

dev0 = qml.device("default.qubit.jax", wires=num_wires + num_entangler_bits, shots=None)
zero_state = jnp.zeros([2 ** num_trash_bits])
zero_state = zero_state.at[0].set(1)


@jax.jit
@qml.qnode(dev0, interface="jax")
def circuit_encoding(params, x, state):
    x = jnp.array([x] * num_wires)
    theta0 = params[: -num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1 = params[-num_trash_bits * 2:].reshape([1, -1])

    qml.AmplitudeEmbedding(state, wires=range(num_wires), normalize=True)

    make_entangled_bits(entangler_bits_encoding)

    U(
        theta0,
        x,
        trash_bits_encoding,
        data_bits_encoding,
        entangler_bits_encoding[::2],
        pattern_encode_0,
    )

    qml.broadcast(
        qml.RY,
        wires=range(num_trash_bits),
        pattern="single",
        parameters=jnp.array(
            [
                theta1[0][j] * x[i] + theta1[0][j + 1]
                for i, j in zip(range(num_trash_bits), range(0, 2 * num_trash_bits, 2))
            ]
        ),
    )
    return qml.density_matrix(wires=range(num_trash_bits))


def encoding_fidelity(params, x, state):
    trash = circuit_encoding(params, x, state)
    fid = jnp.dot(jnp.conj(zero_state), jnp.dot(trash, zero_state))
    return jnp.real(fid)


def cost_encoding(params, X, Y):
    return -jnp.mean(jax.vmap(encoding_fidelity, in_axes=[None, 0, 0])(params, Y, X))


train_fid_hist = []
test_fid_hist = []


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        loss_value, grads = jax.value_and_grad(cost_encoding)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        x_batches, y_batches = iterate_batches(X_train, y_train, batch)
        for x_batch, y_batch in zip(x_batches, y_batches):
            params, opt_state, loss_value = step(
                params, opt_state, jnp.array(x_batch), jnp.array(y_batch)
            )

        train_fid_hist.append(-cost_encoding(params, X_train, y_train))
        test_fid_hist.append(-cost_encoding(params, X_test, y_test))

    return params


"""
fig, ax = qml.draw_mpl(circuit_encoding)(weights_encoder, y_train[0], X_train[0])
fig.show()
"""
print(
    -cost_encoding(weights_encoder, X_train, y_train),
    "encoder fidelity before train - train data",
)
print(
    -cost_encoding(weights_encoder, X_test, y_test),
    "encoder fidelity before train - test data",
)

optimizer = optax.adam(lr)
t1 = time.time()
weights_encoder = fit(weights_encoder, optimizer)

plt.plot(train_fid_hist, c="r")
plt.plot(test_fid_hist, c="b")
plt.legend(["Train", "Test"])
# plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Fidelity')
plt.savefig(
    f'e3_trained_{num_trash_bits}_{num_data_bits}_{num_entangler_bits}_{num_layers}_trash_fidelity_training.png')
plt.title("Trash fidelity")
plt.show()

print(time.time() - t1, "seconds - training time")


def get_fid_scores_encoded(params, X, Y):
    return jax.vmap(encoding_fidelity, in_axes=[None, 0, 0])(params, Y, X)


fid_scores_train_encoded = get_fid_scores_encoded(weights_encoder, X_train, y_train)
fid_scores_test_encoded = get_fid_scores_encoded(weights_encoder, X_test, y_test)

print(
    jnp.mean(fid_scores_train_encoded),
    "encoder fidelity after train - train data",
)
print(
    jnp.mean(fid_scores_test_encoded),
    "encoder fidelity after train - test data",
)
plt.scatter(
    y=fid_scores_train_encoded[:train_digits_each],
    x=range(train_digits_each),
    marker="*",
    c="r",
)
plt.scatter(
    y=fid_scores_train_encoded[train_digits_each:],
    x=range(train_digits_each, 2 * train_digits_each),
    marker="+",
    c="r",
)
plt.scatter(
    y=fid_scores_test_encoded[:test_digits_each],
    x=range(2 * train_digits_each, 2 * train_digits_each + test_digits_each),
    marker="*",
    c="b",
)
plt.scatter(
    y=fid_scores_test_encoded[test_digits_each:],
    x=range(
        2 * train_digits_each + test_digits_each,
        2 * train_digits_each + 2 * test_digits_each,
    ),
    marker="+",
    c="b",
)
plt.legend(["Train 0", "Train 1", "Test 0", "Test 1"], loc=3)
plt.title("Trash fidelity after encoder")
plt.ylabel('Fidelity')
plt.savefig(f'e3_trained_{num_trash_bits}_{num_data_bits}_{num_entangler_bits}_{num_layers}_trash_fidelity.png')
plt.show()
print("*" * 50)

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
    theta0_encode = weights_encoder[: -num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1_encode = weights_encoder[-num_trash_bits * 2:].reshape([1, -1])
    theta0_decode = params[: -num_trash_bits * 2].reshape([num_layers, 2, -1])
    theta1_decode = params[-num_trash_bits * 2:].reshape([1, -1])

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


cost_pre = fid_score(weights_decoder, X_train, y_train)
print(-cost_pre, "decoder fidelity before train - train data")
print(
    -fid_score(weights_decoder, X_test, y_test),
    "decoder fidelity before train - test data",
)

train_fid_hist = []
test_fid_hist = []


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        loss_value, grads = jax.value_and_grad(fid_score)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        x_batches, y_batches = iterate_batches(X_train, y_train, batch)
        for x_batch, y_batch in zip(x_batches, y_batches):
            params, opt_state, loss_value = step(
                params, opt_state, jnp.array(x_batch), jnp.array(y_batch)
            )

        train_fid_hist.append(-fid_score(params, X_train, y_train))
        test_fid_hist.append(-fid_score(params, X_test, y_test))

    return params


optimizer = optax.adam(lr)
t1 = time.time()
weights_decoder = fit(weights_decoder, optimizer)

plt.plot(train_fid_hist)
plt.plot(test_fid_hist)
plt.legend(["Train", "Test"])
# plt.yscale('log')
plt.title("Decoder Fidelity")
plt.ylabel('Fidelity')
plt.xlabel('Epochs')
plt.savefig(
    f'e3_trained_{num_trash_bits}_{num_data_bits}_{num_entangler_bits}_{num_layers}_decoder_fidelity_training.png')
plt.show()

print(time.time() - t1, "seconds - training time")


def get_fid_scores(params, X, Y):
    return jax.vmap(check_fidelity, in_axes=[None, 0, 0])(params, Y, X)


fid_scores_train = get_fid_scores(weights_decoder, X_train, y_train)
fid_scores_test = get_fid_scores(weights_decoder, X_test, y_test)
print(jnp.mean(fid_scores_train), "decoeder fidelity after decoding - train data")
print(jnp.mean(fid_scores_test), "decoder fidelity after decoding - test data")

plt.scatter(
    y=fid_scores_train[:train_digits_each],
    x=range(train_digits_each),
    marker="*",
    c="r",
)
plt.scatter(
    y=fid_scores_train[train_digits_each:],
    x=range(train_digits_each, 2 * train_digits_each),
    marker="+",
    c="r",
)
plt.scatter(
    y=fid_scores_test[:test_digits_each],
    x=range(2 * train_digits_each, 2 * train_digits_each + test_digits_each),
    marker="*",
    c="b",
)
plt.scatter(
    y=fid_scores_test[test_digits_each:],
    x=range(
        2 * train_digits_each + test_digits_each,
        2 * train_digits_each + 2 * test_digits_each,
    ),
    marker="+",
    c="b",
)
plt.title("Final fidelity -Decoded state with initial state")
plt.legend(["Train 0", "Train 1", "Test 0", "Test 1"], loc=3)
plt.ylabel('Fidelity')
plt.savefig(f'e3_trained_{num_trash_bits}_{num_data_bits}_{num_entangler_bits}_{num_layers}_final_fidelity.png')
plt.show()
"""
fig, ax = qml.draw_mpl(circuit_decoding)(weights_decoder, y_train[0], X_train[0])
fig.show()
"""
print(PARAMETERS)
