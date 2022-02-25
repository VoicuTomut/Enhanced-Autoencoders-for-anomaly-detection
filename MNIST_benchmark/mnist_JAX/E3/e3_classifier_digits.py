from jax.config import config
from sklearn import datasets

config.update("jax_enable_x64", True)
import pandas as pd
import pennylane.numpy as np
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
import optax
import time
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

num_trash_bits = 5
num_data_bits = 1
num_entangler_bits = 0
num_layers = 1
batch = 10
epochs = 100
lr = 0.01

PARAMETERS = {
    "Trash Bits": num_trash_bits,
    "Data Bits": num_data_bits,
    "EPR Pairs": num_entangler_bits // 2,
    "Layers": num_layers,
    "Training Epochs": epochs,
    "Batch Size": batch,
    "Learning Rate": lr,
}

digits = datasets.load_digits(n_class=2)
data = digits["data"]
labels = digits["target"]
digits_zero = data[labels == 0]
digits_one = data[labels == 1]
X = digits_zero
Y = np.ones(len(X))
data_illegal = jnp.array(digits_one)
y_illegal = jnp.ones(len(data_illegal))

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=seed
)

num_wires = num_trash_bits + num_data_bits
trash_bits_encoding = list(range(num_trash_bits))
data_bits_encoding = list(range(num_trash_bits, num_wires))
entangler_bits_encoding = list(range(num_wires, num_wires + num_entangler_bits))

X_train, X_test, y_train, y_test = (
    jnp.array(X_train),
    jnp.array(X_test),
    jnp.array(y_train),
    jnp.array(y_test),
)

num_weights = (
        num_layers * (num_wires + num_entangler_bits // 2) * 2 * 2 + num_trash_bits * 2
)

weights_encoder = np.random.uniform(-np.pi, np.pi, num_weights, requires_grad=True)
weights_encoder = jnp.array(weights_encoder)


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
illegal_fid = []


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
        illegal_fid.append(-cost_encoding(params, data_illegal, y_illegal))

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

print(PARAMETERS)
plt.plot(train_fid_hist, label="train class 0 fid")
plt.plot(test_fid_hist, label="test class 0 fid")
plt.plot(illegal_fid, label="class 1 fid")
plt.legend()
plt.title("Encoder fidelity")
plt.xlabel("Epochs")
plt.ylabel("Fidelity")
plt.show()
print("fidelity:", train_fid_hist[-1])

# CLASSIFIER
legal_flist = fid_scores_train_encoded.tolist() + fid_scores_test_encoded.tolist()

print(min(legal_flist))
print(max(legal_flist))

illegal_flist = get_fid_scores_encoded(weights_encoder, data_illegal, y_illegal).tolist()

print(min(illegal_flist))
print(max(illegal_flist))

plt.hist(illegal_flist, bins=100, label="Class 1", color="skyblue", alpha=0.4)
plt.hist(legal_flist, bins=100, label="Class 0", color="red", alpha=0.4)
plt.title("Classification")
plt.legend()
plt.show()

split = 0.53

print("split:", split)
class_1 = []
for i in illegal_flist:
    if i < split:
        class_1.append(1)
    else:
        class_1.append(0)
class_1_acc = sum(class_1) / len(class_1)
print("class 1 classification accuracy:", class_1_acc)
class_0 = []
for i in legal_flist:
    if i > split:
        class_0.append(1)
    else:
        class_0.append(0)
class_0_acc = sum(class_0) / len(class_0)
print("class 0 classification accuracy:", class_0_acc)
t_ac = (sum(class_1) + sum(class_0)) / (len(class_1) + len(class_0))
print("total accuracy:", t_ac)
