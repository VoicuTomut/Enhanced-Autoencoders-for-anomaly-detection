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

# PARAMETERS
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

# DIGITS DATASET
digits = datasets.load_digits(n_class=2)
data = digits["data"]
labels = digits["target"]
digits_zero = data[labels == 0]
digits_one = data[labels == 1]
X = digits_zero
data_illegal = jnp.array(digits_one)

X_train, X_test = train_test_split(X, test_size=0.3, random_state=seed)

num_wires = num_trash_bits + num_data_bits
trash_bits_encoding = list(range(num_trash_bits))
data_bits_encoding = list(range(num_trash_bits, num_wires))
entangler_bits_encoding = list(range(num_wires, num_wires + num_entangler_bits))

X_train, X_test = jnp.array(X_train), jnp.array(X_test)

num_weights = num_layers * (
        2 * 3 * (num_wires + num_entangler_bits)
        + 3 * ((num_wires + num_entangler_bits) - 1) * (num_wires + num_entangler_bits)
)

weights_encoder = np.random.uniform(-np.pi, np.pi, num_weights, requires_grad=True)
weights_encoder = jnp.array(weights_encoder)


def layer(params, trash, data, entangler):
    wires = trash + data + entangler
    # Add the first rotational gates:
    idx = 0
    for i in wires:
        qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
        idx += 3

    # Add the controlled rotational gates
    for i in wires:
        for j in wires:
            if i != j:
                qml.CRot(params[idx], params[idx + 1], params[idx + 2], wires=[i, j])
                idx += 3

    # Add the last rotational gates:
    for i in wires:
        qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
        idx += 3


def make_entangled_bits(entangler):
    for i, j in zip(entangler[::2], entangler[1::2]):
        qml.Hadamard(i)
        qml.CNOT(wires=[i, j])


def U(params, trash, data, entangler):
    for l in range(num_layers):
        layer(params[l], trash, data, entangler)


def iterate_batches(X, batch_size):
    global key
    key, subkey = jax.random.split(key)
    X = jax.random.shuffle(subkey, X)
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
    return batch_list_x


dev0 = qml.device("default.qubit.jax", wires=num_wires + num_entangler_bits, shots=None)
zero_state = jnp.zeros([2 ** num_trash_bits])
zero_state = zero_state.at[0].set(1)


@jax.jit
@qml.qnode(dev0, interface="jax")
def circuit_encoding(params, state):
    params = params.reshape([num_layers, -1])
    qml.AmplitudeEmbedding(state, wires=range(num_wires), normalize=True)

    make_entangled_bits(entangler_bits_encoding)

    U(
        params,
        trash_bits_encoding,
        data_bits_encoding,
        entangler_bits_encoding[::2],
    )

    return qml.density_matrix(wires=range(num_trash_bits))


def encoding_fidelity(params, state):
    trash = circuit_encoding(params, state)
    fid = jnp.dot(jnp.conj(zero_state), jnp.dot(trash, zero_state))
    return jnp.real(fid)


def cost_encoding(params, X):
    return -jnp.mean(jax.vmap(encoding_fidelity, in_axes=[None, 0])(params, X))


train_fid_hist = []
test_fid_hist = []
illegal_fid = []


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X):
        loss_value, grads = jax.value_and_grad(cost_encoding)(params, X)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        x_batches = iterate_batches(X_train, batch)
        for x_batch in x_batches:
            params, opt_state, loss_value = step(params, opt_state, jnp.array(x_batch))

        train_fid_hist.append(-cost_encoding(params, X_train))
        test_fid_hist.append(-cost_encoding(params, X_test))
        illegal_fid.append(-cost_encoding(params, data_illegal))

    return params


"""
fig, ax = qml.draw_mpl(circuit_encoding)(weights_encoder,  X_train[0])
fig.show()
"""
print(
    -cost_encoding(weights_encoder, X_train),
    "encoder fidelity before train - train data",
)
print(
    -cost_encoding(weights_encoder, X_test),
    "encoder fidelity before train - test data",
)

optimizer = optax.adam(lr)
t1 = time.time()
weights_encoder = fit(weights_encoder, optimizer)

print(time.time() - t1, "seconds - training time")


def get_fid_scores_encoded(params, X):
    return jax.vmap(encoding_fidelity, in_axes=[None, 0])(params, X)


fid_scores_train_encoded = get_fid_scores_encoded(weights_encoder, X_train)
fid_scores_test_encoded = get_fid_scores_encoded(weights_encoder, X_test)

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
plt.title(
    "Encoder Fidelity",
)
plt.xlabel("Epochs")
plt.ylabel("Fidelity")
plt.show()
print("fidelity:", train_fid_hist[-1])

# CLASSIFIER
legal_flist = fid_scores_train_encoded.tolist() + fid_scores_test_encoded.tolist()

print(min(legal_flist))
print(max(legal_flist))

illegal_flist = get_fid_scores_encoded(weights_encoder, data_illegal).tolist()

print(min(illegal_flist))
print(max(illegal_flist))

plt.hist(illegal_flist, bins=100, label="class 1", color="skyblue", alpha=0.4)
plt.hist(legal_flist, bins=100, label="class 0", color="red", alpha=0.4)
plt.title("Classification")
plt.xlabel('Fidelity')
plt.ylabel('Count')
plt.legend()
plt.show()

split = 0.67

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
