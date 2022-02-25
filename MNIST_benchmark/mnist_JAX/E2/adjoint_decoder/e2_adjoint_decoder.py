import itertools
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
X_test = jnp.concatenate(
    [
        digits_zero[train_digits_each: train_digits_each + test_digits_each],
        digits_one[train_digits_each: train_digits_each + test_digits_each],
    ],
    axis=0,
)

num_weights = num_layers * (
        15 * (num_wires + num_entangler_bits) * ((num_wires + num_entangler_bits) - 1) // 2
)

weights_encoder = np.random.uniform(-np.pi, np.pi, num_weights, requires_grad=True)
weights_encoder = jnp.array(weights_encoder)


def layer(params, trash, data, entangler):
    wires = trash + data + entangler
    idx = 0

    wire_list = itertools.combinations(wires, 2)

    for [i, j] in wire_list:
        qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
        qml.Rot(params[idx + 3], params[idx + 4], params[idx + 5], wires=j)
        qml.CNOT(wires=[j, i])
        qml.RZ(params[idx + 6], wires=i)
        qml.RY(params[idx + 7], wires=j)
        qml.CNOT(wires=[i, j])
        qml.RY(params[idx + 8], wires=j)
        qml.CNOT(wires=[j, i])
        qml.Rot(params[idx + 9], params[idx + 10], params[idx + 11], wires=i)
        qml.Rot(params[idx + 12], params[idx + 13], params[idx + 14], wires=j)

        idx += 15


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

    return params


"""
fig, ax = qml.draw_mpl(circuit_encoding)(weights_encoder, X_train[0])
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

plt.plot(train_fid_hist, c="r")
plt.plot(test_fid_hist, c="b")
plt.legend(["Train", "Test"])
plt.xlabel('Epochs')
plt.ylabel('Fidelity')
# plt.yscale('log')
plt.title("Trash fidelity")
plt.savefig(
    f'e2_adjoint_{num_trash_bits}_{num_data_bits}_{num_entangler_bits}_{num_layers}_trash_fidelity_training.png')
plt.show()

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
plt.savefig(
    f'e2_adjoint_{num_trash_bits}_{num_data_bits}_{num_entangler_bits}_{num_layers}_trash_fidelity.png')
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


@jax.jit
@qml.qnode(dev2, interface="jax")
def circuit_decoding(params, state):
    params = params.reshape([num_layers, -1])
    qml.AmplitudeEmbedding(
        state, wires=trash_bits_encoding + data_bits_decoding, normalize=True
    )

    make_entangled_bits(entangler_bits_decoding)

    U(
        params,
        trash_bits_encoding,
        data_bits_decoding,
        entangler_bits_decoding[::2],
    )

    qml.Barrier(wires=range(num_wires + num_trash_bits + num_entangler_bits))

    qml.adjoint(U)(
        params,
        trash_bits_decoding,
        data_bits_decoding,
        entangler_bits_decoding[1::2],
    )

    return qml.density_matrix(wires=trash_bits_decoding + data_bits_decoding)


def check_fidelity(params, state):
    original = get_init_states(state)
    decoded = circuit_decoding(params, state)
    fid = jnp.dot(jnp.conj(original), jnp.dot(decoded, original))
    return jnp.real(fid)


def get_fid_scores(params, X):
    return jax.vmap(check_fidelity, in_axes=[None, 0])(params, X)


fid_scores_train = get_fid_scores(weights_encoder, X_train)
fid_scores_test = get_fid_scores(weights_encoder, X_test)
print(jnp.mean(fid_scores_train), "decoder fidelity after decoding - train data")
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
plt.savefig(f'e2_adjoint_{num_trash_bits}_{num_data_bits}_{num_entangler_bits}_{num_layers}_final_fidelity.png')
plt.show()
"""
fig, ax = qml.draw_mpl(circuit_decoding)(weights_encoder, X_train[0])
fig.show()
"""
print(PARAMETERS)
