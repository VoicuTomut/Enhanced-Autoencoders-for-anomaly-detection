from jax.config import config

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

"""
df = pd.read_csv('creditcard.csv')

# time -> time from previous transaction
new_time = [0]
for t in range(1, len(df.Time)):
    new_time.append(df.Time[t] - df.Time[t - 1])
df["Time"] = new_time

## engineering two new features to have 32 feutures that can be encoded om 5 qubits.
over_average = []
under_average = []
mean = {}
std = {}
for col in df:
    if col not in ["Class"]:
        mean[col] = df[col].mean()
        std[col] = df[col].std()

for index, row in df.iterrows():
    o_average = 0
    u_average = 0
    for col in df:
        if col not in ['Class']:
            if row[col] > mean[col] + 2 * std[col]:
                o_average = o_average + 1
            if row[col] < mean[col] + 2 * std[col]:
                u_average = u_average + 1

    over_average.append(o_average)
    under_average.append(u_average)

df["over_average"] = over_average
df["under_average"] = under_average

df.to_csv('creditcard_updated.csv', index=False)
"""
df_new = pd.read_csv('creditcard_updated.csv')

df_0 = df_new[df_new['Class'] == 0]
df_1 = df_new[df_new['Class'] == 1]

df_0 = df_0.iloc[0:1000]

std_scaler = StandardScaler()
Y = df_0['Class'].to_numpy() + 1
X = df_0.drop(['Class'], axis=1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, stratify=Y, random_state=seed)

X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

data_illegal = jnp.array(df_1.drop(['Class'], axis=1).to_numpy())
y_illegal = jnp.ones(len(data_illegal))

num_trash_bits = 2
num_data_bits = 3
num_entangler_bits = 0
num_layers = 10
batch = 700
steps = 1000
lr = 0.01

PARAMETERS = {
    "Trash Bits": num_trash_bits,
    "Data Bits": num_data_bits,
    "EPR Pairs": num_entangler_bits // 2,
    "Layers": num_layers,
    "Training Steps": steps,
    "Batch Size": batch,
    "Learning Rate": lr,
}

num_wires = num_trash_bits + num_data_bits
trash_bits_encoding = list(range(num_trash_bits))
data_bits_encoding = list(range(num_trash_bits, num_wires))
entangler_bits_encoding = list(range(num_wires, num_wires + num_entangler_bits))

X_train, X_test, y_train, y_test = jnp.array(X_train), jnp.array(X_test), jnp.array(y_train), jnp.array(y_test)

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

    costs_train = []
    costs_test = []
    for i in range(steps):
        subset = np.random.choice(list(range(len(X_train))), batch, replace=False)
        x_batch = X_train[subset]
        y_batch = y_train[subset]
        params, opt_state, loss_value = step(params, opt_state, x_batch, y_batch)
        costs_train.append(-loss_value)

        if i % 5 == 0:
            train_fid_hist.append(-cost_encoding(params, X_train, y_train))
            test_fid_hist.append(-cost_encoding(params, X_test, y_test))
            illegal_fid.append(-cost_encoding(params, data_illegal, y_illegal))

    return params, costs_train, costs_test


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
(
    weights_encoder,
    costs_train,
    costs_test,
) = fit(weights_encoder, optimizer)

plt.plot(costs_train, c="r")
# plt.yscale('log')
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

print(PARAMETERS)

plt.plot([x for x in range(0, len(train_fid_hist) * 5, 5)], np.array(train_fid_hist), label="train fid")
plt.plot([x for x in range(0, len(test_fid_hist) * 5, 5)], np.array(test_fid_hist), label="test fid")
plt.plot([x for x in range(0, len(illegal_fid) * 5, 5)], np.array(illegal_fid), label="illegal fid")

plt.legend()
plt.title("compression fidelity", )
plt.xlabel("Steps")
plt.ylabel("fid")
plt.show()
print("fidelity:", train_fid_hist[-1])

# CLASSIFIER
legal_flist = fid_scores_train_encoded.tolist() + fid_scores_test_encoded.tolist()

print(min(legal_flist))
print(max(legal_flist))

illegal_flist = get_fid_scores_encoded(weights_encoder, data_illegal, y_illegal).tolist()

print(min(illegal_flist))
print(max(illegal_flist))

plt.hist(illegal_flist, bins=100, label="illegal", color="skyblue", alpha=0.4)
plt.hist(legal_flist, bins=100, label="legal", color="red", alpha=0.4)
plt.title("Compression fidelity", )
plt.legend()
plt.show()
