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
num_trash_bits = 2
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


def find_strongest_correlations(dataframe, qubits):
    class_correlations = dataframe.loc["Class", :]
    class_correlations = class_correlations.drop(index="Class")

    feature_list = list(class_correlations.index)
    correlation_list = [class_correlations[x] for x in feature_list]

    features = []
    correlations = []

    for i in range(int(qubits / 2)):
        correlations.append(max(correlation_list))
        features.append(feature_list[correlation_list.index(max(correlation_list))])

        del feature_list[correlation_list.index(max(correlation_list))]
        del correlation_list[correlation_list.index(max(correlation_list))]

        correlations.append(min(correlation_list))
        features.append(feature_list[correlation_list.index(min(correlation_list))])

        del feature_list[correlation_list.index(min(correlation_list))]
        del correlation_list[correlation_list.index(min(correlation_list))]

    return features, correlations


# CREDIT CARD DATA

df = pd.read_csv('creditcard.csv')

new_time = [0]
for t in range(1, len(df.Time)):
    new_time.append(df.Time[t] - df.Time[t - 1])
df["Time"] = new_time

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

df_new = df

df_0 = df_new[df_new['Class'] == 0]
df_1 = df_new[df_new['Class'] == 1]
df_corr = df_new.corr()
feature_list, correlations = find_strongest_correlations(df_corr, 8)
illegal = df_1[feature_list]
legal = df_0[feature_list]

# num of legal transactions
df_0 = legal.iloc[0:1000]

std_scaler = StandardScaler()
X = df_0.to_numpy()
data_illegal = jnp.array(illegal.to_numpy())

X_train, X_test = train_test_split(X, test_size=0.3, random_state=seed)


X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

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
