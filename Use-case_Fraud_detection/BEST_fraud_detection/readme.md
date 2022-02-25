#Credit Card Fraud - A Growing Issue

Credit card fraud is a growing issue with $28.5 billion lost globally due to credit card fraud in 2020 and is expected to exceed $49 billion by 2030 (1)). In 2020, around 1 out of 4 digital interactions were credit card fraud attempts (1). Since there are so many non fraudulent transactions, it is challenging to detect the fraudulent transactions. In this notebook, we will be using a quantum auto-encoder to perform anomaly detection.

We can use the quantum auto encoder to encode the 4 qubit state into a 3 qubit state and then use a decoder to decode the 3 qubit state back into 4 qubit state. The quantum auto encoder is trained on the normal dataset (or in this case the non fraudulent transactions) which means the quantum autoencoder can encode and decode a normal (non fraudulent transaction) with high fidelity. The quantum autoencoder will not be able to encode and decode an anomaly (fraudulent transaction) with high fidelity. So, to tell if a data point is an anomaly, running it through the quantum auto encoder should output a dissimilar state.

We use the credit card transaction data set from https://www.kaggle.com/mlg-ulb/creditcardfraud. It contains over 280,000 credit card transactions with 30 features and a class for each transaction of either 0 or 1 which represent a non fraudulent transaction and a fraudulent transaction, respectively. 

For this, we also used 3 different encoders

e2 - Detailed in Figure 3a in https://arxiv.org/pdf/1612.02806.pdf

e3 - Detailed in Figure 3 in https://arxiv.org/pdf/2010.06599.pdf

e5 - Is inspired by https://arxiv.org/pdf/2112.12563.pdf but a different ansatz is used

We also ran this on the pennylane "default.qubit" device, IBM's 16 qubit Guadalupe through Qiskit, and Rigetti's Aspen-11 QPU through Amazon Braket. 

The main idea of the project is to used amplitude encoding to encode data of a transaction onto 4 qubits. Then we train a quantum auto encoder to be able to compress the non fraudulent transactions into 3 qubits with high fidelity so that we can distinguish between a non fraudulent transaction and a fraudulent transaction. A fraudulent transaction would not be encoded into 3 qubits well so that when it is decoded, the quanutm state will be different than the input state and we will know that it is a fraudulent transaction.


