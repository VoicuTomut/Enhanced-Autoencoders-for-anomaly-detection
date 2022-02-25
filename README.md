# Entanglement-assisted quantum autoencoders (EAQAE)

## Team: Samras
#### Members:  Michał Bączyk, Stephen DiAdamo, Sean Mcilvane, Eraraya Ricardo Muten, Ankit Khandelwal, Andrei Tomut, and Renata Wong. 

This repository is composed of two parts: 

1) Testing autoencoding strategies for anomaly detection specifically for two datasets - breast cancer and credit card fraud.
2) Testing how entanglement resources can be added to autoencoders to enhance their encoding and decoding ability.


The preliminary results for 1) are presented in Use-case_Cancer_detection, Use-case_Fraud_detection, respectively. For that, we follow the approach developed in https://arxiv.org/pdf/2112.08869.pdf. To speed up the training for the financial data, we used Jax to multi-process the optimization step.

For testing with entanglement, we try various methods of encoding and compare the results against schemes without entanglement. This part of the project is under ongoing investigation but our progress can be found in the ... folders.
