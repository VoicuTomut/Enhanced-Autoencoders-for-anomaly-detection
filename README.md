# Entanglement-assisted quantum autoencoders (EAQAE) - Samras
#temporaryRepoQhack2022 - power up entry

We deploy quantum autoencoders for anomaly detection tasks for two datasets - breast cancer and credit card fraud. The preliminary results are presented in Use-case_Cancer_detection, Use-case_Fraud_detection, respectively. 
For that, we follow the approach developed in https://arxiv.org/pdf/2112.08869.pdf. To speed up the training for the financial data, we used Jax.

We also want to test how an entanglement-assisted quantum autoencoder will perform for the same tasks. We have already developed necessary architectured which you can find in qencode - trained examples will be added soon.


Code structure:
------------

│
├── MNIST_benchmark   
│   ├ Here, we keep our experiments with the MNIST data set for benchmark and comparison with past paper implementation.
│   │ benchmark and comparison with past paper implementation.
│   │
│   ├── best results 
│   ├──                           
│   └──  
│
├── Use-case_Cancer_detection
│   ├We used and applied the Quantum autoencoder for anomaly detection in order to identify the Bening cels from the Kaggle
│   │dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/discussion . 
│   │
│   ├── best results: 
│   ├──                           
│   └── 
│
│
├── Use-case_Fraud_detection
│   ├We used and applied the Quantum autoencoder for anomaly detection on the Kaggle dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud. ) 
│   │that contain card transaction to spot the fraudulent transactions.And we get decent results. 
│   │
│   ├── best results: 
│   ├──                           
│   └── 
│
├── qencode                                               
│   └── This module aims to keep all the pieces of an autoencoder easy to connect with each other by using QubitsArrangement class. It also provides a range of: initializers, encoder, and decoder circuits that we implemented using Pennylane. 
│
├── 
│   
├── LICENSE
│   
├── requirements.txt
│   
└──README.md                                            <- project README

--------
