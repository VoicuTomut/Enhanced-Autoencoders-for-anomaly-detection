# Enhanced Autoencoders for Anomaly Detection

## Team: Samras
#### Members:  Michał Bączyk, Stephen DiAdamo, Sean Mcilvane, Eraraya Ricardo Muten, Ankit Khandelwal, Andrei Tomut, and Renata Wong.

> ### Our Final Presentation Notebook, presenting clearly all the methods and results, can be found [here](https://colab.research.google.com/drive/1qE2KCy4SBKtLRlL55SCjvZUf4IZaC2-y?usp=sharing).
> ### Note that all the codes and notebooks used to produce the results in that notebook are in this repository.

This repository is composed of two parts: 

1) Testing autoencoding strategies for anomaly detection specifically for two datasets - breast cancer and credit card fraud.
2) Testing how entanglement resources can be added to autoencoders to enhance their encoding and decoding ability.


The preliminary results for 1) are presented in Use-case_Cancer_detection and for 2) are presented in Use-case_Fraud_detection, respectively. For that, we follow the approach developed in https://arxiv.org/pdf/2112.08869.pdf but explore a lot more approaches than the original anomaly detection paper; by also implementing an [enhanced autoencoder](https://arxiv.org/pdf/2010.06599.pdf) and a [patch autoencoder](https://arxiv.org/pdf/2112.12563.pdf) that were not previously used for anomaly detection but some improvements (for example connecting the encoder circuits).

Our results show high accuracy for the experiments on the MNIST data sets, and for cancer detection, we observe an improvement using our version of the patch encoder. To speed up the training for the financial data, we used Jax to multi-process the optimization step. To read more about our results please check: 

------------

### Results:

##### Cancer detection:
    Compression accuracy: 0.8609424846861528
    Classification:
        split: 0.845
        benign classification accuracy: 0.9230769230769231
        malign clasification accuracy: 0.9098360655737705
        total accuracy: 0.9157175398633257
 
Simulator:

![image](plots/cancer_detect_1.png)


Hardware:

![image](plots/cancer_detect_2.png)



##### Fraud detection:
    Compression accuracy: 0.9106666637654454
    Classification:
        split:0.75
        fraund classification accuracy: 0.83
        legal classification accuracy: 0.93
        total accuracy: 0.88

Results
![image](plots/fraud.png)



##### MNIST Classification:
    With E1:
        split: 0.67
        class 1 classification accuracy: 1.0
        class 0 classification accuracy: 0.9943820224719101
        total accuracy: 0.9972222222222222
    With E2:
        split: 0.67
        class 1 classification accuracy: 1.0
        class 0 classification accuracy: 0.9943820224719101
        total accuracy: 0.9972222222222222
    With E3:
        split: 0.53
        class 1 classification accuracy: 1.0
        class 0 classification accuracy: 0.949438202247191
        total accuracy: 0.975
![image](MNIST_benchmark/mnist_JAX/e1-e2-e3_classification.png)

    
------------
    ├──  EAQAE approaches
    │   ├ 3 conceptually different approaches presenting how etanglement might be used as a resource in training of QAEs.
    │   │
    │   ├── EAQAE 3-1-3; entangling directly encoder and decoder qubits; training both encoder and decoder.ipynb
    │   ├── EAQAE 4-1-4; 2 EPR pairs shared.ipynb
    │   └── EAQAE 4-1-4; entangling directly encoder and decoder qubits.ipynb
    │   
    │
    ├── MNIST_benchmark   
    │   ├ Here, we keep our experiments with the MNIST data set for benchmark and comparison with past paper implementation.
    │   │
    │   ├── mnist_JAX
    │   ├── six_one_six   
    │   ├── six_three_six
    │   ├── six_two_six
    │   └── results: mnist_JAX/digits data.xlsx   
    │
    ├── Use-case_Cancer_detection
    │   ├We used and applied the Quantum autoencoder for anomaly detection in order to identify the Bening cels from the Kaggle
    │   │dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/discussion . 
    │   │
    │   ├── best results: Cancer_encoder_e5-SelectedFeautures-ent.ipynb
    │   ├── hardware results: e5_real-ent.ipynb    Noise messes things on real hardware, we can correct and train on real hardware with more time to mitigate the error.                      
    │   └── hardware results: e5_real.ipynb    Maybe during an internship ;)
    │
    ├── Use-case_Fraud_detection
    │   ├We used and applied the Quantum autoencoder for anomaly detection on the Kaggle dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud. ) 
    │   │that contain card transaction to spot the fraudulent transactions.And we get decent results. 
    │   │
    │   ├── best results: BEST_fraud_detection ; QuantumCreditFraud-best_pre_Braket.ipynb
    │   └── hardware results: QuantumCreditFraud_BraketResults.ipynb                       
    │   
    │      
    │
    ├── qencode                                               
    │   └── This module aims to keep all the pieces of an autoencoder easy to connect with each other by using QubitsArrangement class. It also provides a range of: initializers, encoder, and decoder circuits that we implemented using Pennylane. 
    │
    │   
    ├── LICENSE
    │   
    ├── requirements.txt
    │   
    └──README.md                                            <- project README

Project done during QHack 2022. 

Submitted to the following challenges:

- Bio-QML Challenge
- Quantum Finance Challenge
- Amazon Braket Challenge
- IBM Qiskit Challenge
- Hybrid Algorithms Challenge


