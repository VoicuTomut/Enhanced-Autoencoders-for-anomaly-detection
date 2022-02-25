
## QENCODE

This module is part of the [Enhanced-Autoencoders-for-anomaly-detection
Public](https://github.com/VoicuTomut/Enhanced-Autoencoders-for-anomaly-detection) repository. 
This module aims to keep all the pieces of an autoencoder easy to connect with each other by using QubitsArrangement class.
It also provides a range of: initializers, encoder, and decoder circuits that we implemented using [Pennylane](https://pennylane.ai/).

Code structure:
------------

    │
    ├── decoders                                               
    │   ├── base.py                                        <- basic decoder
    │   ├── classic_parametrized_decoder.py        	       <- an example of a parametrized decoder
    │   └── ent_assist_decoder.py                          <- entanglement assisted decoder    
    │
    ├── encoders 
    │   ├── base.py                                         <- encoder circuits from:https://arxiv.org/abs/1612.02806
    │   ├── enhance_e3.py                                   <- encoder circuit inspired  from the: https://arxiv.org/abs/2010.06599   
    │   ├── ent_assist_encode.py                            <- entanglement assisted encoder with no interaction
    │   ├── ent_assist_encode_sean.py
    │   ├── ent_zoom_e4.py                                  <- An encoder thet darst extens the feuture space then compress it.
    │   └── patched_autoencoder_e5.py                       <- a patch encoder 
    │
    ├── initialize                       			
    │   └── base.py                                	         <- colection of pennylane initializers addapted to our project 
    │
    ├── training_circuits ( a coection of helpful quantum circuits)                                                
    │   └── swap_test.py                                    <- Swap test circuit      
    │
    ├── utils (utility function for implementing examples) 
    │   └── mnist.py 					                    <- function to import nmist data set
    │   
    ├── qubits_arrangement.py                       	    <- we use this class to keep our qubit register in order
    │   
    ├── requirements.txt                                    <-  requirements for qencode module
    │
    └──README.md                                            <- qencode README .
    


--------
