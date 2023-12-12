# PyKnot for Machine Learned, Generative Knots & Knot Polynomials.

### PyTorch MLKnotsProject code for TAPLAB 

PyKnot is a machine learning (ML) package with PyTorch and PyTorch.Lightning dependency to investigate the classification of mathematical knots. The package offers ML models to predict the classification of knots upto x crossings, and uses generative ML techniques to create new knots.
It's in studying this generation that we hope to investigate the methods by which the ML method is learning to classify knots, hopefully teaching us about hidden invariants beyond those already known.

The PyTorch version here is based off of an original Tensorflow code developed by TAPLAB.

## Running the code

To run a simple, class based knot prediction based on some input data (eg. SIGWRITHE (StA)) run main.py with the required arguments, for example:

```
    python main.py -modtyp NN -pred class -p 5Class -m train -d SIGWRITHE -t FFNN 
```
outputs the result of a neural network classification problem on the 5 class set (knots: 0_1, 3_1, 4_1, 5_1, 5_2) trained on segment to all (StA) data using a feedforward neural network

The argument parameters are as follows:

-modtyp: takes in the required machine learning model.
Options: NN, DT, LR
-pdct: takes in the prediction being made. "class" arguments correspond to classification of knots, whereas other invariants/codes can be specified. For example, -pred dowker will return a training on dowker code prediction capability.
Options: class, dowker, jones, quantumA2
-p: 
-m: 
-d:
-t: 


## +  Additions :

The following additions have been implemented into the existing framework in the new PyTorch version:

* Prediction (-pred): new code capability to compute other prediction types other than standard class predicition, such as XYZ coordinate to SIGWRITHE predicitions
    * "std" (standard classification prediction) - [x] 
    * "dual" ('dual' classification prediction e.g. StA2XYZ) - [x]

* Generative models (nn_generative_models.py): Torch.nn generative modules
    * VariationalAutoencoder(pl.LightningModule) - [x]
    * Autoencoder(pl.LightningModule) - [x]

* ml_models.py: other machine learning methods
    * Decision tree - [-]
    * Logistic Regression - [-]
    * Linear Regression - [-]

## - Removals :

Currently, not implemented into the PyTorch version:

* DENSXSIGWRITHE (Data) - [-]
* RNN2, 2b (network) - [-]
* FFNN2 (network) - [-]
* localise_* (network) - [-]
* randFOR (network) - [-]

## = Difference :

The following methods are implemented differently to the TensorFlow base code and are subject to change:

*Currently no separate train/test functionality; testing occurs immediately after training


## ! W.I.P :

* Implementation of XYZ to StA predicition & vice versa.
     Model | XYZ2StA  | StA2XYZ
    ------| ------------- | -------------
    FFNN | - [x] | - [x]
    RNN  | - [x] | - [x]
    CNN  | - [ ] | - [ ]

      >> Current implementation using FFNN however results are bad - can achieve a general shape but far off from true prediction. 

* Generative models.
    * VAE
    *     >> VAE current implementation works, however the latent space is still somewhat entangled -> main work is to generate a disentangled latent space.
        * ! WIP (StA complete, working on XYZ) --> disentangling the latent space: possible avenues
            * beta-VAE - [x]
            * CNN Encoder - [ ]
            * LSTM Encoder - [x]
            * Attention mechanisms - [ ]  

* Model analysis.
*     >> These need to be fully understood to make meaningful analysis! 
    * SHAP - [ ]
    * LIME - [ ]
    * Partial Dependence Plots - [ ] 




