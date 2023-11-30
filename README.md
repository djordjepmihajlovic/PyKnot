# PyKnot for Machine Learned, Generative Knots.

### PyTorch MLKnotsProject code for TAPLAB 

PyKnot is a machine learning (ML) package with PyTorch and PyTorch.Lightning dependency to investigate the classification of mathematical knots. The package offers ML models to predict the classification of knots upto x crossings, and uses generative ML techniques to create new knots.
It's in studying this generation that we hope to investigate the methods by which the ML method is learning to classify knots, hopefully teaching us about hidden invariants beyond those already known.

The PyTorch version here is based off of an original Tensorflow code developed by TAPLAB.

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




