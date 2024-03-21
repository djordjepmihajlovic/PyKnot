# PyKnot for Machine Learned Knots & Knot Polynomial.

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

-pdct: takes in the prediction being made. "class" arguments correspond to classification of knots, whereas other invariants/codes can be specified. For example, "dowker" will return a training on dowker code prediction capability.
Options: class, dowker, v2, v3, jones, quantumA2


## +  Additions :

The following additions have been implemented into the existing framework in the new PyTorch version:

* Prediction (-pdct): new code capability to compute other prediction types other than standard class predicition, such as XYZ coordinate to SIGWRITHE predicitions

* Generative models (nn_generative_models.py): Torch.nn generative modules
    * VariationalAutoencoder(pl.LightningModule)
    * Autoencoder(pl.LightningModule)

* Concept Bottle neck models (nn_concept_models.py): Torch.nn concept modules
    * ConceptNN(pl.LightningModule) 
    * OnlyConceptNN(pl.LightningModule)

* ml_models.py: other machine learning methods
    * Decision trees

## - Removals :

Currently, not implemented into the PyTorch version:

* DENSXSIGWRITHE (Data) - [-]
* RNN2, 2b (network) - [-]
* FFNN2 (network) - [-]
* localise_* (network) - [-]
* randFOR (network) - [-]
